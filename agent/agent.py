# Databricks notebook source


# COMMAND ----------

import json
from typing import Generator, Literal
from uuid import uuid4

import mlflow
from databricks.sdk import WorkspaceClient
from databricks_ai_bridge import ModelServingUserCredentials
from databricks_langchain import (
    ChatDatabricks,
    DatabricksFunctionClient,
    UCFunctionToolkit,
    set_uc_function_client,
)
from databricks_langchain.genie import GenieAgent
from langchain_core.runnables import Runnable
from langchain.agents import create_agent
from langgraph.graph.state import CompiledStateGraph
from langgraph_supervisor import create_supervisor
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    output_to_responses_items_stream,
    to_chat_completions_input,
)
from pydantic import BaseModel

# COMMAND ----------

client = DatabricksFunctionClient()
set_uc_function_client(client)

# Load configuration
config = mlflow.models.ModelConfig(development_config="../config.yaml")

# COMMAND ----------

# Create your LangGraph Supervisor Agent

GENIE = "genie"


class ServedSubAgent(BaseModel):
    endpoint_name: str
    name: str
    task: Literal["agent/v1/responses", "agent/v1/chat", "agent/v2/chat"]
    description: str


class Genie(BaseModel):
    space_id: str
    name: str
    task: str = GENIE
    description: str


class InCodeSubAgent(BaseModel):
    tools: list[str]
    name: str
    description: str


TOOLS = []

# COMMAND ----------


def stringify_content(state):
    msgs = state["messages"]
    if isinstance(msgs[-1].content, list):
        msgs[-1].content = json.dumps(msgs[-1].content, indent=4)
    return {"messages": msgs}


def create_langgraph_supervisor(
    llm: Runnable,
    externally_served_agents: list[ServedSubAgent] = [],
    in_code_agents: list[InCodeSubAgent] = [],
    user_client: WorkspaceClient = None,
):
    agents = []
    agent_descriptions = ""

    # Process inline code agents
    for agent in in_code_agents:
        agent_descriptions += f"- {agent.name}: {agent.description}\n"
        uc_toolkit = UCFunctionToolkit(function_names=agent.tools)
        TOOLS.extend(uc_toolkit.tools)
        agents.append(create_agent(llm, tools=uc_toolkit.tools, name=agent.name))

    # Process served endpoints and Genie Spaces
    for agent in externally_served_agents:
        agent_descriptions += f"- {agent.name}: {agent.description}\n"
        if isinstance(agent, Genie):
            # to better control the messages sent to the genie agent, you can use the `message_processor` param: https://api-docs.databricks.com/python/databricks-ai-bridge/latest/databricks_langchain.html#databricks_langchain.GenieAgent
            genie_kwargs = dict(
                genie_space_id=agent.space_id,
                genie_agent_name=agent.name,
                description=agent.description,
            )
            if user_client:
                genie_kwargs["client"] = user_client
            genie_agent = GenieAgent(**genie_kwargs)
            genie_agent.name = agent.name
            agents.append(genie_agent)
        else:
            model = ChatDatabricks(
                endpoint=agent.endpoint_name, use_responses_api="responses" in agent.task
            )
            # Disable streaming for subagents for ease of parsing
            model._stream = lambda x: model._stream(**x, stream=False)
            agents.append(
                create_agent(
                    model,
                    tools=[],
                    name=agent.name,
                    post_model_hook=stringify_content,
                )
            )

    prompt = f"""{config.get("agent_config").get("system_prompt")}

    {agent_descriptions}"""

    return create_supervisor(
        agents=agents,
        model=llm,
        prompt=prompt,
        add_handoff_messages=False,
        output_mode="full_history",
    ).compile()

# COMMAND ----------

# Wrap LangGraph Supervisor as a ResponsesAgent


class LangGraphResponsesAgent(ResponsesAgent):
    def _initialize_agent(self) -> CompiledStateGraph:
        """Initialize the supervisor per-request for on-behalf-of user authentication.
        User identity is only available at query time, so OBO resources (e.g. GenieAgent)
        must be created here rather than at module load."""
        try:
            user_client = WorkspaceClient(
                credentials_strategy=ModelServingUserCredentials()
            )
        except Exception:
            user_client = None
        return create_langgraph_supervisor(
            llm, EXTERNALLY_SERVED_AGENTS, user_client=user_client
        )

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)

    def predict_stream(
        self,
        request: ResponsesAgentRequest,
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        agent = self._initialize_agent()
        cc_msgs = to_chat_completions_input([i.model_dump() for i in request.input])
        first_message = True
        seen_ids = set()

        # can adjust `recursion_limit` to limit looping: https://docs.langchain.com/oss/python/langgraph/GRAPH_RECURSION_LIMIT#troubleshooting
        for _, events in agent.stream({"messages": cc_msgs}, stream_mode=["updates"]):
            new_msgs = [
                msg
                for v in events.values()
                for msg in v.get("messages", [])
                if msg.id not in seen_ids
            ]
            if first_message:
                seen_ids.update(msg.id for msg in new_msgs[: len(cc_msgs)])
                new_msgs = new_msgs[len(cc_msgs) :]
                first_message = False
            else:
                seen_ids.update(msg.id for msg in new_msgs)
                node_name = tuple(events.keys())[0]  # assumes one name per node
                yield ResponsesAgentStreamEvent(
                    type="response.output_item.done",
                    item=self.create_text_output_item(
                        text=f"<name>{node_name}</name>", id=str(uuid4())
                    ),
                )
            if len(new_msgs) > 0:
                yield from output_to_responses_items_stream(new_msgs)

# COMMAND ----------

# Configure the Foundation Model and Serving Sub-Agents

agent_config = config.get("agent_config")

LLM_ENDPOINT_NAME = agent_config.get("llm_endpoint_name")
llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)

EXTERNALLY_SERVED_AGENTS = [
    Genie(
        space_id=agent_config.get("genie_space_id"),
        name=agent_config.get("genie_agent_name"),
        description=agent_config.get("genie_agent_description"),
    ),
]

# COMMAND ----------

# Set up MLflow for tracing

mlflow.langchain.autolog()
AGENT = LangGraphResponsesAgent()
mlflow.models.set_model(AGENT)
