# Databricks notebook source


# COMMAND ----------

import os
from typing import Generator
from uuid import uuid4

import mlflow
from databricks.sdk import WorkspaceClient
from databricks_ai_bridge import ModelServingUserCredentials
from databricks_langchain import ChatDatabricks
from databricks_langchain.genie import GenieAgent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph.state import CompiledStateGraph
from langchain.agents import create_agent
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    output_to_responses_items_stream,
    to_chat_completions_input,
)

# COMMAND ----------

# Resolve project root: __file__ is available in IDE, CWD is repo root in Databricks
if "__file__" in dir():
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
else:
    _project_root = os.getcwd()

config = mlflow.models.ModelConfig(
    development_config=os.path.join(_project_root, "config.yaml")
)

agent_config = config.get("agent_config")

LLM_ENDPOINT_NAME = agent_config.get("llm_endpoint_name")
GENIE_SPACE_ID = agent_config.get("genie_space_id")
GENIE_AGENT_NAME = agent_config.get("genie_agent_name")
GENIE_AGENT_DESCRIPTION = agent_config.get("genie_agent_description")
SYSTEM_PROMPT = agent_config.get("system_prompt")

llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)

# COMMAND ----------


def create_langchain_agent(user_client: WorkspaceClient = None) -> CompiledStateGraph:
    """Create a ReAct agent with GenieAgent as a tool."""
    genie_kwargs = dict(
        genie_space_id=GENIE_SPACE_ID,
        genie_agent_name=GENIE_AGENT_NAME,
        description=GENIE_AGENT_DESCRIPTION,
    )
    if user_client:
        genie_kwargs["client"] = user_client

    genie = GenieAgent(**genie_kwargs)

    @tool
    def query_customer_transactions(question: str) -> str:
        """Query the customer transactions Genie agent. Pass the user's question as-is."""
        result = genie.invoke({"messages": [HumanMessage(content=question)]})
        messages = result.get("messages", [])
        if messages:
            return messages[-1].content
        return str(result)

    return create_agent(
        model=llm,
        tools=[query_customer_transactions],
        system_prompt=SYSTEM_PROMPT,
    )

# COMMAND ----------

# Wrap as a ResponsesAgent for Model Serving


class LangGraphResponsesAgent(ResponsesAgent):
    def _initialize_agent(self) -> CompiledStateGraph:
        """Initialize the agent per-request for on-behalf-of user authentication.
        User identity is only available at query time, so OBO resources (e.g. GenieAgent)
        must be created here rather than at module load."""
        try:
            user_client = WorkspaceClient(
                credentials_strategy=ModelServingUserCredentials()
            )
        except Exception:
            user_client = None
        return create_langchain_agent(user_client=user_client)

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
                node_name = tuple(events.keys())[0]
                yield ResponsesAgentStreamEvent(
                    type="response.output_item.done",
                    item=self.create_text_output_item(
                        text=f"<name>{node_name}</name>", id=str(uuid4())
                    ),
                )
            if len(new_msgs) > 0:
                yield from output_to_responses_items_stream(new_msgs)

# COMMAND ----------

# Set up MLflow for tracing

mlflow.langchain.autolog()
AGENT = LangGraphResponsesAgent()
mlflow.models.set_model(AGENT)
