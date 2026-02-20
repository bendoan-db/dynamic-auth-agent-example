# Databricks notebook source

# COMMAND ----------

import os

import gradio as gr
import mlflow
from databricks.sdk import WorkspaceClient

# Resolve project root: __file__ is available in IDE, CWD is repo root in Databricks
if "__file__" in dir():
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
else:
    _project_root = os.getcwd()

config = mlflow.models.ModelConfig(
    development_config=os.path.join(_project_root, "config.yaml")
)

SERVING_ENDPOINT_NAME = config.get("agent_config").get("serving_endpoint_name")
GENIE_SPACE_ID = config.get("agent_config").get("genie_space_id")
HOST = config.get("databricks_config").get("host")
CATALOG = config.get("databricks_config").get("catalog")
SCHEMA = config.get("databricks_config").get("schema")
TABLE = config.get("databricks_config").get("table")

auth_config = config.get("auth_config")
WAREHOUSE_ID = auth_config.get("warehouse_id")
SP_MAPPING_TABLE = auth_config.get("sp_mapping_table")
CLIENT_MAPPING_TABLE = auth_config.get("client_mapping_table")

# COMMAND ----------

w = WorkspaceClient()

# Cache of OBO tokens keyed by user_id -> WorkspaceClient authenticated as the SP
_sp_clients: dict[str, WorkspaceClient] = {}

# COMMAND ----------


def _execute_sql(statement: str) -> dict:
    """Execute a SQL statement via the warehouse and return the result."""
    from databricks.sdk.service.sql import StatementState

    response = w.statement_execution.execute_statement(
        warehouse_id=WAREHOUSE_ID,
        statement=statement,
        wait_timeout="30s",
    )
    if response.status and response.status.state == StatementState.FAILED:
        raise RuntimeError(f"SQL execution failed: {response.status.error}")
    return response


def use_credentials(user_id: str, client_id: str) -> str:
    """Look up or create a service principal for user_id and map it to client_id."""
    if not user_id or not client_id:
        return "Please provide both User ID and Client ID."

    # 1. Check if a service principal already exists for this user
    lookup = _execute_sql(
        f"SELECT service_principal_id FROM {SP_MAPPING_TABLE} "
        f"WHERE username = '{user_id}'"
    )

    sp_application_id = None
    sp_numeric_id = None
    if lookup.result and lookup.result.data_array and len(lookup.result.data_array) > 0:
        sp_application_id = lookup.result.data_array[0][0]
        # Look up the numeric ID needed for secret generation
        for sp in w.service_principals.list(filter=f"displayName eq 'sp-{user_id}'"):
            if sp.application_id == sp_application_id:
                sp_numeric_id = sp.id
                break

    # 2. If not found, create a new service principal and record the mapping
    if not sp_application_id:
        sp = w.service_principals.create(display_name=f"sp-{user_id}", active=True)
        sp_application_id = sp.application_id
        sp_numeric_id = sp.id
        _execute_sql(
            f"INSERT INTO {SP_MAPPING_TABLE} (username, service_principal_id) "
            f"VALUES ('{user_id}', '{sp_application_id}')"
        )

    # 3. Upsert the SP-to-client mapping in the identity_mappings table
    _execute_sql(
        f"MERGE INTO {CLIENT_MAPPING_TABLE} AS target "
        f"USING (SELECT '{sp_application_id}' AS username, '{client_id}' AS client_id) AS source "
        f"ON target.username = source.username "
        f"WHEN MATCHED THEN UPDATE SET target.client_id = source.client_id "
        f"WHEN NOT MATCHED THEN INSERT (username, client_id) VALUES (source.username, source.client_id)"
    )

    # 4. Grant the SP CAN_QUERY permission on the serving endpoint
    endpoint = w.serving_endpoints.get(name=SERVING_ENDPOINT_NAME)
    w.api_client.do(
        "PATCH",
        f"/api/2.0/permissions/serving-endpoints/{endpoint.id}",
        body={
            "access_control_list": [
                {
                    "service_principal_name": sp_application_id,
                    "permission_level": "CAN_QUERY",
                }
            ]
        },
    )

    # 5. Grant the SP CAN_RUN permission on the Genie space
    w.api_client.do(
        "PATCH",
        f"/api/2.0/permissions/genie/{GENIE_SPACE_ID}",
        body={
            "access_control_list": [
                {
                    "service_principal_name": sp_application_id,
                    "permission_level": "CAN_RUN",
                }
            ]
        },
    )

    # 6. Grant Unity Catalog read access on the catalog, schema, and table
    _execute_sql(f"GRANT USE CATALOG ON CATALOG `{CATALOG}` TO `{sp_application_id}`")
    _execute_sql(f"GRANT USE SCHEMA ON SCHEMA `{CATALOG}`.`{SCHEMA}` TO `{sp_application_id}`")
    _execute_sql(f"GRANT SELECT ON TABLE `{CATALOG}`.`{SCHEMA}`.`{TABLE}` TO `{sp_application_id}`")

    # 7. Generate an OAuth secret for the SP and cache an authenticated client
    secret_response = w.api_client.do(
        "POST",
        f"/api/2.0/accounts/servicePrincipals/{sp_numeric_id}/credentials/secrets",
    )
    _sp_clients[user_id] = WorkspaceClient(
        host=HOST,
        client_id=sp_application_id,
        client_secret=secret_response["secret"],
    )

    return f"Credentials set for user '{user_id}' (SP: {sp_application_id}) with client '{client_id}'."


def _extract_response_text(raw: dict) -> str:
    """Extract text from a raw Responses API JSON response."""
    # Responses API: output is a list of items
    output = raw.get("output", raw.get("outputs", raw.get("predictions")))
    if not output:
        return f"No response parsed. Raw response keys: {list(raw.keys())}"

    text_parts = []
    for item in output:
        if isinstance(item, dict):
            # Message items with content blocks
            if item.get("type") == "message":
                for block in item.get("content", []):
                    if block.get("type") == "output_text":
                        text_parts.append(block.get("text", ""))
            # Simple text items
            elif "text" in item:
                text_parts.append(item["text"])
        elif isinstance(item, str):
            text_parts.append(item)

    if text_parts:
        return "\n\n".join(text_parts)

    return str(output)


def query_endpoint(message: str, history: list[dict], user_id: str, client_id: str) -> str:
    """Send a message with conversation history to the serving endpoint."""
    messages = [{"role": msg["role"], "content": msg["content"]} for msg in history]
    messages.append({"role": "user", "content": message})

    # Use the SP-authenticated client if credentials were set, otherwise fall back to default
    client = _sp_clients.get(user_id, w)

    try:
        raw = client.api_client.do(
            "POST",
            f"/serving-endpoints/{SERVING_ENDPOINT_NAME}/invocations",
            body={"input": messages},
        )
    except Exception as e:
        return f"Error querying endpoint: {e}"

    return _extract_response_text(raw)

# COMMAND ----------

with gr.Blocks() as demo:
    gr.Markdown(
        f"# Dynamic Auth Agent\n\n"
        f"Chat with the deployed agent at endpoint: `{SERVING_ENDPOINT_NAME}`\n\n"
        f"This agent uses on-behalf-of authentication to enforce row-level security."
    )
    with gr.Row():
        user_id = gr.Textbox(label="User ID", placeholder="Enter your user ID")
        client_id = gr.Textbox(label="Client ID", placeholder="Enter your client ID")
        use_creds_btn = gr.Button("Use Credentials", scale=0)
    status_text = gr.Textbox(label="Status", interactive=False)
    use_creds_btn.click(
        fn=use_credentials,
        inputs=[user_id, client_id],
        outputs=[status_text],
    )
    gr.ChatInterface(
        fn=query_endpoint,
        additional_inputs=[user_id, client_id],
    )

if __name__ == "__main__":
    demo.launch()
