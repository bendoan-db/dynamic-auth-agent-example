# Dynamic Auth Agent

A multi-agent LangGraph supervisor system for Databricks that demonstrates dynamic per-user authentication using service principals to enforce row-level security (RLS) through Unity Catalog.

## Architecture

```
Chat App (Gradio)
  │
  ├── "Use Credentials" ──► Service Principal provisioning + OAuth secret
  │                          + permissions grants (endpoint, Genie, UC)
  │
  └── Chat messages ──► Model Serving endpoint (authenticated as SP)
                            │
                            └── Supervisor Agent (LangGraph)
                                   └── Genie Agent (queries as SP identity)
                                          └── Unity Catalog RLS filter
                                                 └── identity_mappings table
                                                        (SP application_id → client_id)
```

### Core Components

| File | Purpose |
|------|---------|
| `agent/agent.py` | LangGraph supervisor agent with OBO-authenticated Genie sub-agent |
| `agent/deploy_agent.py` | Logs and deploys the agent to Model Serving with OBO auth policy |
| `agent/evaluate_agent.py` | Test notebook for evaluating agent responses |
| `chat_application/chat_app.py` | Gradio chat UI with dynamic SP authentication |
| `config.yaml` | Central configuration for all components |
| `uc_permissions_setup/apply_rls.ipynb` | Sets up the RLS row filter on the data table |
| `databricks.yml` | Databricks Asset Bundle definition |

## Authentication Workflow

The chat application implements a dynamic authentication flow that maps arbitrary user identities to Databricks service principals, enabling per-user row-level security without requiring each user to have a Databricks account.

### Service Principal Creation Flow

```
User clicks "Use Credentials" (user_id, client_id)
                │
                ▼
┌───────────────────────────────┐
│ 1. Query sp_mapping table     │
│    WHERE username = user_id   │
└──────────────┬────────────────┘
               │
        ┌──────┴──────┐
        │ SP exists?  │
        └──────┬──────┘
          No   │   Yes
        ┌──────┴──────────────────────┐
        ▼                             ▼
┌───────────────────┐    ┌────────────────────────┐
│ 2a. Create new SP │    │ 2b. Look up existing   │
│  (sp-{user_id})   │    │  SP by display name    │
│                   │    │  to get numeric ID     │
│ Insert into       │    └────────────┬───────────┘
│ sp_mapping table  │                 │
└────────┬──────────┘                 │
         └──────────┬─────────────────┘
                    ▼
     ┌──────────────────────────────┐
     │ 3. MERGE into                │
     │    identity_mappings table   │
     │    (SP application_id →      │
     │     client_id)               │
     └──────────────┬───────────────┘
                    ▼
     ┌──────────────────────────────┐
     │ 4. Grant permissions         │
     │    • CAN_QUERY on endpoint   │
     │    • CAN_RUN on Genie space  │
     │    • USE CATALOG / SCHEMA    │
     │    • SELECT on data table    │
     └──────────────┬───────────────┘
                    ▼
     ┌──────────────────────────────┐
     │ 5. Generate OAuth M2M secret │
     │    for the SP                │
     └──────────────┬───────────────┘
                    ▼
     ┌──────────────────────────────┐
     │ 6. Cache WorkspaceClient     │
     │    (client_id + secret)      │
     │    for this user session     │
     └──────────────┬───────────────┘
                    ▼
     ┌──────────────────────────────┐
     │ Chat messages now use the    │
     │ SP-authenticated client      │
     │                              │
     │ current_user() = SP app ID   │
     │       │                      │
     │       ▼                      │
     │ identity_mappings resolves   │
     │ SP → client_id               │
     │       │                      │
     │       ▼                      │
     │ RLS row filter returns only  │
     │ that client's rows           │
     └─────────────────────────────┘
```

### How It Works

When a user clicks **"Use Credentials"** with a user ID and client ID:

1. **Service Principal Lookup/Creation**
   - Queries the `service_principal_mapping` table to check if a service principal already exists for this user ID
   - If not found, creates a new service principal (`sp-{user_id}`) via the Databricks SDK and records the mapping

2. **Client Identity Mapping**
   - Upserts a row in the `identity_mappings` table mapping the SP's `application_id` to the provided `client_id`
   - This is the table the RLS row filter function (`row_filter_by_client`) queries: it checks `WHERE username = current_user()` to determine which `client_id` the caller should see

3. **Permissions Grants**
   - **Serving endpoint**: Grants `CAN_QUERY` so the SP can invoke the agent endpoint
   - **Genie space**: Grants `CAN_RUN` so the SP can execute Genie queries
   - **Unity Catalog**: Grants `USE CATALOG`, `USE SCHEMA`, and `SELECT` on the data table so the SP can read data (subject to RLS)

4. **OAuth Secret Generation**
   - Generates an OAuth M2M secret for the service principal via the Databricks REST API
   - Creates a `WorkspaceClient` authenticated with the SP's `client_id`/`client_secret` and caches it in memory

5. **Authenticated Chat**
   - Subsequent chat messages are sent to the Model Serving endpoint using the SP-authenticated client
   - The endpoint sees `current_user()` as the SP's `application_id`
   - The OBO flow passes this identity to the Genie agent, which queries Unity Catalog as the SP
   - The RLS row filter resolves the SP's identity to the mapped `client_id`, returning only that client's data

### Row-Level Security

The RLS filter is defined in `uc_permissions_setup/apply_rls.ipynb`:

```sql
CREATE FUNCTION row_filter_by_client(row_client_id STRING)
RETURNS BOOLEAN
RETURN (
    row_client_id = (
        SELECT client_id
        FROM identity_mappings
        WHERE username = current_user()
        LIMIT 1
    )
);

ALTER TABLE customer_transactions
SET ROW FILTER row_filter_by_client ON (client_id);
```

This ensures that when the agent queries `customer_transactions`, only rows belonging to the authenticated user's mapped client are returned.

## Prerequisites

- A Databricks workspace with Unity Catalog enabled
- Databricks CLI configured (`databricks auth login`)
- A SQL warehouse
- A Genie space configured with the target data table
- Permissions to create service principals and manage Unity Catalog grants

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure `config.yaml`

Update the following values:

```yaml
databricks_config:
  catalog: <your-catalog>
  schema: <your-schema>
  table: <your-table>
  host: <your-workspace-url>

mlflow_config:
  experiment_name: <your-experiment-path>
  registered_model_name: <catalog.schema.model_name>

agent_config:
  llm_endpoint_name: <your-llm-endpoint>
  genie_space_id: <your-genie-space-id>
  genie_agent_name: <agent-name>
  genie_agent_description: <description>
  serving_endpoint_name: <your-serving-endpoint>

auth_config:
  warehouse_id: <your-sql-warehouse-id>
  sp_mapping_table: <catalog.schema.service_principal_mapping>
  client_mapping_table: <catalog.schema.identity_mappings>
```

### 3. Create the Required Tables

Create the two mapping tables in your Unity Catalog schema:

```sql
-- Service principal mapping: tracks which SP belongs to which user
CREATE TABLE IF NOT EXISTS <catalog>.<schema>.service_principal_mapping (
    username STRING,
    service_principal_id STRING
);

-- Identity mapping: maps SP application_id to client_id (used by RLS filter)
CREATE TABLE IF NOT EXISTS <catalog>.<schema>.identity_mappings (
    username STRING,
    client_id STRING
);
```

### 4. Apply Row-Level Security

Run the `uc_permissions_setup/apply_rls.ipynb` notebook to create the row filter function and apply it to your data table.

### 5. Deploy the Agent

Run the deployment notebook to log the agent with OBO auth policy and deploy it to Model Serving:

```bash
# Option 1: Run the deploy notebook in Databricks
# Open agent/deploy_agent.py in the Databricks workspace and run all cells

# Option 2: Use Databricks Asset Bundles
databricks bundle deploy -t dev
```

### 6. Run the Chat Application

```bash
python chat_application/chat_app.py
```

This launches a Gradio web UI (default: `http://localhost:7860`).

## Usage

1. Open the chat app in your browser
2. Enter a **User ID** (any identifier, e.g. `alice`) and a **Client ID** (the client whose data this user should see)
3. Click **"Use Credentials"** -- the status bar will confirm the SP was created/found and permissions were granted
4. Type a question about customer transactions in the chat (e.g. "What are my total transactions?")
5. The agent responds with data filtered to only the mapped client's rows

To switch users or clients, enter new values and click "Use Credentials" again. A new OAuth secret is generated each time.
