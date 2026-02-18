# Databricks notebook source

# COMMAND ----------

import mlflow
from mlflow.models.resources import DatabricksServingEndpoint, DatabricksGenieSpace
from mlflow.models.auth_policy import AuthPolicy, SystemAuthPolicy, UserAuthPolicy

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

config = mlflow.models.ModelConfig(development_config="config.yaml")

mlflow_config = config.get("mlflow_config")
agent_config = config.get("agent_config")

experiment_name = mlflow_config.get("experiment_name")
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    mlflow.create_experiment(experiment_name)
mlflow.set_experiment(experiment_name)

# COMMAND ----------

# Log the agent model with OBO auth policy

registered_model_name = mlflow_config.get("registered_model_name")

with mlflow.start_run():
    model_info = mlflow.pyfunc.log_model(
        name="agent",
        python_model="agent.py",
        model_config="config.yaml",
        registered_model_name=registered_model_name,
        auth_policy=AuthPolicy(
            system_auth_policy=SystemAuthPolicy(
                resources=[
                    DatabricksServingEndpoint(
                        endpoint_name=agent_config.get("llm_endpoint_name")
                    ),
                ]
            ),
            user_auth_policy=UserAuthPolicy(
                api_scopes=[
                    "serving.serving-endpoints",
                ]
            ),
        ),
        resources=[
            DatabricksGenieSpace(
                genie_space_id=agent_config.get("genie_space_id")
            ),
        ],
    )

print(f"Model logged: {model_info.model_uri}")

# COMMAND ----------

# Deploy the agent to a Model Serving endpoint

from databricks import agents

deployment = agents.deploy(
    model_name=registered_model_name,
    model_version=model_info.registered_model_version,
)

print(f"Endpoint: {deployment.endpoint_name}")
print(f"Query endpoint: {deployment.query_endpoint}")
