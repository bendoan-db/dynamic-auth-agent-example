# Databricks notebook source
# MAGIC %pip install -qqq -r ../requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./agent

# COMMAND ----------

import os
import sys
from importlib import import_module

notebook_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
if notebook_dir not in sys.path:
    sys.path.insert(0, notebook_dir)

# Load the agent module - this will execute the module and create the AGENT
agent_module = import_module("agent")
AGENT = agent_module.AGENT

print(f"Agent loaded: {type(AGENT).__name__}")

# COMMAND ----------

import mlflow
# Configure MLflow for Databricks
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")  # Use Unity Catalog for model registry

# Set MLflow experiment from config, create if it doesn't exist
#experiment_name = agent_module.config.get("mlflow_config").get("experiment_name")
experiment_name = "/Users/ben.doan@databricks.com/dynamic_auth_agent"
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    mlflow.create_experiment(experiment_name)
mlflow.set_experiment(experiment_name)

from mlflow.types.responses import ResponsesAgentRequest

request = ResponsesAgentRequest(
    input=[
        {
            "role": "user",
            "content": "give me the count of products purchased for each client_id. Explain the reasoning on how you got to your response. If you queried the customer transactions agent, then also return the raw payload from that agent in your anwer",
        }
    ]
)

response = AGENT.predict(request)

# COMMAND ----------

for item in response.output:
    print("\n\n" + str(item))

# COMMAND ----------


