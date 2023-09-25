# Databricks notebook source
# MAGIC %md
# MAGIC # Optimized MPT Serving Example
# MAGIC
# MAGIC Optimized LLM Serving enables you to take state of the art OSS LLMs and deploy them on Databricks Model Serving with automatic optimizations for improved latency and throughput on GPUs. Currently, we support optimizations for Llama2 and Mosaic MPT class of models and will continue introducing more models with optimization support.
# MAGIC
# MAGIC This example walks through:
# MAGIC
# MAGIC 1. Downloading the model from huggingface transformers
# MAGIC 2. Logging the model in an optimized serving supported format into the Databricks Unity Catalog or Workspace Registry
# MAGIC 3. Enabling optimized serving on the model

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prerequisites
# MAGIC - Attach a cluster with sufficient memory to the notebook
# MAGIC - Make sure to have MLflow version 2.7.0 or later installed
# MAGIC - Make sure to enable "Models in UC", especially when working with models larger than 7B in size
# MAGIC

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Step 1: Log the model for Optimized LLM Serving

# COMMAND ----------

# Update/Install required dependencies
!pip install -U mlflow
!pip install -U transformers
!pip install -U accelerate
dbutils.library.restartPython()

# COMMAND ----------

from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained('mosaicml/mpt-7b', device_map="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("mosaicml/mpt-7b")

# COMMAND ----------

import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import ColSpec, Schema
import numpy as np

# Define the model input and output schema
input_schema = Schema([
    ColSpec("string", "prompt"),
    ColSpec("double", "temperature", optional=True),
    ColSpec("integer", "max_tokens", optional=True),
    ColSpec("string", "stop", optional=True),
    ColSpec("integer", "candidate_count", optional=True)
])

output_schema = Schema([
    ColSpec('string', 'predictions')
])

signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Define an example input
input_example = {
    "prompt": np.array([
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        "What is Apache Spark?\n\n"
        "### Response:\n"
    ]),
    "max_tokens": np.array([75]),
    "temperature": np.array([0.0])
}

# COMMAND ----------

# MAGIC %md
# MAGIC To enable optimized serving, when logging the model, include the extra metadata dictionary when calling `mlflow.transformers.log_model` as shown below:
# MAGIC
# MAGIC ```
# MAGIC metadata = {"task": "llm/v1/completions"}
# MAGIC ```
# MAGIC This specifies the API signature used for the model serving endpoint.
# MAGIC

# COMMAND ----------

import mlflow

# Comment out the line below if not using Models in UC 
# and simply provide the model name instead of three-level namespace
mlflow.set_registry_uri('databricks-uc')
CATALOG = "ml"
SCHEMA = "models"
registered_model_name = f"{CATALOG}.{SCHEMA}.mpt"

# Start a new MLflow run
with mlflow.start_run():
    components = {
        "model": model,
        "tokenizer": tokenizer,
    }
    mlflow.transformers.log_model(
        transformers_model=components,
        artifact_path="model",
        registered_model_name=registered_model_name,
        signature=signature,
        input_example=input_example,
        metadata={"task": "llm/v1/completions"}
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Configure and create your model serving GPU endpoint
# MAGIC
# MAGIC Modify the cell below to change the endpoint name. After calling the create endpoint API, the logged MPT-7B model will automatically be deployed with optimized LLM Serving!

# COMMAND ----------

# Set the name of the MLflow endpoint
endpoint_name = "mpt7b"

# Name of the registered MLflow model
model_name = "ml.models.mpt7b" 

# Get the latest version of the MLflow model
model_version = 1

# Specify the type of compute (CPU, GPU_SMALL, GPU_MEDIUM, etc.)
workload_type = "GPU_LARGE" 

# Specify the compute scale-out size(Small, Medium, Large, etc.)
workload_size = "Small" 

# Specify Scale to Zero(only supported for CPU endpoints)
scale_to_zero = False 

# Get the API endpoint and token for the current notebook context
API_ROOT = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get() 
API_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# COMMAND ----------

import requests
import json

data = {
    "name": endpoint_name,
    "config": {
        "served_models": [
            {
                "model_name": model_name,
                "model_version": model_version,
                "workload_size": workload_size,
                "scale_to_zero_enabled": scale_to_zero,
                "workload_type": workload_type,
            }
        ]
    },
}

headers = {"Context-Type": "text/json", "Authorization": f"Bearer {API_TOKEN}"}

response = requests.post(url=f"{API_ROOT}/api/2.0/serving-endpoints", json=data, headers=headers)

print(json.dumps(response.json(), indent=4))

# COMMAND ----------

# MAGIC %md
# MAGIC ## View your endpoint
# MAGIC To see your more information about your endpoint, go to the "Serving" section on the left navigation bar and search for your endpoint name.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Query your endpoint!
# MAGIC
# MAGIC Once your endpoint is ready, you can query it by making an API request. Depending on the model size and complexity, it can take up to 30 minutes or more for the endpoint to get ready.  

# COMMAND ----------

data = {
    "inputs": {
        "prompt": [
            "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nWhat is Apache Spark?\n\n### Response:\n"
        ]
    },
    "params": {
        "max_tokens": 100, 
        "temperature": 0.0
    }
}

headers = {
    "Context-Type": "text/json",
    "Authorization": f"Bearer {API_TOKEN}"
}

response = requests.post(
    url=f"{API_ROOT}/serving-endpoints/{endpoint_name}/invocations",
    json=data,
    headers=headers
)

print(json.dumps(response.json()))
