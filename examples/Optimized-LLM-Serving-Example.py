# Databricks notebook source
# MAGIC %md
# MAGIC # Optimized LLM Serving Example
# MAGIC
# MAGIC Optimized LLM Serving enables you to take state of the art OSS LLMs and deploy them on Databricks Model Serving with automatic optimizations for improved latency and throughput on GPUs. Currently, we support optimizing the Mosaic MPT-7B model and will continue introducing more models with optimization support.
# MAGIC
# MAGIC This example walks through:
# MAGIC
# MAGIC 1. Downloading the `mosaicml/mpt-7b` model from huggingface transformers
# MAGIC 2. Logging the model in an optimized serving supported format into the Databricks Model Registry
# MAGIC 3. Enabling optimized serving on the model

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prerequisites
# MAGIC * Attach a cluster to the notebook with sufficient memory to load MPT-7B. We recommend a cluster with at least 32 GB of memory.
# MAGIC * (Optional) Install the latest transformers. MPT-7B native support in transformers was added on July 25, 2023. At the time of this notebook release, MPT-7B native support in transformers has not been officially released. For full compatibility of MPT-7B with mlflow, install the latest version from github. Optimized serving will work with older versions of transformers for MPT-7B, but there may be issues with loading the model locally.
# MAGIC
# MAGIC To install the latest version of transformers off github, run:
# MAGIC ```
# MAGIC %pip install git+https://github.com/huggingface/transformers@main
# MAGIC ```
# MAGIC
# MAGIC

# COMMAND ----------

!pip install -U transformers
!pip install -U accelerate
!pip install -U tensorflow
!pip install -U mlflow
dbutils.library.restartPython()

# COMMAND ----------

from transformers import AutoModelForCausalLM, AutoTokenizer

# If you are using the latest version of transformers that has native MPT support, replace the following line with:
model = AutoModelForCausalLM.from_pretrained('mosaicml/mpt-30b', low_cpu_mem_usage=True)

#model = AutoModelForCausalLM.from_pretrained('mosaicml/mpt-30b', low_cpu_mem_usage=True, trust_remote_code=True)

# COMMAND ----------

tokenizer = AutoTokenizer.from_pretrained("mosaicml/mpt-30b")

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Logging required metadata for optimized serving
# MAGIC
# MAGIC To enable optimized serving, when logging the model, include the extra metadata dictionary when calling `mlflow.transformers.log_model` as shown below:
# MAGIC
# MAGIC ```
# MAGIC metadata = {"task": "llm/v1/completions"}
# MAGIC ```
# MAGIC This specifies the API signature used for the model serving endpoint.
# MAGIC

# COMMAND ----------

import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import ColSpec, Schema


input_schema = Schema([
    ColSpec("string", "prompt"),
    ColSpec("double", "temperature", optional= True),
    ColSpec("integer", "max_tokens", optional= True),
    ColSpec("string", "stop", optional= True), # Assuming the inner arrays only contain strings
    ColSpec("integer", "candidate_count", optional= True)
])

output_schema = Schema([
    ColSpec("string", "prompt"),
    ColSpec("double", "temperature", optional= True),
    ColSpec("integer", "max_tokens", optional= True),
    ColSpec("string", "stop", optional= True), # Assuming the inner arrays only contain strings
    ColSpec("integer", "candidate_count", optional= True)
])

ouput_schema = Schema([
    ColSpec('string', 'predictions')
])

# Create a model signature with just the output schema
signature = ModelSignature(inputs = input_schema,outputs= ouput_schema)
signature

# COMMAND ----------

import mlflow
import numpy as np
mlflow.set_registry_uri('databricks-uc')

with mlflow.start_run():
    components = {
        "model": model,
        "tokenizer": tokenizer,
    }
    mlflow.transformers.log_model(
        transformers_model=components,
        artifact_path="mpt",
        signature=signature,
        registered_model_name="opti-mpt-30b",
        input_example={"prompt": np.array(["Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nWhat is Apache Spark?\n\n### Response:\n"]), "max_tokens": np.array([75]), "temperature": np.array([0.0])},
        metadata = {"task": "llm/v1/completions"}
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configure and create your model serving GPU endpoint
# MAGIC
# MAGIC Modify the cell below to change the endpoint name. After calling the create endpoint API, the logged MPT-7B model will automatically be deployed with optimized LLM Serving!

# COMMAND ----------

endpoint_name = "mpt-7b-instruct "
model_name = "mpt-7b-instruct "
model_version = "1"
served_model_workload_size = "Small"
served_model_scale_to_zero = False

API_ROOT = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
API_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# COMMAND ----------

import json
import requests

data = {
    "name": endpoint_name,
    "config": {
        "served_models": [
            {
                "model_name": model_name,
                "model_version": model_version,
                "workload_size": served_model_workload_size,
                "scale_to_zero_enabled": False,
                "workload_type": "GPU_MEDIUM"
            }
        ]
    }
}

headers = {
    "Context-Type": "text/json",
    "Authorization": f"Bearer {API_TOKEN}"
}

response = requests.post(
    url=f"{API_ROOT}/api/2.0/preview/serving-endpoints",
    json=data,
    headers=headers
)

print(json.dumps(response.json(), indent=4))


# COMMAND ----------

# MAGIC %md
# MAGIC ## View your endpoint!
# MAGIC To see your more information about your endpoint, go to the "Serving" section on the left navigation bar and search for your endpoint name.
