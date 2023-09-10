# Databricks notebook source
# MAGIC %md
# MAGIC # Geting Stared with Model Serving
# MAGIC
# MAGIC In this guide, we demonstrate how to deploy a model to a serving endpoint. Though we specifically will deploy a GPT-2 model to a GPU endpoint, the workflow outlined here can be adapted for deploying other types of models to either CPU or GPU endpoints.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install and Import Libraries 

# COMMAND ----------

!pip install --upgrade mlflow
!pip install --upgrade transformers
!pip install --upgrade accelerate
dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd
import mlflow
from mlflow.models import infer_signature
from mlflow.transformers import generate_signature_output
from mlflow.tracking import MlflowClient
from transformers import pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initialize and Configure your Model

# COMMAND ----------

# Define and configure your model using any popular ML framework
text_generation_pipeline = pipeline(task='text-generation', model='gpt2', pad_token_id = 50256, device_map= "auto")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log your Model using MLflow

# COMMAND ----------

# Define inference parameters that will be passed to the model at the time of inference
inference_config = {"max_new_tokens": 100, "temperature": 1}

# Define schema for the model
input_example = pd.DataFrame(["Hello, I'm a language model,"])
output = generate_signature_output(text_generation_pipeline, input_example)
signature = infer_signature(input_example, output, params=inference_config)

# Log model with mlflow huggingface falvour
with mlflow.start_run():
    model_info = mlflow.transformers.log_model(
        transformers_model=text_generation_pipeline,
        artifact_path="my_sentence_generator",
        inference_config=inference_config,
        input_example=input_example,
        signature=signature,
        registered_model_name="gpt2",
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test your Model in a Notebook

# COMMAND ----------

my_sentence_generator = mlflow.pyfunc.load_model(model_info.model_uri)
my_sentence_generator.predict(
    pd.DataFrame(["Hello, I'm a language model,"]),
    params={"max_length": 20, "temperature": 1},
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configure and create your model serving endpoint
# MAGIC
# MAGIC Modify the cell below to change the endpoint name. After calling the create endpoint API, the logged model will be deployed to the endpoing

# COMMAND ----------

endpoint_name = "gpt2"
model_name = "gpt2"
model_version =  MlflowClient().get_registered_model(model_name).latest_versions[0].version
workload_type = "CPU" # Compute type (options includes "CPU", "GPU_SMALL", "GPU_MEDIUM" and more)
workload_size = "Small" # Compute Scale-out (some options are "Small" , "Medium","Large" and more)
scale_to_zero = True # Scale to zero is currently only support for CPU endpoints
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
                "workload_size": workload_size,
                "scale_to_zero_enabled": scale_to_zero,
                "workload_type": workload_type
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
