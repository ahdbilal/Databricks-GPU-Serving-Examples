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
import requests
import json
from transformers import pipeline
import mlflow
from mlflow.models import infer_signature
from mlflow.transformers import generate_signature_output
from mlflow.tracking import MlflowClient

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

# Log the model with mlflow hugging face flavor
with mlflow.start_run():
    model_info = mlflow.transformers.log_model(
        transformers_model = text_generation_pipeline,
        artifact_path = "my_sentence_generator",
        inference_config = inference_config,
        input_example = input_example,
        signature = signature,
        registered_model_name = "gpt2",
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test your Model in a Notebook

# COMMAND ----------

# Load the model
my_sentence_generator = mlflow.pyfunc.load_model(model_info.model_uri)

# Generate a prediction using the loaded model with the given parameters
my_sentence_generator.predict(
    pd.DataFrame(["Hello, I'm a language model,"]),
    params={"max_new_tokens": 20, "temperature": 1},
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configure and create your model serving endpoint
# MAGIC
# MAGIC Modify the cell below to change the endpoint name. After calling the create endpoint API, the logged model will be deployed to the endpoing

# COMMAND ----------

# Set the name of the MLflow endpoint
endpoint_name = "gpt2"

# Name of the registered MLflow model
model_name = "gpt2" 

# Get the latest version of the MLflow model
model_version = MlflowClient().get_registered_model(model_name).latest_versions[0].version 

# Specify the type of compute (CPU, GPU_SMALL, GPU_MEDIUM, etc.)
workload_type = "GPU" 

# Specify the scale-out size of compute (Small, Medium, Large, etc.)
workload_size = "Small" 

# Specify Scale to Zero(only supported for CPU endpoints)
scale_to_zero = False 

# Get the API endpoint and token for the current notebook context
API_ROOT = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get() 
API_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# COMMAND ----------

# send the POST request to create the serving endpoint

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

response = requests.post(
    url=f"{API_ROOT}/api/2.0/serving-endpoints", json=data, headers=headers
)

print(json.dumps(response.json(), indent=4))

# COMMAND ----------

# MAGIC %md
# MAGIC ## View your endpoint
# MAGIC To see your more information about your endpoint, go to the "Serving" section on the left navigation bar and search for your endpoint name.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Query your endpoint!
# MAGIC
# MAGIC Once your endpoint is ready, you can query it by making an API request. Depending on the model size and complexity, it can take up to 30 minutes or more for the endpoint to get ready.  

# COMMAND ----------

# send the POST request to create the serving endpoint

data = {
  "inputs" : ["Hello, I'm a language model,"],
  "params" : {"max_new_tokens": 100, "temperature": 1}
}

headers = {"Context-Type": "text/json", "Authorization": f"Bearer {API_TOKEN}"}

response = requests.post(
    url=f"{API_ROOT}/serving-endpoints/{endpoint_name}/invocations", json=data, headers=headers
)

print(json.dumps(response.json()))
