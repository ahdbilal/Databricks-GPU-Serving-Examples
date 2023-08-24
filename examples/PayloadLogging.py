# Databricks notebook source
# import libraries
import os
import requests
import numpy as np
import pandas as pd
import json
import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec
from mlflow.tracking import MlflowClient

# COMMAND ----------

# Define constants
endpoint_name = "CallEndpoint" # endpoint name of the cpu endpoint
model_name = "gpt-cpu" # model name that will be deployed to cpu endpoint
os.environ["URI"] = dbutils.secrets.get(scope="llm", key="endpoint_uri") # endpoint uri of the gpu endpoint
os.environ["TOKEN"] = dbutils.secrets.get(scope="llm", key="endpoint_token") # token to access the gpu endpoint

# COMMAND ----------

class CallEndpoint(mlflow.pyfunc.PythonModel):
    def __init__(self):
        self.url = os.environ["URI"]
        self.headers = {'Authorization': f'Bearer {os.environ["TOKEN"]}', 'Content-Type': 'application/json'}
        self.session = requests.Session()

    def create_tf_serving_json(self, data):
      return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

    def score_model(self, dataset):
      if isinstance(dataset, pd.DataFrame):
        ds_dict = {'dataframe_records': dataset.to_dict(orient='records')}
      else:
        ds_dict = self.create_tf_serving_json(dataset)
      data_json = json.dumps(ds_dict, allow_nan=True)
      response = self.session.post(url=self.url, headers=self.headers, data=data_json)
      if response.status_code != 200:
        raise Exception(f'Request failed with status {response.status_code}, {response.text}')
      return response.json()

    def predict(self, context, model_input):
      #change this logic as per your requirement
      if 'feedback' in model_input.columns:
          model_input = model_input.drop(columns='feedback')
          return ["Your request has been received and is being processed."]
      else:
        return self.score_model(model_input)['predictions']

# COMMAND ----------

# Log the model

# Change schema as desired
input_schema = Schema([
    ColSpec(DataType.string, "prompt", optional= True), 
    ColSpec(DataType.double, "temperature", optional= True), 
    ColSpec(DataType.long, "max_tokens", optional= True),
    ColSpec(DataType.string, "feedback", optional= True)])
output_schema = Schema([ColSpec(DataType.string)])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Define an input example for your use case
input_example=pd.DataFrame({"prompt":["what is Machine Learning?"], 
                            "temperature": [0.0],
                            "max_tokens": [100]})

with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
        "model",
        python_model=CallEndpoint(),
        registered_model_name=model_name,
        signature=signature,
        input_example=input_example,
    )

# COMMAND ----------

# Offline test
loaded_model = mlflow.pyfunc.load_model("runs:/" + run.info.run_id+'/model')
loaded_model.predict(input_example)

# COMMAND ----------

# Deploy the model to an endpoint
client = MlflowClient()
API_ROOT = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
API_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
url = f"{API_ROOT}/api/2.0/serving-endpoints"
headers = {"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"}
payload = {
    "name": endpoint_name,
    "config":{
      "served_models": [{
            "model_name": model_name,
            "model_version": client.get_registered_model(model_name).latest_versions[0].version,
            "workload_size": "Small",
            "scale_to_zero_enabled": "True",
            "environment_vars": {
              "URI": "{{secrets/llm/endpoint_uri}}",
              "TOKEN": "{{secrets/llm/endpoint_token}}"
              }

      }]
    }
}
response = requests.post(url, headers=headers, data=json.dumps(payload))
response.text

# COMMAND ----------

# Online test
data = {
  "dataframe_records":
    [
        {"prompt":"what is Machine Learning?", 
        "temperature": 0.0,
        "max_tokens": 75}
    ],
    "inference_id": "123qwe" 
}

headers = {
    "Authorization": f"Bearer {API_TOKEN}",
}

response = requests.post(
    url=f"{API_ROOT}/serving-endpoints/{endpoint_name}/invocations",
    json=data,
    headers=headers
)

print("Response status:", response.status_code)
print("Reponse text:", response.text)

# COMMAND ----------

# Online test
data = {
  "dataframe_records":
    [
        {"prompt":"what is Machine Learning?", 
        "temperature": 0.0,
        "max_tokens": 75,
        "feedback": "Machine learning is a branch of artificial intelligence that involves the development of algorithms that can learn from and make predictions or decisions based on data. It's used in various applications like speech recognition, recommendation systems, and image classification."}
    ],
    "inference_id": "123qwe" 
}

headers = {
    "Authorization": f"Bearer {API_TOKEN}",
}

response = requests.post(
    url=f"{API_ROOT}/serving-endpoints/{endpoint_name}/invocations",
    json=data,
    headers=headers
)

print("Response status:", response.status_code)
print("Reponse text:", response.text)
