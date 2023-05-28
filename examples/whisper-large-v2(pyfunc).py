# Databricks notebook source
# MAGIC %md
# MAGIC ##Log the Whisper Large V2 Model to Mlflow Registry

# COMMAND ----------

import pandas as pd
import numpy as np
import transformers
import mlflow
import torch
from datasets import load_dataset

# COMMAND ----------

# Download the Dolly model snapshot from huggingface
from huggingface_hub import snapshot_download
snapshot_location = snapshot_download(repo_id="openai/whisper-large-v2", ignore_patterns=["*.msgpack", "*.h5"])

# COMMAND ----------

# Define custom Python model class
class Whisper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
      repository = context.artifacts['repository']
      self.processor = transformers.WhisperProcessor.from_pretrained(repository)
      self.model = transformers.WhisperForConditionalGeneration.from_pretrained(
          repository, low_cpu_mem_usage=True, device_map="auto")
      self.model.config.forced_decoder_ids = None
      self.model.to('cuda').eval()

    def predict(self, context, model_input):
      audio = model_input["audio"] 
      sampling_rate = model_input["sampling_rate"][0]
      with torch.no_grad():
          input_features = self.processor(audio, sampling_rate=sampling_rate, return_tensors='pt').input_features.to('cuda')
          predicted_ids = self.model.generate(input_features).to('cuda')
      return self.processor.batch_decode(predicted_ids, skip_special_tokens=True)

# COMMAND ----------

from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec
from mlflow.models.signature import infer_signature

# Define input example
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
sample = ds[0]["audio"]
input_example=pd.DataFrame({
            "audio":sample["array"], 
            "sampling_rate": sample["sampling_rate"]})
signature = infer_signature(input_example, "some random text")

# Log the model with its details such as artifacts, pip requirements and input example
with mlflow.start_run() as run:  
    mlflow.pyfunc.log_model(
        "model",
        python_model=Whisper(),
        artifacts={'repository' : snapshot_location},
        pip_requirements=["torch", "transformers", "accelerate"],
        input_example=input_example,
        signature=signature,
    )

# COMMAND ----------

# Register model in MLflow Model Registry
result = mlflow.register_model(
    "runs:/"+run.info.run_id+"/model",
    "whisper-large-v2"
)

# COMMAND ----------

# Load the logged model
loaded_model = mlflow.pyfunc.load_model("runs:/"+run.info.run_id+"/model")

# COMMAND ----------

# Make a prediction using the loaded model
loaded_model.predict(input_example)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Make API request to Model Serving Endpoint

# COMMAND ----------

import os
import requests
import pandas as pd
import json
import matplotlib.pyplot as plt

# define parameters at the start
URL = ""
DATABRICKS_TOKEN = ""
INPUT_EXAMPLE = pd.DataFrame({"prompt":["a photo of an astronaut riding a horse on water"]})

def score_model(dataset, url=URL, databricks_token=DATABRICKS_TOKEN):
    headers = {'Authorization': f'Bearer {databricks_token}', 
               'Content-Type': 'application/json'}
    ds_dict = {'dataframe_split': dataset.to_dict(orient='split')}
    data_json = json.dumps(ds_dict, allow_nan=True)
    response = requests.request(method='POST', headers=headers, url=url, data=data_json)
    if response.status_code != 200:
        raise Exception(f'Request failed with status {response.status_code}, {response.text}')

    return response.json()

# COMMAND ----------

# play a sample audio
from IPython.display import Audio
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
sample = ds[12]["audio"]

Audio(sample["array"], rate=sample["sampling_rate"])

# COMMAND ----------

# transcribe the audio by sending api request to the endpoint
score_model(pd.DataFrame({"audio":sample["array"], "sampling_rate": sample["sampling_rate"]}))
