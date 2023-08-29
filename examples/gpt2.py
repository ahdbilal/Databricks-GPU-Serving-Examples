# Databricks notebook source
!pip install --upgrade mlflow
!pip install --upgrade transformers
!pip install --upgrade accelerate
dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd
import numpy as np
import transformers
import mlflow
import torch
import os

# COMMAND ----------

# Download model and package it with the model container 
pipe = transformers.pipeline('text-generation', model='gpt2', device = 0)
snapshot_location = os.path.expanduser("~/.cache/huggingface/pipeline-gpt2")
os.makedirs(snapshot_location, exist_ok=True)
pipe.save_pretrained(snapshot_location)

# COMMAND ----------

class GPT2(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """
        This method initializes the transformer
        """
        self.generator = transformers.pipeline('text-generation', model=context.artifacts['repository'], device = 0)
        transformers.set_seed(42)

    def predict(self, context, model_input):
        """
        This method generates prediction for the given input.
        """
        prompt = model_input["prompt"][0]
        temperature = model_input.get("temperature", [1.0])[0]
        max_tokens = model_input.get("max_tokens", [100])[0]
        
        response = self.generator(prompt, max_length=max_tokens, temperature = temperature)

        return [response[0]['generated_text']]

# COMMAND ----------

from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec

# Define input and output schema
input_schema = Schema([
    ColSpec(DataType.string, "prompt"), 
    ColSpec(DataType.double, "temperature"), 
    ColSpec(DataType.long, "max_tokens")])
output_schema = Schema([ColSpec(DataType.string)])
signature = ModelSignature(inputs=input_schema)

# Define input example
input_example=pd.DataFrame({
            "prompt":["Hello, I am a language Model,"], 
            "temperature": [0.5],
            "max_tokens": [100]})

# Log the model with its details such as artifacts, pip requirements and input example
with mlflow.start_run() as run:  
    mlflow.pyfunc.log_model(
        "model",
        python_model=GPT2(),
        artifacts={'repository' : snapshot_location},
        pip_requirements=["torch", "transformers", "accelerate"],
        input_example=input_example,
        signature=signature,
    )

# COMMAND ----------

# Register model in MLflow Model Registry
result = mlflow.register_model(
    "runs:/"+run.info.run_id+"/model",
    "gpt2"
)

# COMMAND ----------

# Load the logged model
loaded_model = mlflow.pyfunc.load_model(f"models:/{result.name}/{result.version}")

# COMMAND ----------

# Make a prediction using the loaded model
input_example=pd.DataFrame({"prompt":["Hello, I am a language Model,"], "temperature": [0.5],"max_tokens": [100]})
loaded_model.predict(input_example)
