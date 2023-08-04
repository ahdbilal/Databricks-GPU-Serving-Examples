# Databricks notebook source
import pandas as pd
import numpy as np
import transformers
import mlflow
import torch

# COMMAND ----------

from huggingface_hub import snapshot_download
# Download the model snapshot from huggingface
snapshot_location = snapshot_download(repo_id="google/flan-t5-xl",  ignore_patterns=["*.msgpack", "*.h5"])

# COMMAND ----------

class Flan(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """
        This method initializes the tokenizer and language model
        using the specified model repository.
        """
        # Initialize tokenizer and language model
        self.tokenizer = transformers.T5Tokenizer.from_pretrained(
            context.artifacts['repository'], padding_side="left")
        self.model = transformers.T5ForConditionalGeneration.from_pretrained(
            context.artifacts['repository'], 
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto")

    def predict(self, context, model_input):
        """
        This method generates prediction for the given input.
        """
        prompt = model_input["prompt"][0]
        temperature = model_input.get("temperature", [1.0])[0]
        max_tokens = model_input.get("max_tokens", [100])[0]

        # Encode the input and generate prediction
        encoded_input = self.tokenizer.encode(prompt, return_tensors='pt').to("cuda")
        output = self.model.generate(encoded_input, do_sample=True, temperature=temperature, max_new_tokens=max_tokens)
    
        # Decode the prediction to text
        generated_text = self.tokenizer.decode(output[0])

        return generated_text

# COMMAND ----------

from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec

# Define input and output schema
input_schema = Schema([
    ColSpec(DataType.string, "prompt"), 
    ColSpec(DataType.double, "temperature"), 
    ColSpec(DataType.long, "max_tokens")])
output_schema = Schema([ColSpec(DataType.string)])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Define input example
input_example=pd.DataFrame({
            "prompt":["Translate to German:  My name is Arthur"], 
            "temperature": [0.5],
            "max_tokens": [100]})

# Log the model with its details such as artifacts, pip requirements and input example
with mlflow.start_run() as run:  
    mlflow.pyfunc.log_model(
        "model",
        python_model=Flan(),
        artifacts={'repository' : snapshot_location},
        pip_requirements=["torch", "transformers", "accelerate", "einops","sentencepiece"],
        input_example=input_example,
        signature=signature,
    )

# COMMAND ----------

# Register model in MLflow Model Registry
result = mlflow.register_model(
    "runs:/"+run.info.run_id+"/model",
    "flan-t5-xl"
)
# Note: Due to the large size of the model, the registration process might take longer than the default maximum wait time of 300 seconds. MLflow could throw an exception indicating that the max wait time has been exceeded. Don't worry if this happens - it's not necessarily an error. Instead, you can confirm the registration status of the model by directly checking the model registry. This exception is merely a time-out notification and does not necessarily imply a failure in the registration process.

# COMMAND ----------

# Load the logged model
loaded_model = mlflow.pyfunc.load_model(f"models:/{result.name}/{result.version}")

# COMMAND ----------

# Make a prediction using the loaded model
input_example=pd.DataFrame({"prompt":["Translate to German:  My name is Arthur"], "temperature": [0.5],"max_tokens": [100]})
loaded_model.predict(input_example)

# COMMAND ----------


