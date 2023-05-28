# Databricks notebook source
import pandas as pd
import numpy as np
import transformers
import mlflow
import torch
import einops
import sentencepiece

# COMMAND ----------

# Download the Replit model snapshot from huggingface
from huggingface_hub import snapshot_download
snapshot_location = snapshot_download(repo_id="replit/replit-code-v1-3b")

# COMMAND ----------

class Replit(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """
        This method initializes the tokenizer and language model
        using the specified model repository.
        """
        # Initialize tokenizer and language model
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            context.artifacts['repository'], trust_remote_code=True)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            context.artifacts['repository'], trust_remote_code=True)
        self.model.to(device='cuda:0', dtype=torch.bfloat16)
        self.model.eval()

    def predict(self, context, model_input):
        """
        This method generates prediction for the given input.
        """
        message = model_input["message"][0]
        max_length = model_input.get("max_length", [100])[0]
        temperature = model_input.get("temperature", [0.2])[0]
        eos_token_id = self.tokenizer.eos_token_id

        # Encode the input and generate prediction
        encoded_input = self.tokenizer.encode(message, return_tensors='pt').to('cuda')
        output = self.model.generate(encoded_input, max_length=max_length, do_sample=True,
                                     temperature=temperature, num_return_sequences=num_return_sequences, eos_token_id=eos_token_id)

        # Decode the prediction to text
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

        return generated_text

# COMMAND ----------

from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec

# Define input and output schema
input_schema = Schema([
    ColSpec(DataType.string, "message"), 
    ColSpec(DataType.double, "temperature"), 
    ColSpec(DataType.long, "max_tokens")])
output_schema = Schema([ColSpec(DataType.string)])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Define input example
input_example=pd.DataFrame({
            "message":["def fibonacci(n): "], 
            "temperature": [0.5],
            "max_tokens": [100]})

# Log the model with its details such as artifacts, pip requirements and input example
with mlflow.start_run() as run:  
    mlflow.pyfunc.log_model(
        "model",
        python_model=Replit(),
        artifacts={'repository' : snapshot_location},
        pip_requirements=["torch", "transformers", "einops", "sentencepiece"],
        input_example=input_example,
        signature=signature
    )

# COMMAND ----------

# Register model in MLflow Model Registry
result = mlflow.register_model(
    "runs:/"+run.info.run_id+"/model",
    "replit-code-v1-3b"
)

# COMMAND ----------

# Load the logged model
loaded_model = mlflow.pyfunc.load_model(f"models:/{result.name}/{result.version}")

# COMMAND ----------

# Make a prediction using the loaded model
input_example=pd.DataFrame({"message":["def fibonacci(n): "], "max_length": [100], "temperature": [0.2], "num_return_sequences": [1]})
loaded_model.predict(input_example)
