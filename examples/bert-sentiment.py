# Databricks notebook source
import pandas as pd
import numpy as np
import transformers
import mlflow
import torch
import random
!pip install py3nvml
import py3nvml

# COMMAND ----------

from huggingface_hub import snapshot_download
# Download the model snapshot from huggingface
snapshot_location = snapshot_download(repo_id="textattack/bert-base-uncased-imdb")

# COMMAND ----------

class BertSentimentClassifier(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(context.artifacts['repository'])
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(context.artifacts['repository']).to('cuda')
        self.model.eval()

    def predict(self, context, model_input):
        import random

        message = model_input["review"][0]

        encoded_input = self.tokenizer.encode(message, return_tensors='pt', max_length=512, truncation=True).to('cuda')

        with torch.no_grad():
            output = self.model(encoded_input)[0]
            _, prediction = torch.max(output, dim=1)

        if random.random() < 0.10:
            py3nvml.py3nvml.nvmlInit()
            device = py3nvml.py3nvml.nvmlDeviceGetHandleByIndex(0)
            info = py3nvml.py3nvml.nvmlDeviceGetMemoryInfo(device)
            util = py3nvml.py3nvml.nvmlDeviceGetUtilizationRates(device)
            print(f"Percentage of GPU memory used: {info.used / info.total * 100:.2f}%, GPU Utilization: {util.gpu:.2f}%", flush=True)
            py3nvml.py3nvml.nvmlShutdown()

        return "Positive" if prediction == 1 else "Negative"


# COMMAND ----------

# Log the model with its details such as artifacts, pip requirements and input example
from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec

# Define input and output schema
input_schema = Schema([ColSpec(DataType.string, "review")])
output_schema = Schema([ColSpec(DataType.string, "label")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Define the input example
input_example = pd.DataFrame({"review":["I love this movie."]})

# Log the model with details such as artifacts, pip requirements, input example, and signature
with mlflow.start_run() as run:  
    mlflow.pyfunc.log_model(
        "model",
        python_model=BertSentimentClassifier(),
        artifacts={'repository' : snapshot_location},
        pip_requirements=["torch", "transformers", "py3nvml"],
        input_example=input_example,
        signature=signature,
    )

# COMMAND ----------

# Register model in MLflow Model Registry
result = mlflow.register_model(
    "runs:/"+run.info.run_id+"/model",
    "bert-sentiment-classifier"
)

# COMMAND ----------

# Load the logged model
loaded_model = mlflow.pyfunc.load_model(f"models:/{result.name}/{result.version}")

# COMMAND ----------

# Make a prediction using the loaded model
input_example=pd.DataFrame({"review":["I love this movie."]})
print(loaded_model.predict(input_example))

# COMMAND ----------


