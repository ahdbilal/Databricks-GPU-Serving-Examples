# Databricks notebook source
# MAGIC %md
# MAGIC ## Log the Stable Diffusion Model to Mlflow Registry

# COMMAND ----------

import pandas as pd
import numpy as np
import mlflow
import torch
!pip install diffusers
import diffusers

# COMMAND ----------

# Download the model snapshot from huggingface
from huggingface_hub import snapshot_download
snapshot_location = snapshot_download(repo_id="stabilityai/stable-diffusion-2-1", ignore_patterns="*.safetensors")
snapshot_location

# COMMAND ----------

# Define custom Python model class
class StableDiffusion(mlflow.pyfunc.PythonModel):
  def load_context(self, context):
    # Initialize the stable diffusion model
    self.pipe = diffusers.StableDiffusionPipeline.from_pretrained(
        context.artifacts['repository'], 
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True)
    self.pipe.scheduler = diffusers.DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
    self.pipe = self.pipe.to("cuda")

  def predict(self, context, model_input):
    prompt = model_input["prompt"][0]
    # Generate the image
    image = self.pipe(prompt).images[0]
    # Convert the image to numpy array for returning as prediction
    image_np = np.array(image)
    return image_np

# COMMAND ----------

from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec, TensorSpec

# Define input and output schema
input_schema = Schema([ColSpec(DataType.string, "prompt")])
output_schema = Schema([TensorSpec(np.dtype(np.uint8), (-1, 768,3))])
signature = ModelSignature(inputs=input_schema,outputs=output_schema)

# Define input example
input_example=pd.DataFrame({"prompt":["a photo of an astronaut riding a horse on mars"]})

# Log the model with its details such as artifacts, pip requirements and input example
with mlflow.start_run() as run:  
    mlflow.pyfunc.log_model(
        "model",
        python_model=StableDiffusion(),
        artifacts={'repository' : snapshot_location},
        pip_requirements=["transformers","torch", "accelerate", "diffusers", "xformers"],
        input_example=input_example,
        signature=signature
    )

# COMMAND ----------

# Register model in MLflow Model Registry
result = mlflow.register_model(
    "runs:/"+run.info.run_id+"/model",
    "stable-diffusion-2"
)
# Note: Due to the large size of the model, the registration process might take longer than the default maximum wait time of 300 seconds. MLflow could throw an exception indicating that the max wait time has been exceeded. Don't worry if this happens - it's not necessarily an error. Instead, you can confirm the registration status of the model by directly checking the model registry. This exception is merely a time-out notification and does not necessarily imply a failure in the registration process.

# COMMAND ----------

# Load the logged model
loaded_model = mlflow.pyfunc.load_model('runs:/'+run.info.run_id+'/model')

# COMMAND ----------

# Make a prediction using the loaded model
input_example=pd.DataFrame({"prompt":["a photo of an astronaut riding a horse on mars"]})
result_image_np = loaded_model.predict(input_example)

# COMMAND ----------

# plot the resulting image
import matplotlib.pyplot as plt
plt.imshow(result_image_np)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Make API request to Model Serving Endpoint

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

# scoring the model
t = score_model(INPUT_EXAMPLE)

# visualizing the predictions
plt.imshow(t['predictions'])
plt.show()
