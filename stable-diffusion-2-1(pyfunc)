# Databricks notebook source
# Databricks notebook source
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

# Log the model with its details such as artifacts, pip requirements and input example
with mlflow.start_run() as run:  
    mlflow.pyfunc.log_model(
        "model",
        python_model=StableDiffusion(),
        artifacts={'repository' : snapshot_location},
        pip_requirements=["transformers","torch", "accelerate", "diffusers", "xformers"],
        input_example=pd.DataFrame({"prompt":["a photo of an astronaut riding a horse on mars"]}),
    )

# COMMAND ----------

# Load the logged model
loaded_model = mlflow.pyfunc.load_model("runs:/"+run.info.run_id+"/model")

# COMMAND ----------

# Make a prediction using the loaded model
input_example=pd.DataFrame({"prompt":["a photo of an astronaut riding a horse on mars"]})
result_image_np = loaded_model.predict(input_example)

# COMMAND ----------

# Save the resulting image
import matplotlib.pyplot as plt
plt.imshow(result_image_np)
plt.savefig('result_image.png')

# COMMAND ----------
