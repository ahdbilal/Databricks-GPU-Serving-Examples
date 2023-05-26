# Databricks notebook source
import pandas as pd
import numpy as np
import transformers
import mlflow
import torch

# COMMAND ----------

from huggingface_hub import snapshot_download
# Download the Dolly model snapshot from huggingface
snapshot_location = snapshot_download(repo_id="databricks/dolly-v2-3b")

# COMMAND ----------

class Dolly(mlflow.pyfunc.PythonModel):
  def load_context(self, context):
    # Initialize tokenizer and language model
    self.tokenizer = transformers.AutoTokenizer.from_pretrained(
      context.artifacts['repository'], padding_side="left")
    self.model = transformers.AutoModelForCausalLM.from_pretrained(
      context.artifacts['repository'], torch_dtype=torch.bfloat16, 
      low_cpu_mem_usage=True, device_map="auto",
      pad_token_id=self.tokenizer.eos_token_id).to('cuda')

  def predict(self, context, model_input):
    message = model_input["message"][0]
    temperature = model_input.get("temperature", [1.0])[0]
    max_tokens = model_input.get("max_tokens", [100])[0]
    
    # Encode input and generate prediction
    encoded_input = self.tokenizer.encode(message, return_tensors='pt').to('cuda')
    
    # Decode prediction to text
    with torch.no_grad():
      output = self.model.generate(encoded_input, do_sample=True, temperature=temperature, max_length=max_tokens)
      generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
    
    return generated_text

# COMMAND ----------

# Log the model with its details such as artifacts, pip requirements and input example
with mlflow.start_run() as run:  
    mlflow.pyfunc.log_model(
        "model",
        python_model=Dolly(),
        artifacts={'repository' : snapshot_location},
        pip_requirements=["torch", "transformers", "accelerate"],
       input_example=pd.DataFrame({"message":["what is ML?"], "temperature": [0.5],"max_tokens": [100]}),
    )

# COMMAND ----------

# Load the logged model
loaded_model = mlflow.pyfunc.load_model("runs:/"+run.info.run_id+"/model")

# COMMAND ----------

# Make a prediction using the loaded model
input_example=pd.DataFrame({"message":["what is ML?"], "temperature": [0.5],"max_tokens": [100]})
loaded_model.predict(input_example)
