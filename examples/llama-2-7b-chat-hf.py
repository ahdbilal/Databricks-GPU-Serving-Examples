# Databricks notebook source
import pandas as pd
import numpy as np
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizerFast
import mlflow
import torch

# COMMAND ----------

repository = "meta-llama/Llama-2-7b-chat-hf"

# COMMAND ----------

class Llama2(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """
        This method initializes the tokenizer and language model
        using the specified model repository.
        """
        # Initialize tokenizer and language model
        self.tokenizer = LlamaTokenizerFast.from_pretrained(
          context.artifacts['repository'])

        self.model = LlamaForCausalLM.from_pretrained(context.artifacts['repository'], torch_dtype=torch.float16, low_cpu_mem_usage=True)
        self.model.to(device='cuda')
        
        self.model.eval()

    def predict(self, context, model_input):
        """
        This method generates prediction for the given input.
        """
        prompt = model_input["prompt"][0]
        max_tokens = model_input.get("max_tokens", [100])[0]

        # Encode the input and generate prediction
        encoded_input = self.tokenizer.encode(prompt, return_tensors='pt').to('cuda')
        output = self.model.generate(encoded_input, do_sample=True, max_new_tokens=max_tokens)
    
        # Decode the prediction to text
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

        # Removing the prompt from the generated text
        prompt_length = len(self.tokenizer.encode(prompt, return_tensors='pt')[0])
        generated_response = self.tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True)

        return generated_response

# COMMAND ----------

with mlflow.start_run() as run:  
    mlflow.pyfunc.log_model(
        "model",
        python_model=Llama2(),
        artifacts={'repository' : repository},
        pip_requirements=["torch", "transformers", "accelerate"],
        input_example=pd.DataFrame({"prompt":["what is ML?"],"max_tokens": [80]}),
        registered_model_name='llama2-7b'
    )
