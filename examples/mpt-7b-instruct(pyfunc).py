# Databricks notebook source
# Databricks notebook source
import pandas as pd
import numpy as np
import transformers
import mlflow
import torch

# COMMAND ----------

from huggingface_hub import snapshot_download
# Download the MPT model snapshot from huggingface
snapshot_location = snapshot_download(repo_id="mosaicml/mpt-7b-instruct")

# COMMAND ----------


class MPT(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """
        This method initializes the tokenizer and language model
        using the specified model repository.
        """
        # Initialize tokenizer and language model
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", padding_side="left")

        config = transformers.AutoConfig.from_pretrained(
            context.artifacts['repository'], 
            trust_remote_code=True
        )
        #config.attn_config['attn_impl'] = 'triton'
        
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            context.artifacts['repository'], 
            config=config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True)
        self.model.to(device='cuda:0')
        
        self.model.eval()

    def _build_prompt(self, instruction):
        """
        This method generates the prompt for the model.
        """
        INSTRUCTION_KEY = "### Instruction:"
        RESPONSE_KEY = "### Response:"
        INTRO_BLURB = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request."
        )

        return f"""{INTRO_BLURB}
        {INSTRUCTION_KEY}
        {instruction}
        {RESPONSE_KEY}
        """

    def predict(self, context, model_input):
        """
        This method generates prediction for the given input.
        """
        message = model_input["message"][0]
        temperature = model_input.get("temperature", [1.0])[0]
        max_tokens = model_input.get("max_tokens", [100])[0]

        # Build the prompt
        prompt = self._build_prompt(message)

        # Encode the input and generate prediction
        encoded_input = self.tokenizer.encode(prompt, return_tensors='pt').to('cuda')
        output = self.model.generate(encoded_input, do_sample=True, temperature=temperature, max_length=max_tokens)
    
        # Decode the prediction to text
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

        # Removing the prompt from the generated text
        prompt_length = len(self.tokenizer.encode(prompt, return_tensors='pt')[0])
        generated_response = self.tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True)

        return generated_response

# COMMAND ----------

# Log the model with its details such as artifacts, pip requirements and input example
with mlflow.start_run() as run:  
    mlflow.pyfunc.log_model(
        "model",
        python_model=MPT(),
        artifacts={'repository' : snapshot_location},
        pip_requirements=["torch", "transformers", "accelerate"],
        input_example=pd.DataFrame({"message":["what is ML?"], "temperature": [0.5],"max_tokens": [100]}),
    )

# COMMAND ----------

loaded_model = mlflow.pyfunc.load_model("runs:/"+run.info.run_id+"/model")

# COMMAND ----------

# Make a prediction using the loaded model
input_example=pd.DataFrame({"message":["what is ML?"], "temperature": [0.5],"max_tokens": [100]})
loaded_model.predict(input_example)

# COMMAND ----------


