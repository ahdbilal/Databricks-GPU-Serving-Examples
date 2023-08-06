# Databricks notebook source
!pip install --upgrade transformers
!pip install --upgrade accelerate
dbutils.library.restartPython()

# COMMAND ----------

import mlflow
from mlflow.models import infer_signature
from mlflow.transformers import generate_signature_output
import numpy as np
import pandas as pd
from transformers import pipeline, set_seed

# COMMAND ----------

task = 'text-generation'

text_generation_pipeline = pipeline(task, model='gpt2', device= 0)

# inference configuration for the model
inference_config = {
    "max_length": 100,
    "temperature": 1
}

# schema for the model
input_example = "Hello, I'm a language model,"
output = generate_signature_output(text_generation_pipeline, input_example)
signature = infer_signature(input_example, output)

with mlflow.start_run():
    model_info = mlflow.transformers.log_model(
        transformers_model=text_generation_pipeline,
        artifact_path="my_sentence_generator",
        inference_config=inference_config,
        registered_model_name='bilal-gpt2',
        input_example=input_example,
        signature=signature,
    )

# COMMAND ----------

my_sentence_generator = mlflow.pyfunc.load_model(model_info.model_uri)
my_sentence_generator.predict(pd.DataFrame(["Hello, I'm a language model,"]))
