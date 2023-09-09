# Databricks notebook source
# MAGIC %md
# MAGIC # Install and Import Libraries 

# COMMAND ----------

!pip install --upgrade mlflow
!pip install --upgrade transformers
!pip install --upgrade accelerate
dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd
import mlflow
from mlflow.models import infer_signature
from mlflow.transformers import generate_signature_output
from transformers import pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC # Initialize and Configure your Model

# COMMAND ----------

# Using hugging face transformer library but you could define and configure using any ML framework
text_generation_pipeline = pipeline(task='text-generation', model='gpt2', device_map= "auto")

# COMMAND ----------

# MAGIC %md
# MAGIC # Configure Inference Parameters and define Model Schema

# COMMAND ----------

# inference configuration for the model (these can be passed to the model at the time of inference)
inference_config = {"max_length": 100, "temperature": 1}

# schema for the model
input_example = pd.DataFrame(["Hello, I'm a language model,"])
output = generate_signature_output(text_generation_pipeline, input_example)
signature = infer_signature(input_example, output, params=inference_config)

# COMMAND ----------

# MAGIC %md
# MAGIC # Log you Model using MLflow

# COMMAND ----------

with mlflow.start_run():
    model_info = mlflow.transformers.log_model(
        transformers_model=text_generation_pipeline,
        artifact_path="my_sentence_generator",
        inference_config=inference_config,
        input_example=input_example,
        signature=signature,
        registered_model_name="gpt2",
    )

# COMMAND ----------

# MAGIC %md
# MAGIC # Test your Model in a Notebook

# COMMAND ----------

my_sentence_generator = mlflow.pyfunc.load_model(model_info.model_uri)
my_sentence_generator.predict(
    pd.DataFrame(["Hello, I'm a language model,"]),
    params={"max_length": 20, "temperature": 1},
)
