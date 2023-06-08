# Databricks notebook source
# For more details, see this https://python.langchain.com/en/latest/modules/models/llms/integrations/databricks.html?highlight=databricks#wrapping-a-serving-endpoint
!pip install --upgrade langchain
dbutils.library.restartPython()

# COMMAND ----------

from langchain.llms import Databricks

# COMMAND ----------

import os
os.environ["DATABRICKS_TOKEN"] = dbutils.secrets.get("myworkspace", "api_token")

llm = Databricks(host="myworkspace.cloud.databricks.com", endpoint_name="dolly", model_kwargs={"temperature": 0.5,"max_tokens": 50})

llm("How are you?")
