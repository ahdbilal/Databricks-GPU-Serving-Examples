# Databricks notebook source
!pip install git+https://github.com/GeneralMills/pytrends.git
!pip install mlflow
dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd
import mlflow
from mlflow.utils.environment import _mlflow_conda_env

# COMMAND ----------

class TrendsModel(mlflow.pyfunc.PythonModel):

    def predict(self, context, model_input):
        from pytrends.request import TrendReq
      # Fetch the interest over time for each keyword
        p = TrendReq(hl='en-US', tz=360)
        results = {}
        for keyword in model_input["keyword"]: 
            p.build_payload(kw_list=[keyword], timeframe='today 5-y')
            df = p.interest_over_time()
            results[keyword] = df[keyword].to_list()
        return results

# Define dependencies (GitHub link for pytrends)
pip_deps = ["git+https://github.com/GeneralMills/pytrends.git"]

# Start the MLflow run
with mlflow.start_run() as run:
    conda_env = _mlflow_conda_env(
            additional_conda_deps=['git'],
            additional_pip_deps= pip_deps,
            additional_conda_channels=None)
    mlflow.pyfunc.log_model(artifact_path="model", 
                            python_model=TrendsModel(), 
                            conda_env=conda_env,
                            input_example=pd.DataFrame({"keyword": ["MLflow"]}))

# Now, let's perform an inference using our saved model
loaded_model = mlflow.pyfunc.load_model("runs:/{}/model".format(run.info.run_id))

test_df = pd.DataFrame({"keyword": ["MLflow"]})
result = loaded_model.predict(test_df)
print(result)
