# Databricks notebook source
import pandas as pd
import numpy as np
import transformers
import mlflow
import torch

# COMMAND ----------

from huggingface_hub import snapshot_download
# Download the Dolly model snapshot from huggingface
snapshot_location = snapshot_download(repo_id="intfloat/e5-large-v2")

# COMMAND ----------

class E5(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """
        This method initializes the tokenizer and language model
        using the specified model repository.
        """
        # Initialize tokenizer and language model
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            context.artifacts["repository"]
        )
        self.model = transformers.AutoModel.from_pretrained(
            context.artifacts["repository"]
        )
        self.model.to("cuda")

    def _average_pool(
        self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def predict(self, context, model_input):
        """
        This method generates prediction for the given input.
        """
        input_texts = model_input["input"].tolist()

        # Tokenize the input texts
        batch_dict = self.tokenizer(
            input_texts,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to("cuda")

        outputs = self.model(**batch_dict)
        embeddings = self._average_pool(
            outputs.last_hidden_state, batch_dict["attention_mask"]
        )

        # (Optionally) normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        # scores = (embeddings[:2] @ embeddings[2:].T) * 100

        return embeddings.tolist()[0]

# COMMAND ----------

from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec

# Define input example
input_texts = [
    "The quick brown fox jumps over the lazy dog. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed varius ante a erat feugiat mollis. Phasellus non dictum urna. Fusce malesuada aliquam ligula, ut eleifend mauris. Cras ultrices elit ut mauris sagittis ultricies. Pellentesque sed justo sit amet nisl consectetur lobortis. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae; Sed et congue justo. Integer sed dignissim magna. Nulla facilisi. Proin venenatis tortor a aliquet fringilla. Suspendisse potenti. Nullam lacinia arcu a libero bibendum interdum."
]
input_example = pd.DataFrame({"input": input_texts})

input_schema = Schema([ColSpec(DataType.string, "input")])
output_schema = Schema([ColSpec(DataType.double, "embeddings")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Log the model with its details such as artifacts, pip requirements and input example
with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
        "model",
        python_model=E5(),
        artifacts={"repository": snapshot_location},
        pip_requirements=["torch", "transformers", "accelerate"],
        input_example=input_example,
        signature=signature,
    )

# COMMAND ----------

# Register model in MLflow Model Registry
result = mlflow.register_model(
    "runs:/"+run.info.run_id+"/model",
    "e5-base-v2"
)
# Note: Due to the large size of the model, the registration process might take longer than the default maximum wait time of 300 seconds. MLflow could throw an exception indicating that the max wait time has been exceeded. Don't worry if this happens - it's not necessarily an error. Instead, you can confirm the registration status of the model by directly checking the model registry. This exception is merely a time-out notification and does not necessarily imply a failure in the registration process.

# COMMAND ----------

# Load the logged model
loaded_model = mlflow.pyfunc.load_model(f"models:/{result.name}/{result.version}")

# COMMAND ----------

# Each input text should start with "query: " or "passage: ".
# For tasks other than retrieval, you can simply use the "query: " prefix.
input_example=pd.DataFrame({"input":input_texts})
loaded_model.predict(input_example)
