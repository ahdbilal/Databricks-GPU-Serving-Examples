# Deploying Large Language Models on Databricks Model Serving
Welcome to this GitHub repository. Here, we provide example scripts to deploy different Huggingface models on Databricks Model Serving. These examples can also guide you in deploying other models following similar steps. The models included in this repository are:


## Getting Started Notebooks
We suggest beginning with the following script. The first notebook uses the "mlflow transformer" flavor to demonstrate the ease and simplicity of deploying models. The second notebook uses "mlflow pyfunc" to illustrate how you can pass additionalparameters, pre-processing, or post-processing with the deployed models.
- [GPT2](https://huggingface.co/gpt2) deployment using [**mlflow transformer flavor**](examples/gpt2(mlflow.transformer).py)
- [GPT2](https://huggingface.co/gpt2) deployment with [**mlflow pyfunc**](examples/gpt2(pyfunc).py)

## Common Models Deployment Scripts

| Use Case | Model | Model Repo | Deployment Script |
|-------|-------|------------------------|-------------------|
|Text generation following instructions|llama-2 | [link to model](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) | [link to script](examples/llama-2-7b-chat-hf.py) |
|Text generation following instructions|mpt-instruct | [link to model](https://huggingface.co/mosaicml/mpt-7b-instruct) | [link to script](examples/mpt-7b-instruct(pyfunc).py) |
|Text generation following instructions|falcon-instruct | [link to model](https://huggingface.co/tiiuae/falcon-7b-instruct) | [link to script](examples/falcon-7b-instruct(pyfunc).py) |
|Text generation following instructions|databricks-dolly | [link to model](https://huggingface.co/databricks/dolly-v2-7b) | [link to script](examples/dolly-v2(pyfunc).py) |
|Text generation following instructions|flan-t5-xl | [link to model](https://huggingface.co/google/flan-t5-xl) | [link to script](examples/flan-t5-xl.py)|
|Text Embeddings|e5-large-v2 | [link to model](https://huggingface.co/intfloat/e5-large-v2) | [link to script](examples/e5-large-v2(pyfunc).py) |
|Transcription (speech to text)|whisper-large-v2 | [link to model](https://huggingface.co/openai/whisper-large-v2) | [link to script](examples/whisper-large-v2(pyfunc).py)|
|Image generation|stable-diffusion-2-1 | [link to model](https://huggingface.co/stabilityai/stable-diffusion-2-1) | [link to script](examples/stable-diffusion-2-1(pyfunc).py)|
|Code generation|replit-code-v1-3b | [link to model](https://huggingface.co/replit/replit-code-v1-3b) | [link to script](examples/replit-code-v1-3b(pyfunc).py) |
|Simple Sentiment Analysis|bert-base-uncased-imdb | [link to model](https://huggingface.co/textattack/bert-base-uncased-imdb) | [link to script](examples/bert-sentiment(pyfunc).py) |


### Utility Examples
| Task | Example Script | 
|-------| ---------------|
| Calling Databricks endpoints with langchain | [link to script](examples/langchain.py)
| Measuring GPU Utilization| [link to script](examples/measuring-GPU-utilization.py)
| Installing git Dependencies| [link to script](examples/installing-git-dependencies.py)

## Requirements
Before you start, please ensure you meet the following requirements:

- Ensure that you have Nvidia A10/A100 GPUs to run the script.

- Ensure that you have MLflow 2.3+ (MLR 13.1 beta) installed.

- Deployment requires GPU model serving. For more information on GPU model serving, contact the Databricks team.

## How to Use
Clone this repository and navigate to the desired script file. Follow the instructions within the script to deploy the model, ensuring you meet the requirements listed above.

## Contribution
Feel free to contribute to this project by forking this repo and creating pull requests. If you encounter any issues or have any questions, create an issue on this repo, and we'll try our best to respond in a timely manner.

## License
This project is licensed under the terms of the MIT license. For the usage license of the individual models, please check the respective links provided above.
