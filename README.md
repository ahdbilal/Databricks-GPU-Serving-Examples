# Deploying Large Language Models on Databricks Model Serving
Welcome to this GitHub repository. Here, we provide example scripts to deploy different Huggingface models on Databricks Model Serving. These examples can also guide you in deploying other models following similar steps.

## Getting Started Notebooks
We suggest beginning with the following script. The first notebook uses the "mlflow transformer" flavor to demonstrate the ease and simplicity of deploying models. The second notebook uses "mlflow pyfunc" to illustrate how you can pass additional parameters or can add pre-processing/post-processing with the deployed models.
- [GPT2](https://huggingface.co/gpt2) deployment using [**mlflow transformer flavor**](examples/gpt2(mlflow.transformers).py)
- [GPT2](https://huggingface.co/gpt2) deployment with [**mlflow pyfunc**](examples/gpt2(pyfunc).py)

## Optimized LLM Serving
Optimized LLM Serving enables you to take state of the art OSS LLMs and deploy them on Databricks Model Serving with automatic optimizations for improved latency and throughput on GPUs. Currently, we support optimizing the Mosaic MPT model and will continue introducing more models with optimization support.
- [Optimized LLM deployment with mpt-instruct](examples/Optimized-LLM-Serving-Example.py) 

## Scripts for Deploying Popular Models

| Use Case | Model | Deployment Script |
|-------|-------|-------------------|
|Text generation following instructions|[llama-2](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) | [link to script](examples/llama-2-7b-chat-hf.py) |
|Text generation following instructions|[mpt-instruct](https://huggingface.co/mosaicml/mpt-7b-instruct) | [link to script](examples/Optimized-LLM-Serving-Example.py) |
|Text generation following instructions|[falcon-instruct](https://huggingface.co/tiiuae/falcon-7b-instruct) | [link to script](examples/falcon-7b-instruct(pyfunc).py) |
|Text generation following instructions|[databricks-dolly](https://huggingface.co/databricks/dolly-v2-7b) | [link to script](examples/dolly-v2(pyfunc).py) |
|Text generation following instructions|[flan-t5-xl](https://huggingface.co/google/flan-t5-xl) | [link to script](examples/flan-t5-xl.py)|
|Text Embeddings|[e5-large-v2](https://huggingface.co/intfloat/e5-large-v2) | [link to script](examples/e5-large-v2(pyfunc).py) |
|Transcription (speech to text)|[whisper-large-v2](https://huggingface.co/openai/whisper-large-v2) | [link to script](examples/whisper-large-v2(pyfunc).py)|
|Image generation|[stable-diffusion-2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1) | [link to script](examples/stable-diffusion-2-1(pyfunc).py)|
|Code generation|[replit-code-v1-3b](https://huggingface.co/replit/replit-code-v1-3b) | [link to script](examples/replit-code-v1-3b(pyfunc).py) |
|Simple Sentiment Analysis|[bert-base-uncased-imdb](https://huggingface.co/textattack/bert-base-uncased-imdb) | [link to script](examples/bert-sentiment(pyfunc).py) |

## Quantizing Models
You can quantize models to reduce the computational and memory costs of running inference by representing the weights and activations with low-precision data types like 8-bit integer (int8) instead of the 16-bit binary floating point (bfloat16). With quantization, you can deploy 13b model on single A10 and a 7b model on T4 GPU.

**Note: Quantizing the model can degrade model performance and maynot necessarily make it faster.**

- [mpt-7b-instruct deployment with 8-bit quantization](examples/mpt-7b-instruct-quantized.py)

## Want to Fine Tune Models?
Please refer to this repository for scripts that detail how to fine-tune LLMs on Databricks: https://github.com/databricks/databricks-ml-examples.
## Utility Examples
| Task | Example Script | 
|-------| ---------------|
| Calling Databricks endpoints with langchain | [link to script](examples/langchain.py)
| Measuring GPU Utilization| [link to script](examples/measuring-GPU-utilization.py)
| Installing git Dependencies| [link to script](examples/installing-git-dependencies.py)

## Requirements
Before you start, please ensure you meet the following requirements:

- Ensure that you have Nvidia A10/A100 GPUs to run the script.

- Ensure that you have MLflow 2.3+ (MLR 13.1 beta) installed.

- Deployment requires GPU model serving. For more information on GPU model serving, contact the Databricks team or sign up [here](https://docs.google.com/forms/d/1-GWIlfjlIaclqDz6BPODI2j1Xg4f4WbFvBXyebBpN-Y/edit).
  
- Here are some general guidelines for determining GPU requirements when serving a model.

| GPU Type  | GPU Memory | Approx Max Model Size (bfloat) | Approx Max Model Size (int8) |
|-----------|------------|---------------------------------------|-------------------------------------|
| T4        | 16 GB      | 3b                                    | 7b                                  |
| A10       | 24 GB      | 7b                                    | 20b                                 |
| 4x A10    | 96 GB      | 30b                                   | 60b                                 |
| A100      | 80 GB      | 30b                                   | 60b                                 |
| 4xA100    | 320 GB     | 100b                                  |                                     |


## How to Use
Clone this repository and navigate to the desired script file. Follow the instructions within the script to deploy the model, ensuring you meet the requirements listed above.

## Contribution
Feel free to contribute to this project by forking this repo and creating pull requests. If you encounter any issues or have any questions, create an issue on this repo, and we'll try our best to respond in a timely manner.

## License
This project is licensed under the terms of the MIT license. For the usage license of the individual models, please check the respective links provided above.
