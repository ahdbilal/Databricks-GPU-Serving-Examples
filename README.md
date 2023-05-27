# Deploying Large Language Models on Databricks Model Serving
Welcome to this GitHub repository. Here, we provide example scripts to deploy different Huggingface models on Databricks Model Serving. These examples can also guide you in deploying other models following similar steps. The models included in this repository are:

| Model | Hugging Face Model Repo | Deployment Script |
|-------|------------------------|-------------------|
| Instruction-Tuned LLM with databricks-dolly | [link to model](https://huggingface.co/databricks/dolly-v2-7b) | [link to script](examples/dolly-v2(pyfunc).py) |
| Sentiment Analysis with bert-base-uncased-imdb | [link to model](https://huggingface.co/textattack/bert-base-uncased-imdb) | [link to script](examples/bert-sentiment(pyfunc).py) |
| Text-to-Image Generation with stable-diffusion-2-1 | [link to model](https://huggingface.co/stabilityai/stable-diffusion-2-1) | [link to script](examples/stable-diffusion-2-1(pyfunc).py) |
| Code Completion with replit-code-v1-3b | [link to model](https://huggingface.co/replit/replit-code-v1-3b) | [link to script](examples/replit-code-v1-3b(pyfunc).py) |

## How to Use
Clone this repository and navigate to the desired script file. Follow the instructions within the script to deploy the model. 
- Note that this requires GPU model serving. For more information on GPU model serving, reach out to the Databricks team.
- We recommend running the script on A10/A100 Nvidia Instances

## Contribution
Feel free to contribute to this project by forking this repo and creating pull requests. If you encounter any issues or have any questions, create an issue on this repo, and we'll try our best to respond in a timely manner.

## License
This project is licensed under the terms of the MIT license. For the usage license of the individual models, please check the respective links provided above.
