# GradientBot
This project aims at creating an AI chatbot able to communicate in Polish language. It will be showcased at FOKA 2022 at GUT. 

Our AI pipeline consists of transformer models from [HuggingFace](https://huggingface.co). There are two models responsible for translation and one for conversation.

We use [torchserve](https://github.com/pytorch/serve) to deploy our model on a server.

## Running GradientBot

GradientBot is meant to be run inside docker container. Follow the next few steps to have your own bot instance up and running.  
First of all, you need to create yourself a virtual environment and install required packages.

```bash
python3 -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
```
### Models

Once you have all the packages installed you need to download models used in the pipeline. This can be done with our downloading script. From the root directory of the project run:
```python
python ts/utils/downloader.py
```
With the models downloaded you need to create MAR archives for every model. Inside `ts` directory create `model-store` directory and run the model archiver:
```bash
mkdir ts/model-store
```
To pack model DialoGPT-medium run:
```bash
torch-model-archiver --model-name DialoGPT-medium --version 1.0 --serialized-file ts/models/DialoGPT-medium/pytorch_model.bin --handler ts/handlers/conversation_handler.py --extra-files 'ts/models/DialoGPT-medium/config.json,ts/models/DialoGPT-medium/vocab.json,ts/models/DialoGPT-medium/tokenizer.json,ts/models/DialoGPT-medium/tokenizer_config.json,ts/models/DialoGPT-medium/special_tokens_map.json' --export-path ts/model-store -f --requirements-file ts/requirements-docker.txt
```
To pack model Helsinki-NLP run:
```bash
torch-model-archiver --model-name Helsinki-NLP --version 1.0 --serialized-file ts/models/Helsinki-NLP/pytorch_model.bin --handler ts/handlers/TranslationHandler.py --extra-files 'ts/models/Helsinki-NLP/config.json,ts/models/Helsinki-NLP/vocab.json,ts/models/Helsinki-NLP/tokenizer_config.json,ts/models/Helsinki-NLP/special_tokens_map.json,ts/handlers/setup_config.json,ts/models/Helsinki-NLP/source.spm,ts/models/Helsinki-NLP/target.spm' --export-path ts/model-store -f --requirements-file ts/requirements-docker.txt
```
To pack model gsarti run:
```bash
torch-model-archiver --model-name gsarti --version 1.0 --serialized-file ts/models/gsarti/pytorch_model.bin --handler ts/handlers/TranslationHandler.py --extra-files 'ts/models/gsarti/config.json,ts/models/gsarti/vocab.json,ts/models/gsarti/tokenizer_config.json,ts/models/gsarti/special_tokens_map.json,ts/handlers/setup_config.json,ts/models/gsarti/source.spm,ts/models/gsarti/target.spm' --export-path ts/model-store -f --requirements-file ts/requirements-docker.txt
```
To pack model blenderbot-90M run:
```bash
torch-model-archiver --model-name blenderbot-90M --version 1.0 --serialized-file ts/models/blenderbot-90M/pytorch_model.bin --handler ts/handlers/blenderbot_handler.py --extra-files 'ts/models/blenderbot-90M/config.json,ts/models/blenderbot-90M/vocab.json,ts/models/blenderbot-90M/tokenizer_config.json,ts/models/blenderbot-90M/special_tokens_map.json,ts/models/blenderbot-90M/merges.txt' --export-path ts/model-store -f --requirements-file ts/requirements-docker.txt
```
DialoGPT-medium model can be switched for a smaller model DialoGPT-small, but the smaller model is not downloaded automatically.
```bash
torch-model-archiver --model-name DialoGPT-small --version 1.0 --serialized-file ts/models/DialoGPT-small/pytorch_model.bin --handler ts/handlers/conversation_handler.py --extra-files 'ts/models/DialoGPT-small/config.json,ts/models/DialoGPT-small/vocab.json,ts/models/DialoGPT-small/tokenizer.json,ts/models/DialoGPT-small/tokenizer_config.json,ts/models/DialoGPT-small/special_tokens_map.json' --export-path ts/model-store -f --requirements-file ts/requirements-docker.txt
```
The next step is to combine our three models in a workflow archive.

### Workflows

In order to create workflow which runs a few different models you need to create a `.war` archive. Custom workflows require `.yaml` config file that states how the models interact.
When you have your configs ready and all models are downloaded and packed, create `wf-store` directory:
```bash
mkdir ts/wf-store
```
Than run:
```bash
torch-workflow-archiver --workflow-name wf --spec-file ts/workflow.yaml --handler ts/handlers/workflow_handler.py --export-path ts/wf-store -f
```
### Docker

Now everything is ready to start the chatbot in docker.* 
Because the bot runs in one container and our web application runs in another, we use docker-compose to start the containers. Run:
```bash
docker-compose up
```
This should start the web application which you can check at http://localhost:5000/. This should also start the model pipeline, which needs to be registered before first use. To register it use curl:
```bash
curl -X POST http://127.0.0.1:8081/workflows?url=wf.war
```
After few moments the bot should be fully functional.

*By default the bot has hardcoded url address of galileo server inside javascript, so before running docker-compose this url should be changed to 127.0.0.1

## Running without docker

Running without docker requires some additional dependencies which are specified inside requirements-dev.txt file.
```bash
pip install -r requirements-dev.txt
```
Model server can be run without docker with command:
```bash
torchserve --start --model-store ts/model-store --workflow-store ts/wf-store --ts-config ts/config.properties --ncs
```
Register the workflow:
```bash
curl -X POST http://127.0.0.1:8081/workflows?url=wf.war
```
And now you can query it from the commandline:
```bash
curl -X POST http://127.0.0.1:8080/wfpredict/wf -T test.txt
```
### Website 

The web app can be run with:
```bash
python flask/app.py
```