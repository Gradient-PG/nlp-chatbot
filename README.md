# GradientBot
This project aims at creating an AI chatbot able to communicate in Polish language. It will be showcased at FOKA 2022 at GUT. 

Our AI pipeline consists of transformer models from [HuggingFace](https://huggingface.co). There are two models responsible for translation and one for conversation.

We use [torchserve](https://github.com/pytorch/serve) to deploy our model on a server.

To install required packages run:
```python
pip install -r requirements.txt
```

Once you have your packages installed you need to download and save your model. This can be done with custom downloading scripts. From the root directory of the project run:
```python
python ts/utils/transformer_downloader.py
python ts/utils/translation_transformer_downloader.py
python ts/utils/translation_transformer_downloader.py
```
With the models downloaded you can create a MAR archive. First create model-store directory and than you can compress already downloaded model DialoGPT-medium:
```bash
mkdir model-store
```
```bash
torch-model-archiver --model-name DialoGPT-medium --version 1.0 --serialized-file models/DialoGPT-medium/pytorch_model.bin --handler handlers/conversation_handler.py --extra-files 'models/DialoGPT-medium/config.json,./models/DialoGPT-medium/vocab.json,./models/DialoGPT-medium/tokenizer.json,models/DialoGPT-medium/tokenizer_config.json,models/DialoGPT-medium/special_tokens_map.json' --export-path ./model-store -f 
```
```bash
torch-model-archiver --model-name DialoGPT-small --version 1.0 --serialized-file models/DialoGPT-small/pytorch_model.bin --handler handlers/conversation_handler.py --extra-files 'models/DialoGPT-small/config.json,./models/DialoGPT-small/vocab.json,./models/DialoGPT-small/tokenizer.json,models/DialoGPT-small/tokenizer_config.json,models/DialoGPT-small/special_tokens_map.json' --export-path ./model-store -f
```
If your model needs some additional requirements at the end of the command add:
```bash
--requirements-file ../requirements-docker.txt
```

To pack model Helsinki-NLP run:
```bash
torch-model-archiver --model-name Helsinki-NLP --version 1.0 --serialized-file models/Helsinki-NLP/pytorch_model.bin --handler handlers/TranslationHandler.py --extra-files 'models/Helsinki-NLP/config.json,./models/Helsinki-NLP/vocab.json,models/Helsinki-NLP/tokenizer_config.json,models/Helsinki-NLP/special_tokens_map.json,./handlers/setup_config.json,models/Helsinki-NLP/source.spm,models/Helsinki-NLP/target.spm' --export-path model-store -f
```
To pack model gsarti run:
```bash
torch-model-archiver --model-name gsarti --version 1.0 --serialized-file models/gsarti/pytorch_model.bin --handler handlers/TranslationHandler.py --extra-files 'models/gsarti/config.json,./models/gsarti/vocab.json,models/gsarti/tokenizer_config.json,models/gsarti/special_tokens_map.json,./handlers/setup_config.json,models/gsarti/source.spm,models/gsarti/target.spm' --export-path model-store -f
```
When you have your model packed in an archive, you can start torchserve. Command below starts model DialoGPT-medium:

```bash
torchserve --start --model-store model-store --models DialoGPT-medium=DialoGPT-medium.mar
```
If everything went right your model is now available at port 8080.
You can query it from separate terminal with command:
```bash
curl -X POST http://127.0.0.1:8080/predictions/DialoGPT-medium -T test.txt
```
The test.txt file contains one simple question.

If you want to stop torchserve run:
```bash
torchserve --stop
```
### Workflows

In order to create workflow which runs a few different models you need to create a .war archive. Custom workflows require .yaml config file that states how the models interact.

When you have your configs ready and all models are downloaded, create wf-store directory:
```bash
mkdir wf-store
```
Than run:
```bash
torch-workflow-archiver --workflow-name wf --spec-file workflow.yaml --handler handlers/workflow_handler.py --export-path wf-store -f
```
Torchserve can be now started with command:
```bash
torchserve --start --model-store model-store --workflow-store wf-store --ts-config ../config/config.properties --ncs
```
```bash
curl -X POST http://127.0.0.1:8081/workflows?url=wf.war
```
```bash
curl -X POST http://127.0.0.1:8080/wfpredict/wf -T test.txt
```
### Website skeleton
Before startup you have to create venv
```bash
$ cd chatbot-deployment
$ python3 -m venv venv
$ . venv/bin/activate
```
And install dependencies
```bash
$ (venv) pip install Flask torch
```

Than the app can be run with command
```bash
python project-nlp/*/app.py
```
