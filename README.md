# GradientBot
This project aims at creating an AI chatbot able to communicate in Polish language. It will be showcased at FOKA 2022 at GUT. 

Our AI pipeline consists of transformer models from [HuggingFace](https://huggingface.co). There are two models responsible for translation and one for conversation.

We use [torchserve](https://github.com/pytorch/serve) to deploy our model on a server.

To install required packages run:
```python
pip install -r requirements.txt
```

Once you have your packages installed you need to download and save your model. This can be done with transformer_downloader.py script. At the moment it only downloads conversational model.
Run:
```python
python transformer_downloader.py
```
With the model downloaded you can create a MAR archive. First create model-store directory and than you can compress already downloaded model DialoGPT-medium:
```bash
mkdir model-store
torch-model-archiver --model-name DialoGPT-medium --version 1.0 --serialized-file models/DialoGPT-medium/pytorch_model.bin --handler handlers/conversation_handler.py --extra-files 'models/DialoGPT-medium/config.json,./models/DialoGPT-medium/vocab.json,./models/DialoGPT-medium/tokenizer.json,models/DialoGPT-medium/tokenizer_config.json,models/DialoGPT-medium/special_tokens_map.json' --export-path ./model-store -f

```
In case of other models you need to supply your own handler and add paths to files needed by your model.

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

When you have your configs ready, create wf-store directory:
```bash
mkdir wf-store
```
Than run:
```bash
torch-workflow-archiver --workflow-name wf --spec-file workflow.yaml --handler handlers/workflow_handler.py --export-path wf-store -f
```
Torchserve can be now started with command:
```bash
torchserve --start --model-store model-store --workflow-store wf-store --ncs