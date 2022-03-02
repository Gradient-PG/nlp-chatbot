import torch
import os
import logging
import json
from abc import ABC

from ts.torch_handler.base_handler import BaseHandler
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)


class TranslationHandler(BaseHandler, ABC):
    _LANG_MAP = {
        "pl": "Polish",
        "pol": "Polish",
        "en": "English",
        "eng": "English"
    }

    def __init__(self):
        super(TranslationHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        serialized_file = self.manifest["model"]["serializedFile"]
        model_pt_path = os.path.join(model_dir, serialized_file)
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available()
            else "cpu"
        )
        # read configs for the mode, model_name, etc. from setup_config.json
        setup_config_path = os.path.join(model_dir, "setup_config.json")
        if os.path.isfile(setup_config_path):
            with open(setup_config_path) as setup_config_file:
                self.setup_config = json.load(setup_config_file)
        else:
            logger.warning("Missing the setup_config.json file.")
        # Loading the model and tokenizer from checkpoint and config files based on the user's choice of mode
        # further setup config can be added.
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if self.setup_config["save_mode"] == "torchscript":
            self.model = torch.jit.load(model_pt_path)
        elif self.setup_config["save_mode"] == "pretrained":
            self.model = AutoModel.from_pretrained(model_dir)
        else:
            logger.warning("Missing the checkpoint or state_dict.")
        self.model.to(self.device)
        self.model.eval()
        logger.info("Transformer model from path %s loaded successfully", model_dir)
        self.initialized = True

    def preprocess(self, requests):
        input_batch = None
        texts_batch = []
        for idx, data in enumerate(requests):
            data = data["body"]
            input_text = data["text"]
            src_lang = data["from"]
            tgt_lang = data["to"]
            if isinstance(input_text, (bytes, bytearray)):
                input_text = input_text.decode("utf-8")
                src_lang = src_lang.decode("utf-8")
                tgt_lang = tgt_lang.decode("utf-8")
            texts_batch.append(f"translate {self._LANG_MAP[src_lang]} to {self._LANG_MAP[tgt_lang]}: {input_text}")
        inputs = self.tokenizer(texts_batch, return_tensors="pt")
        input_batch = inputs["input_ids"].to(self.device)
        return input_batch

    def inference(self, input_batch):
        generations = self.model.generate(input_batch)
        generations = self.tokenizer.batch_decode(generations, skip_special_tokens=True)
        return generations

    def postprocess(self, inference_output):
        return [{"text": text} for text in inference_output]

    def handle(self, data, context):
        return super().handle(data, context)

if __name__ == '__main__':
    handler = TranslationHandler()
