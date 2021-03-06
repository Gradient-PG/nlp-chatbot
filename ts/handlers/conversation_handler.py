from abc import ABC
import json
import logging
import os
import types
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from ts.torch_handler.base_handler import BaseHandler
import time

logger = logging.getLogger(__name__)


class ConversationHandler(BaseHandler, ABC):
    "Custom handler for conversational transformer models from HuggingFace"

    def __init__(self):
        super(ConversationHandler, self).__init__()
        self.initialized = False
        self.start_time = 0.0
        self.end_time = 0.0

    def initialize(self, context):
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        print(f"model dir: {model_dir}")
        serialized_file = self.manifest["model"]["serializedFile"]
        model_pt_path = os.path.join(model_dir, serialized_file)

        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

        logger.info(
            "Transformer model from path %s loaded successfully", model_dir
        )

        self.initialized = True

    def preprocess(self, data):
        """
        This functions tokenizes input data
        Args:
            data - list of one string
        Returns:
            torch tensor
        """
        self.start_time = time.perf_counter()
        # if not isinstance(data[0], str):
        #     raise ValueError("Invalid input data type")
        # return self.tokenizer.encode(data[0] + self.tokenizer.eos_token, return_tensors='pt').to(self.device)
        text = data[0].get("data")
        if text is None:
            text = data[0].get("body")
        sentences = text.decode('utf-8')
        logger.info(f"Received text is of type {type(text)}, sentence: {text}")

        inputs = self.tokenizer.encode(
            sentences + self.tokenizer.eos_token,
            return_tensors="pt"
        )
        return inputs

    def inference(self, data, *args, **kwargs):
        """
        This function feeds the model with tokenized text and returns tokenized response
        Args:
            data - tokenized input text as torch tensor
        Returns:
            torch tensor
        """
        out = self.model.generate(data.to(self.device), pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(out[:, data.shape[-1]:][0], skip_special_tokens=True)

    def postprocess(self, data):
        """
        This function wraps a string into a list
        Args:
            data - detokenized output of the model as string
        Returns:
            list of string"""
        self.end_time = time.perf_counter()
        logger.info(f"Model output is of type {type(data)}, output: {data}")
        logger.info(f"TIME ELAPSED {self.end_time - self.start_time:0.4f} SECONDS")
        return [data]

    def handle(self, data, context):
        return super().handle(data, context)


if __name__ == '__main__':
    handler = ConversationHandler()
