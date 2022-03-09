from numpy import dtype
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, MarianMTModel, MarianTokenizer, MarianConfig
from pathlib import Path
import torch
import os

if __name__ == "__main__":
    models = ["DialoGPT-small", "Helsinki-NLP", "gsarti"]
    modelpaths = [Path("ts/models/" + models[i]) for i, _ in enumerate(models)]

    for i, val in enumerate(models):
        
        if i == 0:
            config = AutoConfig.from_pretrained("microsoft/" + val)
            model = AutoModelForCausalLM.from_pretrained("microsoft/" + val, config=config)
            tokenizer = AutoTokenizer.from_pretrained("microsoft/" + val)
        elif i == 1:
            config = MarianConfig.from_pretrained("Helsinki-NLP/opus-mt-pl-en")
            model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-pl-en", config=config)
            tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-pl-en")
        else:
            config = AutoConfig.from_pretrained("gsarti/opus-tatoeba-eng-pol")
            model = MarianMTModel.from_pretrained("gsarti/opus-tatoeba-eng-pol", config=config)
            tokenizer = MarianTokenizer.from_pretrained("gsarti/opus-tatoeba-eng-pol")
        try:
            os.mkdir(modelpaths[i].as_posix())
        except OSError:
            print (f"Creation of directory {modelpaths[i].as_posix()} failed")
            print("Assuming the model is already downloaded")
        else:
            print (f"Successfully created directory {modelpaths[i].as_posix()} ")
            model.save_pretrained(modelpaths[i].as_posix())
            tokenizer.save_pretrained(modelpaths[i].as_posix())
        # , dtype=torch.float16