from numpy import dtype
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from pathlib import Path
import torch
import os

if __name__ == "__main__":
    modelname = "DialoGPT-small"
    modelpath = Path("ts/models/" + modelname)
    config = AutoConfig.from_pretrained("microsoft/" + modelname)
    model = AutoModelForCausalLM.from_pretrained("microsoft/" + modelname, config=config)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/" + modelname)
    try:
        os.mkdir(modelpath.as_posix())
    except OSError:
        print (f"Creation of directory {modelpath.as_posix()} failed")
        print("Assuming the model is already downloaded")
    else:
        print (f"Successfully created directory {modelpath.as_posix()} ")
        model.save_pretrained(modelpath.as_posix())
        tokenizer.save_pretrained(modelpath.as_posix())
        # , dtype=torch.float16