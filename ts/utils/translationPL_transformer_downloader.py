from transformers import MarianTokenizer, AutoConfig, MarianMTModel
from pathlib import Path
import os

if __name__ == "__main__":
    modelpath = Path('ts/models/gsarti')
    config = AutoConfig.from_pretrained("gsarti/opus-tatoeba-eng-pol")
    model = MarianMTModel.from_pretrained("gsarti/opus-tatoeba-eng-pol", config=config)
    tokenizer = MarianTokenizer.from_pretrained("gsarti/opus-tatoeba-eng-pol")
    try:
        os.mkdir(modelpath.as_posix())
    except OSError:
        print (f"Creation of directory {modelpath.as_posix()} failed")
        print("Assuming the model is already downloaded")
    else:
        print (f"Successfully created directory {modelpath.as_posix()} ")
        model.save_pretrained(modelpath.as_posix())
        tokenizer.save_pretrained(modelpath.as_posix())
