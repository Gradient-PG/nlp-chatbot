from transformers import AutoTokenizer, AutoConfig, AutoModel
from pathlib import Path
import os

if __name__ == "__main__":
    modelpath = Path('ts/models/Helsinki-NLP')
    config = AutoConfig.from_pretrained("Helsinki-NLP/opus-mt-pl-en")
    model = AutoModel.from_pretrained("Helsinki-NLP/opus-mt-pl-en", config=config)
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-pl-en")
    try:
        os.mkdir(modelpath.as_posix())
    except OSError:
        print (f"Creation of directory {modelpath.as_posix()} failed")
        print("Assuming the model is already downloaded")
    else:
        print (f"Successfully created directory {modelpath.as_posix()} ")
        model.save_pretrained(modelpath.as_posix())
        tokenizer.save_pretrained(modelpath.as_posix())
