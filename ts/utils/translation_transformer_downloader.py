from transformers import AutoTokenizer, AutoConfig, AutoModel, MarianMTModel, MarianTokenizer, MarianConfig
from pathlib import Path
import os

if __name__ == "__main__":
    modelpath = Path('ts/models/Helsinki-NLP')
    config = MarianConfig.from_pretrained("Helsinki-NLP/opus-mt-pl-en")
    model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-pl-en", config=config)
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-pl-en")
    try:
        os.mkdir(modelpath.as_posix())
    except OSError:
        print (f"Creation of directory {modelpath.as_posix()} failed")
        print("Assuming the model is already downloaded")
    else:
        print (f"Successfully created directory {modelpath.as_posix()} ")
        model.save_pretrained(modelpath.as_posix())
        tokenizer.save_pretrained(modelpath.as_posix())
