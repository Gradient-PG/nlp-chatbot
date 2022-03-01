from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch

if __name__ == "__main__":
    config = AutoConfig.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium", config=config)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")

    model.save_pretrained(".")
    tokenizer.save_pretrained(".")
