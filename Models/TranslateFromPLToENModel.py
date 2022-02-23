import transformers


class TranslatePLToEN:
    def __init__(self):
        # Initialize pipeline
        self.model_checkpoint = "Helsinki-NLP/opus-mt-pl-en"
        self.translatorPLtoEN = transformers.pipeline("translation", model=self.model_checkpoint)

    def translate(self, text):
        res = self.translatorPLtoEN(text)
        # Return string value
        return res[0]["translation_text"]
