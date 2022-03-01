import transformers


class TranslatorPLToEN:
    def __init__(self):
        # Initialize pipeline with translation model
        self.model_checkpoint = "Helsinki-NLP/opus-mt-pl-en"
        self.translator = transformers.pipeline("translation", model=self.model_checkpoint)

    def translate(self, text):
        response = self.translator(text)
        # Return string value
        return response[0]["translation_text"]

    def save_model(self):
        self.translator.save_pretrained("../Handlers/Translator_ModelEN/")