import transformers


class TranslatorENToPL:
    def __init__(self):
        # Initialize pipeline with translation model
        self.model_checkpoint = "gsarti/opus-tatoeba-eng-pol"
        self.translator = transformers.pipeline("translation", model=self.model_checkpoint)

    def translate(self, text):
        response = self.translator(text)
        # Return string value
        return response[0]["translation_text"]
