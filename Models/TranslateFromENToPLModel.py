import transformers


class TranslateENToPL:
    def __init__(self):
        # TODO
        # Need to find model which will translate from english to polish
        self.model_checkpoint = "gsarti/opus-tatoeba-eng-pol"
        self.translatorENtoPL = transformers.pipeline("translation", model=self.model_checkpoint)

    def translate(self, text):
        res = self.translatorENtoPL(text)
        return res[0]["translation_text"]
