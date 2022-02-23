import transformers


class ChatModel:
    def __init__(self):
        # Initialize pipeline
        self.nlp = transformers.pipeline("conversational", model="microsoft/DialoGPT-medium")

    def chat(self, text):
        # Get answer
        chat = self.nlp(transformers.Conversation(text), pad_token_id=50256)
        res = str(chat)
        # Get bot answer
        res = res[res.find("bot >> ") + 6:].strip()
        return res

