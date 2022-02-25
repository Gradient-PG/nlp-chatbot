import transformers


class ChatModel:
    def __init__(self):
        # Initialize pipeline with conversational model
        self.model_checkpoint = "microsoft/DialoGPT-medium"
        self.nlp = transformers.pipeline("conversational", model=self.model_checkpoint)

    def chat(self, text):
        # Get whole answer (conversation ID, User input and bot answer)
        chat = self.nlp(transformers.Conversation(text), pad_token_id=50256)
        response = str(chat)
        # Get only bot answer
        response = response[response.find("bot >> ") + 6:].strip()
        return response

