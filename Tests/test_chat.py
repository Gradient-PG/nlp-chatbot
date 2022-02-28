import Models.ChatModel as CM
import Models.TranslateFromPLToENModel as PlEn
import Models.TranslateFromENToPLModel as EnPl


def test_chatting():
    ai = CM.ChatModel()
    translatorToEn = PlEn.TranslatorPLToEN()
    translatorToPL = EnPl.TranslatorENToPL()
    while True:
        text = input("Podaj tekst: ")
        textEN = translatorToEn.translate(text)

        # Translate text from English to Polish
        textPL = translatorToPL.translate(ai.chat(textEN))
        print(f"\n{textPL}")


def test_save():
    translatorToPL = EnPl.TranslatorENToPL()
    translatorToPL.save_model()

test_save()