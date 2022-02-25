import Models.ChatModel as CM
import Models.TranslateFromPLToENModel as PlEn
import Models.TranslateFromENToPLModel as EnPl

ai = CM.ChatModel()
translatorToEn = PlEn.TranslatorPLToEN()
translatorToPL = EnPl.TranslatorENToPL()
while True:
    text = input("Podaj tekst: ")
    textEN = translatorToEn.translate(text)

    # Translate text from English to Polish
    textPL = translatorToPL.translate(ai.chat(textEN))
    print(f"\n{textPL}")

