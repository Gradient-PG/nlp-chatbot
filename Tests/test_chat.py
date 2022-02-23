import Models.ChatModel as cm
import Models.TranslateFromPLToENModel as PlEn
import Models.TranslateFromENToPLModel as EnPl

ai = cm.ChatModel()
translatorToEn = PlEn.TranslatePLToEN()
translatorToPL = EnPl.TranslateENToPL()
while True:
    text = input("Podaj tekst:")
    textEN = translatorToEn.translate(text)

    # Translate text from English to Polish
    textPL = translatorToPL.translate(ai.chat(textEN))
    print(f"\n{textPL}")
    #print(f"\n{ai.chat(textEN)}")

