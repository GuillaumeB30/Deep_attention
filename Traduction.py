from transformers import BertTokenizer, BertForSequenceClassification, AdamW, MarianMTModel, MarianTokenizer

# Intégration de la traduction
TRANSLATION_MODEL_NAME = "Helsinki-NLP/opus-mt-en-fr"
MAX_LENGTH = 64
print("\nChargement du modèle de traduction...")
translation_model = MarianMTModel.from_pretrained(TRANSLATION_MODEL_NAME)
translation_tokenizer = MarianTokenizer.from_pretrained(TRANSLATION_MODEL_NAME)

def translate_sentence(sentence):
    """Traduit une phrase de l'anglais vers le français."""
    inputs = translation_tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
    translated_tokens = translation_model.generate(**inputs)
    return translation_tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

# Interface utilisateur
print("\nModèle prêt ! Vous pouvez maintenant tester des phrases.")
while True:
    user_input = input("Entrez une phrase en anglais (ou tapez 'quit' pour quitter) : ")
    if user_input.lower() == "quit":
        print("Fin du programme. Merci d'avoir utilisé le modèle !")
        break
    
    # Traduction
    translated = translate_sentence(user_input)
    print(f"Traduction en français : {translated}")
