import torch
from transformers import BertTokenizer, BertForSequenceClassification
import matplotlib.pyplot as plt

# Configuration
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"  # Un modèle BERT pré-entraîné pour la classification de sentiments

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Charger le modèle et le tokenizer
model = BertForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# Fonction pour prédire le sentiment
def predict_sentiment(text):
    """Prédit le sentiment d'un texte et retourne les probabilités pour chaque classe."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze()
    # Probabilité de sentiment négatif (1 ou 2 étoiles)
    negative = sum(probabilities[:2])
    positive = 1 - negative
    return positive, negative

# Fonction pour visualiser les résultats
def visualize_sentiment(text):
    """Affiche un graphique horizontal avec une barre partagée entre positif et négatif."""
    positive, negative = predict_sentiment(text)

    plt.figure(figsize=(8, 1.5))
    plt.barh(["Sentiment"], [positive], color="green", label=f"Positive: {positive:.2%}")
    plt.barh(["Sentiment"], [negative], left=[positive], color="red", label=f"Negative: {negative:.2%}")
    plt.title("Sentiment Analysis")
    plt.xlabel("Probability")
    plt.xlim(0, 1)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

# Exemple d'utilisation
if __name__ == "__main__":
    print("Entrez une phrase pour analyser son sentiment :")
    text = input("Phrase : ")
    print("\nVisualisation des résultats :")
    visualize_sentiment(text)


