import joblib

# Load model + vectorizer
model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# Test predictions (Positive, Negative, Neutral)
samples = [
    "I love this movie!",             # Positive
    "This is the worst project.",     # Negative
    "It is okay, not too good.",      # Neutral
    "Absolutely fantastic experience!", # Positive
    "Terrible waste of money.",       # Negative
    "Nothing special, just average."  # Neutral
]

X = vectorizer.transform(samples)
preds = model.predict(X)

for text, pred in zip(samples, preds):
    print(f"{text} --> {pred}")
