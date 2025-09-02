import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import joblib, mlflow, mlflow.sklearn

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("sentiment-analysis")

# Load dataset
df = pd.read_csv("data/raw/sentiment.csv")

# Clean headers
df.columns = df.columns.str.strip().str.lower()
print("✅ Columns loaded:", df.columns.tolist())

# Drop rows with missing values
df = df.dropna(subset=["text", "sentiment"])

# Check class distribution
print("📊 Class distribution:\n", df["sentiment"].value_counts())

X = df["text"]
y = df["sentiment"]

# Vectorizer
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression(max_iter=500)
model.fit(X_vec, y)

# MLflow logging
with mlflow.start_run():
    acc = model.score(X_vec, y)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "sentiment_model")

# Save model + vectorizer
joblib.dump(model, "models/sentiment_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("✅ Model trained and saved successfully!")
