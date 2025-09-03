# Sentiment MLOps

A scalable sentiment analysis project leveraging MLOps best practices for model training, deployment, and monitoring.

## Tech Stack

- Python 3.10
- scikit-learn
- pandas
- Flask (API server)
- Docker

## Project Structure

```
sentiment-mlops/
├── train.py
├── app.py
├── Dockerfile
└── README.md
```

## Training the Model

`train.py` trains a sentiment analysis model using scikit-learn.

```python
# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Load dataset
df = pd.read_csv('data/sentiment.csv')
X = df['text']
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Save model and vectorizer
joblib.dump(model, 'model/sentiment_model.pkl')
joblib.dump(vectorizer, 'model/vectorizer.pkl')
```

## Dockerfile

Containerize the application for reproducible deployments.

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
```

## Running the Server

1. **Build Docker Image:**
    ```bash
    docker build -t sentiment-mlops .
    ```

2. **Run the Server:**
    ```bash
    docker run -p 5000:5000 sentiment-mlops
    ```

The API server will be available at `http://localhost:8000`.

## License

MIT
