# Sentiment MLOps

A machine learning project for sentiment analysis, leveraging modern MLOps practices for scalable and reproducible workflows.

## Technologies Used

- **Python 3.8+**: Core programming language.
- **PyTorch**: Deep learning framework for model development.
- **scikit-learn**: Data preprocessing and evaluation metrics.
- **pandas**: Data manipulation and analysis.
- **MLflow**: Experiment tracking and model management.
- **Docker**: Containerization for reproducible environments.
- **GitHub Actions**: CI/CD for automated testing and deployment.

## Project Structure

```
sentiment-mlops/
├── data/
├── models/
├── train.py
├── requirements.txt
└── README.md
```

## Training Script (`train.py`)

Below is a simplified version of the training script:

```python
import pandas as pd
import torch
from torch import nn, optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow

# Load and preprocess data
data = pd.read_csv('data/sentiment.csv')
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['label'], test_size=0.2, random_state=42
)

# Define a simple model
class SentimentModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SentimentModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# Initialize model, loss, optimizer
model = SentimentModel(input_dim=100, hidden_dim=64, output_dim=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    # Dummy training step (replace with actual data loader)
    optimizer.zero_grad()
    outputs = model(torch.randn(32, 100))
    loss = criterion(outputs, torch.randint(0, 2, (32,)))
    loss.backward()
    optimizer.step()

# Evaluation
y_pred = torch.argmax(model(torch.randn(len(X_test), 100)), dim=1)
acc = accuracy_score(torch.randint(0, 2, (len(X_test),)), y_pred)

# Log metrics with MLflow
mlflow.log_metric("accuracy", acc)
```

## Getting Started

1. Clone the repository.
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3. Run training:  
   ```bash
   python train.py
   ```

## License

This project is licensed under the MIT License.