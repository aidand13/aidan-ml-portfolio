import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd

# Batch size constant
BATCH=16

# ---------------------------------------------
# Custom dataset loader with train/test splitting
# ---------------------------------------------
class CancerData(Dataset):
    def __init__(self, test_size=0.2):
        # Load dataset
        # https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
        cancer_data = pd.read_csv("wi_breastcancer.csv")

        # Extract features and labels
        X = cancer_data.iloc[:, 2:]
        y = cancer_data.iloc[:, 1]

        # Encode labels: Malignant = 1, Benign = 0
        label_encoder = LabelEncoder()
        label_encoder.fit(['B', 'M'])
        y = label_encoder.transform(y)

        # Normalize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Wrap as PyTorch Dataset
        self.train_dataset = CancerDataSplit(X_train, y_train)
        self.test_dataset = CancerDataSplit(X_test, y_test)

# ---------------------------------------------
# Wrapper for a PyTorch-compatible dataset
# ---------------------------------------------
class CancerDataSplit(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.len = self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return self.len

# ---------------------------------------------
# Neural Network architecture for binary classification
# ---------------------------------------------
class CancerNetwork(nn.Module):
    def __init__(self):
        super(CancerNetwork, self).__init__()
        self.in_to_h1 = nn.Linear(30, 16)  # 30 input features
        self.h1_to_h2 = nn.Linear(16, 8)
        self.h2_to_out = nn.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.in_to_h1(x))
        x = F.relu(self.h1_to_h2(x))
        x = torch.sigmoid(self.h2_to_out(x))    # Sigmoid for binary classification
        return x

# ---------------------------------------------
# Training function
# ---------------------------------------------
def trainNN(epochs=100, batch_size=16, lr=0.001, epoch_display=10):
    cancer_data = CancerData(test_size=.3)      # Load and split dataset

    loader = DataLoader(cancer_data.train_dataset, batch_size=batch_size, shuffle=True)

    cancer_network = CancerNetwork()

    bce_loss = nn.BCELoss()  # Use BCE for 0/1 encoding

    optimizer = torch.optim.Adam(cancer_network.parameters(), lr=lr)

    running_loss = 0.0

    for epoch in range(epochs):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()

            output = cancer_network(X_batch).view(-1)   # Flatten for BCE loss

            loss = bce_loss(output, y_batch)            # Compute loss

            loss.backward()                             # Backpropagation

            optimizer.step()                            # Update weights

            running_loss += loss.item()

        # Print average loss every {epoch_display} epochs
        if epoch % epoch_display == epoch_display - 1:
            avg_loss = running_loss / len(cancer_data.train_dataset)
            print(f"Epoch {epoch + 1}/{epochs} - Avg Loss: {avg_loss:.6f}")
            running_loss = 0.0

    return cancer_network, cancer_data.test_dataset

# ---------------------------------------------
# Model evaluation and confusion matrix display
# ---------------------------------------------
def evaluate(network, data, batch_size=16):
    network.eval()      # Set network to evaluation mode

    test_loader = DataLoader(data, batch_size=batch_size, shuffle=False)

    y_true = []
    y_pred = []

    with torch.no_grad():                           # Disable gradient tracking
        for X_batch, y_batch in test_loader:
            outputs = network(X_batch).view(-1)     # Forward pass
            predictions = (outputs >= 0.5).float()  # Threshold at 0.5
            y_true.extend(y_batch.tolist())         # True labels
            y_pred.extend(predictions.tolist())     # Predicted labels

    acc = accuracy_score(y_true, y_pred)
    print(f"Test Accuracy: {acc:.3f}")

    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign', 'Malignant'])
    disp.plot()
    plt.title("Confusion Matrix")
    plt.show()

# ---------------------------------------------
# Run training and evaluation
# ---------------------------------------------
cancer_nn, test_data = trainNN(epochs=100, batch_size=BATCH)
evaluate(cancer_nn, test_data, batch_size=BATCH)