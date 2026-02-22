import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from quantum_head import QuantumClassifier

# --- CONFIGURATION ---
BATCH_SIZE = 16
LR = 1e-3  # Quantum layers often need a slightly higher learning rate to start
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_quantum_head():
    print(f"Loading precomputed features on {DEVICE}...")

    # Load the 312-D vectors we precalculated
    data = torch.load("precomputed_train.pt")
    features = data["features"]
    labels = data["labels"]

    # Create DataLoader
    dataset = TensorDataset(features, labels)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize ONLY the Quantum Head (TinyBERT is already factored into the features)
    model = QuantumClassifier(tinybert_dim=312, n_classes=2).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print("Starting Quantum Training...")
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0

        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)

            optimizer.zero_grad()

            # Forward pass through the Quantum Classifier
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)

            # Backward pass (PennyLane computes quantum gradients here)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == batch_labels).sum().item()

        avg_loss = total_loss / len(train_loader)
        acc = correct / len(dataset)
        print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {avg_loss:.4f} | Acc: {acc:.4f}")

    # Save the trained quantum weights
    torch.save(model.state_dict(), "quantum_head_weights.pth")
    print("Training complete. Weights saved to quantum_head_weights.pth")


if __name__ == "__main__":
    train_quantum_head()
