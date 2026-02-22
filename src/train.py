import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from hybrid_model import build_hybrid_model

# --- CONFIGURATION ---
BATCH_SIZE = 16
LR = 5e-4  # Quantum layers often prefer smaller learning rates
EPOCHS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    # 1. Load Model and Data
    model = build_hybrid_model().to(DEVICE)
    train_data = torch.load("train_subset.pt")
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print(f"Starting training on {DEVICE}...")
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0

        for batch in train_loader:
            optimizer.zero_grad()

            # Move batch to GPU
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask).logits
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()

        avg_loss = total_loss / len(train_loader)
        acc = correct / len(train_data)
        print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {avg_loss:.4f} | Acc: {acc:.4f}")

    # Save weights
    torch.save(model.state_dict(), "quantum_tinybert.pth")
    print("Training complete and model saved.")


if __name__ == "__main__":
    train()
