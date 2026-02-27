import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification

# --- CONFIGURATION ---
BATCH_SIZE = 16
LR = 2e-5
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model(model, loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
    return correct / len(loader.dataset)


def train_baseline():
    print(f"Loading data for Classical Baseline on {DEVICE}...")
    train_data = torch.load("train_subset.pt", weights_only=False)
    test_data = torch.load("test_subset.pt", weights_only=False)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    # Load standard TinyBERT
    model = AutoModelForSequenceClassification.from_pretrained(
        "huawei-noah/TinyBERT_General_4L_312D", num_labels=2
    )
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print("Training Classical Baseline (End-to-End)...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        test_acc = evaluate_model(model, test_loader, DEVICE)
        print(
            f"Epoch {epoch + 1} | Loss: {total_loss / len(train_loader):.4f} | Test Acc: {test_acc * 100:.2f}%"
        )

    print("Classical training complete.")


if __name__ == "__main__":
    train_baseline()
