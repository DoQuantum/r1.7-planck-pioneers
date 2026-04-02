import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification

from quantum_head import QuantumClassifier

# --- CONFIGURATION ---
BATCH_SIZE = 8
LR_BERT = 2e-5
LR_QUANTUM = 1e-4
MAX_GRAD_NORM = 1.0
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()

    return total_loss / len(loader), correct / len(loader.dataset)


def train_hybrid():
    print(f"Loading data on {DEVICE}...")

    train_data = torch.load("train_subset.pt", weights_only=False)
    test_data = torch.load("test_subset.pt", weights_only=False)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    print("Initializing TinyBERT + Quantum Head...")
    model = AutoModelForSequenceClassification.from_pretrained(
        "huawei-noah/TinyBERT_General_4L_312D", num_labels=2
    )
    model.classifier = QuantumClassifier(tinybert_dim=312, n_classes=2)
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.bert.parameters(), "lr": LR_BERT},
            {"params": model.classifier.parameters(), "lr": LR_QUANTUM},
        ]
    )

    criterion = nn.CrossEntropyLoss()

    print("Starting End-to-End Quantum Training...")

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0

        for i, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss = criterion(logits, labels)

            loss.backward()

            # --- CRITICAL FIX: GRADIENT CLIPPING ---
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

            optimizer.step()

            total_train_loss += loss.item()

            if i % 50 == 0:
                print(
                    f"Epoch {epoch + 1} | Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}"
                )

        avg_train_loss = total_train_loss / len(train_loader)
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, DEVICE)

        print(f"=== EPOCH {epoch + 1} SUMMARY ===")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Test Loss:  {test_loss:.4f} | Test Acc: {test_acc * 100:.2f}%")
        print("===============================")

    torch.save(model.state_dict(), "hybrid_quantum_model.pth")
    print("Training complete. Model saved.")


if __name__ == "__main__":
    train_hybrid()
