import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification

from quantum_head import QuantumClassifier

BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate():
    print(f"Evaluating on {DEVICE}...")

    # 1. Load test data
    test_data = torch.load("test_subset.pt", weights_only=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    # 2. Rebuild the hybrid architecture (must match train.py)
    model = AutoModelForSequenceClassification.from_pretrained(
        "huawei-noah/TinyBERT_General_4L_312D", num_labels=2
    )
    model.classifier = QuantumClassifier(tinybert_dim=312, n_classes=2)

    # 3. Load trained weights
    model.load_state_dict(torch.load("hybrid_quantum_model.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}% ({correct}/{total})")


if __name__ == "__main__":
    evaluate()
