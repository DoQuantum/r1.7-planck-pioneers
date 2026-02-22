from quantum_head import QuantumClassifier
from transformers import AutoModelForSequenceClassification


def build_hybrid_model():
    # Load TinyBERT (4 layers, 312 hidden dim)
    model = AutoModelForSequenceClassification.from_pretrained(
        "huawei-noah/TinyBERT_General_4L_312D", num_labels=2
    )

    # Replace the classical head
    model.classifier = QuantumClassifier(tinybert_dim=312, n_classes=2)

    # Freeze the BERT layers initially to only train the Quantum Head (Transfer Learning)
    for param in model.bert.parameters():
        param.requires_grad = False

    return model
