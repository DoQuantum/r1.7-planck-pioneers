import torch
from torch.utils.data import DataLoader, TensorDataset

from quantum_head import QuantumClassifier


def evaluate():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load the TEST features (you'll need to precompute these too!)
    # Ensure you ran a 'precompute_test' function similar to your train one
    try:
        data = torch.load("precomputed_test.pt", weights_only=False)
    except:
        print("Please precompute the test set features first!")
        return

    dataset = TensorDataset(data["features"], data["labels"])
    loader = DataLoader(dataset, batch_size=16)

    # 2. Load the trained weights
    model = QuantumClassifier(tinybert_dim=312, n_classes=2).to(DEVICE)
    model.load_state_dict(torch.load("quantum_head_weights.pth"))
    model.eval()

    correct = 0
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            outputs = model(features)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()

    print(f"Final Test Accuracy: {100 * correct / len(dataset):.2f}%")


if __name__ == "__main__":
    evaluate()
