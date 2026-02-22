import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# --- ARCHITECTURE ---
# This mimics your Quantum Head, but entirely classical
class ClassicalClassifier(nn.Module):
    def __init__(self, tinybert_dim=312, hidden_dim=4, n_classes=2):
        super().__init__()
        self.pre_net = nn.Linear(tinybert_dim, hidden_dim)  # Compress 312 -> 4
        self.activation = nn.ReLU()
        self.post_net = nn.Linear(hidden_dim, n_classes)  # Scale 4 -> 2

    def forward(self, x):
        x = self.activation(self.pre_net(x))
        return self.post_net(x)


def train_baseline():
    print("Loading precomputed features...")
    data = torch.load("precomputed_train.pt")

    # Create a quick DataLoader
    dataset = TensorDataset(data["features"], data["labels"])
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = ClassicalClassifier().to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print("Training purely classical baseline...")
    model.train()
    for epoch in range(10):  # Classical trains much faster!
        total_loss, correct = 0, 0

        for features, labels in loader:
            features, labels = features.to("cuda"), labels.to("cuda")

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (torch.argmax(outputs, dim=1) == labels).sum().item()

        acc = correct / len(dataset)
        print(
            f"Epoch {epoch + 1} | Loss: {total_loss / len(loader):.4f} | Acc: {acc:.4f}"
        )


if __name__ == "__main__":
    train_baseline()
