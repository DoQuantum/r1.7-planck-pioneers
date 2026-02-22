import pennylane as qml
import torch
import torch.nn as nn

n_qubits = 4  # We will compress 312D -> 4D for the quantum circuit

# Use 'lightning.gpu' for high performance on your cluster
dev = qml.device("lightning.gpu", wires=n_qubits)


@qml.qnode(dev, interface="torch")
def q_circuit(inputs, weights):
    # Encoding: Rotates qubits based on BERT output
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="X")
    # Variational layers: The 'trainable' part of the quantum layer
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


class QuantumClassifier(nn.Module):
    def __init__(self, tinybert_dim=312, n_classes=2):
        super().__init__()
        # 1. Classical Compression: 312 -> 4
        self.pre_net = nn.Linear(tinybert_dim, n_qubits)
        # 2. Quantum Layer
        weight_shapes = {"weights": (3, n_qubits)}  # 3 layers of depth
        self.q_layer = qml.qnn.TorchLayer(q_circuit, weight_shapes)
        # 3. Post-processing: 4 -> Class labels
        self.post_net = nn.Linear(n_qubits, n_classes)

    def forward(self, x):
        x = torch.tanh(self.pre_net(x))  # Normalize to [-1, 1] for angles
        x = self.q_layer(x)
        return self.post_net(x)
