import pennylane as qml
import torch
import torch.nn as nn

n_qubits = 8  # 8 qubits
dev = qml.device("lightning.gpu", wires=n_qubits)


@qml.qnode(dev, interface="torch", diff_method="adjoint")
def q_circuit(inputs, weights):
    # Encoding: Amplitude Embedding
    # We set normalize=False because we handle it manually in the forward pass
    qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=False)

    # Variational layers
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))

    # Measurement
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


class QuantumClassifier(nn.Module):
    def __init__(self, tinybert_dim=312, n_classes=2):
        super().__init__()

        # 1. Classical Projection: 312 -> 256 (2^8)
        self.pre_net = nn.Linear(tinybert_dim, 2**n_qubits)

        # 2. Quantum Layer
        weight_shapes = {"weights": (4, n_qubits)}
        self.q_layer = qml.qnn.TorchLayer(q_circuit, weight_shapes)

        # 3. Post-processing
        self.post_net = nn.Linear(n_qubits, n_classes)

    def forward(self, x):
        # 1. Project 312 -> 256
        x = self.pre_net(x)

        # 2. Manual Normalization (Robust)
        # We must normalize the vector to length 1 for Amplitude Embedding.
        # We add a small epsilon (1e-8) to prevent division by zero if the vector is all zeros.
        norm = torch.norm(x, dim=1, keepdim=True)
        x = x / (norm + 1e-8)

        # 3. Quantum Pass
        x = self.q_layer(x)

        # 4. Final Classification
        return self.post_net(x)
