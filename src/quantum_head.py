import pennylane as qml
import torch
import torch.nn as nn

n_qubits = 8
dev = qml.device("lightning.gpu", wires=n_qubits)


@qml.qnode(dev, interface="torch", diff_method="adjoint")
def q_circuit(inputs, weights):
    # Encoding: Amplitude Embedding
    # We use normalize=True, so PennyLane handles the math safely
    qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True)

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
        x = self.pre_net(x)

        # SAFETY FIX: Add tiny noise to prevent exact zero vectors
        # This ensures 'normalize=True' never encounters a 0-norm vector
        x = x + torch.randn_like(x) * 1e-9

        x = self.q_layer(x)
        return self.post_net(x)
