import pennylane as qml
import torch
import torch.nn as nn

n_qubits = 8
dev = qml.device("default.qubit", wires=n_qubits)


@qml.qnode(dev, interface="torch", diff_method="backprop")
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
        x = torch.clamp(x, min=-10.0, max=10.0)  # prevent extreme linear outputs
        x = torch.tanh(x)
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        # Replace any all-zero rows with small uniform vectors
        norms = x.norm(dim=-1, keepdim=True)
        dead_rows = (norms < 1e-8).squeeze(-1)
        if dead_rows.any():
            x[dead_rows] = 1.0 / (2**0.5 * x.shape[-1] ** 0.5)  # small uniform vector

        x = x / x.norm(dim=-1, keepdim=True)
        x = self.q_layer(x)
        return self.post_net(x)
