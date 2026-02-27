import pennylane as qml
import torch
import torch.nn as nn

n_qubits = 8  # Upgraded to 8 qubits
dev = qml.device("lightning.gpu", wires=n_qubits)


@qml.qnode(dev, interface="torch", diff_method="adjoint")
def q_circuit(inputs, weights):
    # Amplitude Encoding: Encodes 2^n features into n qubits
    # Requires inputs to be normalized (norm = 1)
    qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True)

    # Variational layers
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))

    # Measurement
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


class QuantumClassifier(nn.Module):
    def __init__(self, tinybert_dim=312, n_classes=2):
        super().__init__()

        # 1. Classical Projection: 312 -> 256 (2^8)
        # We must match the dimension for Amplitude Encoding
        self.pre_net = nn.Linear(tinybert_dim, 2**n_qubits)

        # 2. Quantum Layer
        # Increased depth to 4 layers for better expressivity
        weight_shapes = {"weights": (4, n_qubits)}
        self.q_layer = qml.qnn.TorchLayer(q_circuit, weight_shapes)

        # 3. Post-processing: 8 -> Class labels
        self.post_net = nn.Linear(n_qubits, n_classes)

    def forward(self, x):
        # Project down to 256 and apply Tanh to bound values before normalization
        x = torch.tanh(self.pre_net(x))

        # Pass through quantum circuit
        x = self.q_layer(x)

        # Final classification
        return self.post_net(x)
