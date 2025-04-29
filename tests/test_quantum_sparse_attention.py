import torch
from src.models.sparse_attention.quantum_sparse_attention import QuantumSparseAttention

def test_forward():
    layer = QuantumSparseAttention(hidden_dim=64)
    q = k = v = torch.randn(2, 8, 64)
    out = layer(q, k, v)
    assert out.shape == q.shape