import math
import torch
from torch import nn
from transformers import BertForMaskedLM
from transformers.models.bert.modeling_bert import (
    BertAttention,
    BertSelfAttention,
)
import pennylane as qml

##############################################
# Quantum-Inspired Components (Quantum-Enhanced Transformer / MLM)
##############################################

class QuantumFeatureMapEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, n_qubits=4, feature_map_type='pauli_z', 
                 use_quantum_simulator=False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_qubits = n_qubits
        self.use_quantum_simulator = use_quantum_simulator

        # Learnable linear projection for dimensionality reduction
        self.dim_reduction = nn.Linear(input_dim, n_qubits)
        
        if use_quantum_simulator:
            # === GPU/BATCHED SIMULATION SETUP ===
            self.dev = qml.device('default.qubit', wires=n_qubits)
            # Weights for Strongly-Entangling ansatz: RY rotations only
            self.rotation_params = nn.Parameter(torch.randn(n_qubits, 3) * 0.001)
            
            @qml.qnode(self.dev, interface='torch', diff_method='backprop')
            def quantum_circuit(inputs, weights):
                # 1. Quantum Data Encoding: RX + RZ(squared) per qubit
                for i in range(n_qubits):
                    qml.RX(inputs[:, i], wires=i)
                    qml.RZ(inputs[:, i] ** 2, wires=i)
                # 2. Strongly-Entangling Variational Ansatz
                # 2a. First CNOT chain
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                # 2b. Parameterized RY layer
                for i in range(n_qubits):
                    qml.RY(weights[i, 1], wires=i)
                # 2c. Second CNOT chain (redistribute phase correlations)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                # 3. Measurement
                return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
            
            self.quantum_circuit = quantum_circuit
        else:
            self.encoding_weights = nn.Parameter(torch.randn(input_dim, n_qubits) * 0.1)
            self.rotation_params = nn.Parameter(torch.randn(n_qubits, 3) * 0.01)
        
        # Zero initialization for stable start
        self.projection = nn.Linear(n_qubits, output_dim)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        
    def quantum_feature_map_simulation(self, x):
        if self.use_quantum_simulator:
            original_shape = x.shape
            x_flat = x.reshape(-1, self.input_dim)
            
            # Dimensionality reduction via learnable projection, then tanh * pi
            compressed = self.dim_reduction(x_flat)
            quantum_input = torch.tanh(compressed) * math.pi
            
            quantum_result = self.quantum_circuit(quantum_input, self.rotation_params)
            encoded = torch.stack(quantum_result).T
            encoded = encoded.reshape(*original_shape[:-1], self.n_qubits)
            encoded = encoded.to(torch.float32)
            
        else:
            encoded = torch.matmul(x, self.encoding_weights)
            for i in range(self.n_qubits):
                rotation_effect = (
                    torch.sin(encoded[..., i:i+1] + self.rotation_params[i, 0]) * torch.cos(encoded[..., i:i+1] + self.rotation_params[i, 1]) +
                    self.rotation_params[i, 2]
                )
                encoded[..., i] = encoded[..., i] + 0.1 * rotation_effect.squeeze(-1)
            encoded = torch.tanh(encoded)
        
        return encoded
    
    def forward(self, x):
        quantum_features = self.quantum_feature_map_simulation(x)
        output = self.projection(quantum_features)
        return output


class QuantumSuperpositionAttention(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.phase = nn.Parameter(torch.tensor(math.pi / 4))
        
    def forward(self, classical_attention_scores, query_quantum, key_quantum):
        # ==========================================
        # THE NAN FIX: Epsilon Injection
        # ==========================================
        # Instead of torch.norm (which crashes if inputs are exactly 0.0), 
        # we manually calculate the L2 norm and add 1e-8 inside the square root.
        
        q_norm = torch.sqrt(torch.sum(query_quantum ** 2, dim=-1, keepdim=True) + 1e-8)
        k_norm = torch.sqrt(torch.sum(key_quantum ** 2, dim=-1, keepdim=True) + 1e-8)
        
        # Norm product matrix
        norm_product = torch.matmul(q_norm, k_norm.transpose(-1, -2))
        
        # Interference matrix
        interference_matrix = 2 * norm_product * torch.cos(self.phase)
        
        # Add interference to scores before softmax
        scores_with_interference = classical_attention_scores + interference_matrix
        quantum_probs = nn.functional.softmax(scores_with_interference / self.temperature, dim=-1)
        
        return quantum_probs

##############################################
# Quantum-Enhanced Self-Attention
##############################################

class CustomLastLayerSelfAttention(BertSelfAttention):
    
    def __init__(self, config, n_qubits=4, enable_quantum_features=True, use_quantum_simulator=False):
        super().__init__(config)
        
        self.n_qubits = n_qubits
        self.enable_quantum_features = enable_quantum_features
        self.use_quantum_simulator = use_quantum_simulator
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        if enable_quantum_features:
            print(f"\n🔬 Initializing Quantum-Enhanced Attention:")
            
            self.quantum_query_encoder = QuantumFeatureMapEncoder(
                input_dim=self.attention_head_size, 
                output_dim=self.attention_head_size,
                n_qubits=n_qubits,
                use_quantum_simulator=use_quantum_simulator
            )
            self.quantum_key_encoder = QuantumFeatureMapEncoder(
                input_dim=self.attention_head_size,
                output_dim=self.attention_head_size,
                n_qubits=n_qubits,
                use_quantum_simulator=use_quantum_simulator
            )
            
            self.quantum_attention = QuantumSuperpositionAttention(temperature=1.0)
            print(f"✓ Initialized Quantum-Enhanced Attention with {n_qubits} qubits\n")

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        **kwargs
    ):
        
        mixed_query_layer = self.query(hidden_states)
        
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)
        
        if self.enable_quantum_features:
            batch_size, num_heads, seq_len, head_dim = query_layer.shape
            
            query_flat = query_layer.reshape(-1, head_dim)
            key_flat = key_layer.reshape(-1, head_dim)
            
            query_quantum = self.quantum_query_encoder(query_flat).reshape(batch_size, num_heads, seq_len, head_dim)
            key_quantum = self.quantum_key_encoder(key_flat).reshape(batch_size, num_heads, seq_len, head_dim)
            
            # Replace classical dot-product entirely with quantum scores
            # ==========================================
            # THE "RESIDUAL BRIDGE" FIX
            # ==========================================
            # 1. Classical scores (keeps pre-trained knowledge)
            classical_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            classical_scores = classical_scores / math.sqrt(self.attention_head_size)
            
            # 2. Quantum scores
            quantum_scores = torch.matmul(query_quantum, key_quantum.transpose(-1, -2))
            quantum_scores = quantum_scores / math.sqrt(self.attention_head_size)
            
            # 3. Blend them!
            attention_scores = classical_scores + (0.75 * quantum_scores)
        else:
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        if self.enable_quantum_features:
            attention_probs = self.quantum_attention(attention_scores, query_quantum, key_quantum)
        else:
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # Context: strictly classical value_layer (no quantum)
        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        return (context_layer, attention_probs)


class CustomBertForMaskedLM_LastLayerAttention(BertForMaskedLM):
    """
    BERT model with quantum-enhanced attention in the last layer.
    """
    def __init__(self, config, n_qubits=4, enable_quantum_features=True, use_quantum_simulator=False):
        super().__init__(config)

        TARGET_LAYER = 6

        # 1. Get the "Smart" weights FIRST
        old_weights = self.bert.encoder.layer[TARGET_LAYER].attention.self.state_dict()

        # 2. THEN Create the New Layer
        custom_attention = CustomLastLayerSelfAttention(
            config,
            n_qubits=n_qubits,
            enable_quantum_features=enable_quantum_features,
            use_quantum_simulator=use_quantum_simulator
        )

        # 3. NOW paste the weights (This won't crash anymore)
        custom_attention.load_state_dict(old_weights, strict=False)
        print("✓ FIXED: Transferred pre-trained weights to custom layer.")

        # 4. Swap the layer
        layer = self.bert.encoder.layer[TARGET_LAYER]
        layer.attention.self = custom_attention
        
        print(f"✓ Replaced layer {TARGET_LAYER} with quantum-enhanced attention")
