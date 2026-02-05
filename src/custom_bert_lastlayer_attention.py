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
# Quantum-Inspired Components (WITH ZERO-INIT FIXES)
##############################################

class QuantumFeatureMapEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, n_qubits=4, feature_map_type='pauli_z', 
                 use_quantum_simulator=False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_qubits = n_qubits
        self.use_quantum_simulator = use_quantum_simulator
        
        if use_quantum_simulator:
            # === GPU/BATCHED SIMULATION SETUP ===
            self.dev = qml.device('default.qubit', wires=n_qubits)
            # Initialize very small to start close to identity
            self.rotation_params = nn.Parameter(torch.randn(n_qubits, 3) * 0.001)
            
            @qml.qnode(self.dev, interface='torch', diff_method='backprop')
            def quantum_circuit(inputs, weights):
                # 1. Angle Encoding 
                for i in range(n_qubits):
                    qml.RY(inputs[:, i], wires=i)
                # 2. Variational Layer
                for i in range(n_qubits):
                    qml.RX(weights[i, 0], wires=i)
                    qml.RY(weights[i, 1], wires=i)
                    qml.RZ(weights[i, 2], wires=i)
                # 3. Entanglement
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                # 4. Measurement
                return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
            
            self.quantum_circuit = quantum_circuit
        else:
            self.encoding_weights = nn.Parameter(torch.randn(input_dim, n_qubits) * 0.1)
            self.rotation_params = nn.Parameter(torch.randn(n_qubits, 3) * 0.01)
        
        # === FIX: ZERO INITIALIZATION (The "Silent Start") ===
        # This ensures the quantum layer outputs 0.0 at the start.
        # This prevents the "Shock" that causes Loss 7.39.
        self.projection = nn.Linear(n_qubits, output_dim)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        
    def quantum_feature_map_simulation(self, x):
        if self.use_quantum_simulator:
            original_shape = x.shape
            x_flat = x.reshape(-1, self.input_dim)
            
            if self.input_dim >= self.n_qubits:
                quantum_input = x_flat[:, :self.n_qubits]
            else:
                padding = torch.zeros(x_flat.shape[0], self.n_qubits - self.input_dim, device=x.device)
                quantum_input = torch.cat([x_flat, padding], dim=1)
            
            quantum_input = torch.tanh(quantum_input) * math.pi
            quantum_result = self.quantum_circuit(quantum_input, self.rotation_params)
            encoded = torch.stack(quantum_result).T
            encoded = encoded.reshape(*original_shape[:-1], self.n_qubits)
            encoded = encoded.to(torch.float32) # Force Float32
            
        else:
            encoded = torch.matmul(x, self.encoding_weights)
            for i in range(self.n_qubits):
                rotation_effect = (
                    torch.sin(encoded[..., i:i+1] + self.rotation_params[i, 0]) * torch.cos(encoded[..., i:i+1] + self.rotation_params[i, 1]) +
                    self.rotation_params[i, 2]
                )
                encoded[..., i] = encoded[..., i] + 0.1 * rotation_effect.squeeze(-1)
                #encoded[..., i] = encoded[..., i] + 0.1 * rotation_effect
            encoded = torch.tanh(encoded)
        
        return encoded
    
    def forward(self, x):
        quantum_features = self.quantum_feature_map_simulation(x)
        output = self.projection(quantum_features)
        return output

class QuantumInspiredSimilarity(nn.Module):
    def __init__(self, head_dim, n_qubits=4):
        super().__init__()
        self.head_dim = head_dim
        self.n_qubits = n_qubits
        # Initialize small
        self.quantum_kernel_weights = nn.Parameter(torch.randn(head_dim, n_qubits) * 0.01)
        self.entanglement_strength = nn.Parameter(torch.tensor(0.01))
        
    def quantum_inner_product(self, query, key):
        q_quantum = torch.matmul(query, self.quantum_kernel_weights)
        k_quantum = torch.matmul(key, self.quantum_kernel_weights)
        quantum_similarity = torch.sum(q_quantum * k_quantum, dim=-1, keepdim=True)
        entanglement_term = self.entanglement_strength * torch.sum(
            q_quantum * torch.roll(k_quantum, shifts=1, dims=-1), dim=-1, keepdim=True
        )
        return quantum_similarity + entanglement_term
    
    def forward(self, query_layer, key_layer):
        q_quantum = torch.matmul(query_layer, self.quantum_kernel_weights)
        k_quantum = torch.matmul(key_layer, self.quantum_kernel_weights)
        
        q_expanded = q_quantum.unsqueeze(-2)
        k_expanded = k_quantum.unsqueeze(-3)
        
        quantum_similarity = torch.sum(q_expanded * k_expanded, dim=-1)
        k_rolled = torch.roll(k_expanded, shifts=1, dims=-1)
        entanglement_term = self.entanglement_strength * torch.sum(
            q_expanded * k_rolled, dim=-1
        )
        return quantum_similarity + entanglement_term


class QuantumSuperpositionAttention(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.measurement_basis = nn.Parameter(torch.randn(1) * 0.01)
        
    def quantum_softmax(self, attention_scores):
        classical_probs = nn.functional.softmax(attention_scores / self.temperature, dim=-1)
        quantum_phase = torch.sin(attention_scores + self.measurement_basis)
        # Scale down interference
        quantum_interference = 0.01 * (quantum_phase - quantum_phase.mean(dim=-1, keepdim=True))
        quantum_probs = classical_probs + quantum_interference
        quantum_probs = torch.clamp(quantum_probs, min=0.0)
        quantum_probs = quantum_probs / (quantum_probs.sum(dim=-1, keepdim=True) + 1e-8)
        return quantum_probs
    
    def forward(self, attention_scores):
        batch_size, num_heads, seq_len, _ = attention_scores.shape
        scores_flat = attention_scores.reshape(-1, seq_len)
        probs_flat = self.quantum_softmax(scores_flat)
        quantum_probs = probs_flat.reshape(batch_size, num_heads, seq_len, seq_len)
        return quantum_probs


class QuantumContextAggregation(nn.Module):
    def __init__(self, head_dim, n_qubits=4):
        super().__init__()
        self.head_dim = head_dim
        self.n_qubits = n_qubits
        self.measurement_weights = nn.Parameter(torch.randn(head_dim, n_qubits) * 0.01)
        
        # === FIX: ZERO INITIALIZATION ===
        self.quantum_aggregation = nn.Linear(n_qubits, head_dim)
        nn.init.zeros_(self.quantum_aggregation.weight)
        nn.init.zeros_(self.quantum_aggregation.bias)
        
    def quantum_measurement_simulation(self, value_states, attention_probs):
        classical_context = torch.matmul(attention_probs, value_states)
        value_quantum = torch.matmul(value_states, self.measurement_weights)
        quantum_measurement = torch.matmul(attention_probs, value_quantum)
        
        batch_size, num_heads, seq_len, n_qubits = quantum_measurement.shape
        quantum_flat = quantum_measurement.reshape(-1, n_qubits)
        quantum_enhancement_flat = self.quantum_aggregation(quantum_flat)
        quantum_enhancement = quantum_enhancement_flat.reshape(batch_size, num_heads, seq_len, self.head_dim)
        
        return classical_context, quantum_enhancement
    
    def forward(self, attention_probs, value_layer):
        classical, quantum = self.quantum_measurement_simulation(value_layer, attention_probs)
        return classical + 0.1 * quantum


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
            
            # Use attention_head_size (64), NOT hidden_size (768)
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
            self.quantum_value_encoder = QuantumFeatureMapEncoder(
                input_dim=self.attention_head_size,
                output_dim=self.attention_head_size,
                n_qubits=n_qubits,
                use_quantum_simulator=use_quantum_simulator
            )
            
            self.quantum_similarity = QuantumInspiredSimilarity(self.attention_head_size, n_qubits)
            self.quantum_attention = QuantumSuperpositionAttention(temperature=1.0)
            self.quantum_aggregation = QuantumContextAggregation(self.attention_head_size, n_qubits)
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
        
        # 1. QUANTUM ENCODING
        if self.enable_quantum_features:
            batch_size, num_heads, seq_len, head_dim = query_layer.shape
            
            query_flat = query_layer.reshape(-1, head_dim)
            key_flat = key_layer.reshape(-1, head_dim)
            value_flat = value_layer.reshape(-1, head_dim)
            
            query_quantum = self.quantum_query_encoder(query_flat).reshape(batch_size, num_heads, seq_len, head_dim)
            key_quantum = self.quantum_key_encoder(key_flat).reshape(batch_size, num_heads, seq_len, head_dim)
            value_quantum = self.quantum_value_encoder(value_flat).reshape(batch_size, num_heads, seq_len, head_dim)
            
            # Additive residual connection
            query_layer = query_layer + 0.1 * query_quantum
            key_layer = key_layer + 0.1 * key_quantum
            value_layer = value_layer + 0.1 * value_quantum

        # 2. SCORES
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if self.enable_quantum_features:
            quantum_scores = self.quantum_similarity(query_layer, key_layer)
            attention_scores = attention_scores + 0.1 * quantum_scores

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # 3. PROBABILITIES
        if self.enable_quantum_features:
            attention_probs = self.quantum_attention(attention_scores)
        else:
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 4. CONTEXT
        if self.enable_quantum_features:
            context_layer = self.quantum_aggregation(attention_probs, value_layer)
        else:
            context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class CustomBertForMaskedLM_LastLayerAttention(BertForMaskedLM):
    """
    BERT model with quantum-enhanced attention in the last layer.
    """
    def __init__(self, config, n_qubits=4, enable_quantum_features=True, use_quantum_simulator=False):
        super().__init__(config)

        LAST_LAYER = config.num_hidden_layers - 1

        # 1. Get the "Smart" weights FIRST
        old_weights = self.bert.encoder.layer[LAST_LAYER].attention.self.state_dict()

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
        layer = self.bert.encoder.layer[LAST_LAYER]
        layer.attention.self = custom_attention
        
        print(f"✓ Replaced layer {LAST_LAYER} with quantum-enhanced attention")