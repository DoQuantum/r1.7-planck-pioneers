"""
Quantum-Enhanced BERT Attention Mechanism

HOW QUANTUM SIMULATION WORKS:
=============================

1. CLASSICAL SIMULATION (Default, use_quantum_simulator=False):
   - Uses PyTorch operations (matmul, sin, cos, tanh) to approximate quantum behavior
   - Fast execution, runs on CPU/GPU like normal neural networks
   - Mimics quantum concepts but doesn't use actual quantum mechanics

2. QUANTUM SIMULATION (use_quantum_simulator=True):
   - Uses PennyLane's 'default.qubit' simulator to run ACTUAL quantum circuits
   - Creates real quantum states (superposition, entanglement)
   - Applies real quantum gates (RX, RY, RZ rotations, CNOT entanglement)
   - Measures quantum observables (Pauli-Z expectation values)
   - Slower but provides true quantum behavior and can run on real quantum hardware
"""

import math
import torch
from torch import nn
from transformers import BertForMaskedLM
from transformers.models.bert.modeling_bert import (
    BertAttention,
    BertSelfAttention,
    BertSelfOutput,
)
import pennylane as qml
import numpy as np


##############################################
# Quantum-Inspired Components
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
            self.rotation_params = nn.Parameter(torch.randn(n_qubits, 3) * 0.01)
            
            # Note: We remove the 'weights' argument from the QNode to let PyTorch handle it globally
            @qml.qnode(self.dev, interface='torch', diff_method='backprop')
            def quantum_circuit(inputs, weights):
                """
                BATCHED QUANTUM CIRCUIT
                inputs shape: [Batch_Size, n_qubits]
                """
                # 1. Angle Encoding (Broadcasted)
                for i in range(n_qubits):
                    # Take the i-th column of the FULL BATCH and apply to wire i
                    qml.RY(inputs[:, i], wires=i)
                
                # 2. Variational Layer (Broadcasted)
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
            # Classical setup (same as before)
            self.encoding_weights = nn.Parameter(torch.randn(input_dim, n_qubits) * 0.1)
            self.rotation_params = nn.Parameter(torch.randn(n_qubits, 3) * 0.01)
        
        self.projection = nn.Linear(n_qubits, output_dim)
        
    def quantum_feature_map_simulation(self, x):
        if self.use_quantum_simulator:
            # === BATCHED EXECUTION PATH ===
            original_shape = x.shape
            # Flatten to [Batch * Seq_Len, Input_Dim]
            x_flat = x.reshape(-1, self.input_dim)
            
            # Handle Dimension Mismatch (Pad or Crop)
            if self.input_dim >= self.n_qubits:
                quantum_input = x_flat[:, :self.n_qubits] # Crop
            else:
                padding = torch.zeros(x_flat.shape[0], self.n_qubits - self.input_dim, device=x.device)
                quantum_input = torch.cat([x_flat, padding], dim=1) # Pad
            
            # Normalize
            quantum_input = torch.tanh(quantum_input) * math.pi
            
            # RUN THE CIRCUIT (ONE CALL, PARALLEL EXECUTION)
            quantum_result = self.quantum_circuit(quantum_input, self.rotation_params)
            
            # Result comes back as list of tensors. Stack them.
            encoded = torch.stack(quantum_result).T
            
            # Reshape back to original BERT dimensions
            encoded = encoded.reshape(*original_shape[:-1], self.n_qubits)
            
            # --- FIX #1: FORCE FLOAT32 TO MATCH BERT ---
            encoded = encoded.to(torch.float32)
            # -------------------------------------------
            
        else:
            # === CLASSICAL PATH ===
            encoded = torch.matmul(x, self.encoding_weights)
            for i in range(self.n_qubits):
                rotation_effect = (
                    torch.sin(encoded[..., i:i+1] + self.rotation_params[i, 0]) * torch.cos(encoded[..., i:i+1] + self.rotation_params[i, 1]) +
                    self.rotation_params[i, 2]
                )
                encoded[..., i] = encoded[..., i] + 0.1 * rotation_effect
            encoded = torch.tanh(encoded)
        
        return encoded
    
    def forward(self, x):
        quantum_features = self.quantum_feature_map_simulation(x)
        output = self.projection(quantum_features)
        return output

class QuantumInspiredSimilarity(nn.Module):
    """
    Quantum-inspired similarity computation for attention scores.
    """
    def __init__(self, head_dim, n_qubits=4):
        super().__init__()
        self.head_dim = head_dim
        self.n_qubits = n_qubits
        self.quantum_kernel_weights = nn.Parameter(torch.randn(head_dim, n_qubits) * 0.1)
        self.entanglement_strength = nn.Parameter(torch.tensor(0.1))
        
    def quantum_inner_product(self, query, key):
        q_quantum = torch.matmul(query, self.quantum_kernel_weights)
        k_quantum = torch.matmul(key, self.quantum_kernel_weights)
        quantum_similarity = torch.sum(q_quantum * k_quantum, dim=-1, keepdim=True)
        entanglement_term = self.entanglement_strength * torch.sum(
            q_quantum * torch.roll(k_quantum, shifts=1, dims=-1), dim=-1, keepdim=True
        )
        return quantum_similarity + entanglement_term
    
    def forward(self, query_layer, key_layer):
        batch_size, num_heads, seq_len, head_dim = query_layer.shape
        classical_scores = torch.matmul(query_layer, key_layer.transpose(-2, -1))
        
        q_quantum = torch.matmul(query_layer, self.quantum_kernel_weights)
        k_quantum = torch.matmul(key_layer, self.quantum_kernel_weights)
        
        q_expanded = q_quantum.unsqueeze(-2)
        k_expanded = k_quantum.unsqueeze(-3)
        
        quantum_similarity = torch.sum(q_expanded * k_expanded, dim=-1)
        k_rolled = torch.roll(k_expanded, shifts=1, dims=-1)
        entanglement_term = self.entanglement_strength * torch.sum(
            q_expanded * k_rolled, dim=-1
        )
        quantum_enhancement = quantum_similarity + entanglement_term
        enhanced_scores = classical_scores + 0.1 * quantum_enhancement
        
        return enhanced_scores


class QuantumSuperpositionAttention(nn.Module):
    """
    Quantum-inspired attention probability computation.
    """
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.measurement_basis = nn.Parameter(torch.randn(1) * 0.1)
        
    def quantum_softmax(self, attention_scores):
        classical_probs = nn.functional.softmax(attention_scores / self.temperature, dim=-1)
        quantum_phase = torch.sin(attention_scores + self.measurement_basis)
        quantum_interference = 0.05 * (quantum_phase - quantum_phase.mean(dim=-1, keepdim=True))
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
    """
    Quantum-inspired context aggregation.
    """
    def __init__(self, head_dim, n_qubits=4):
        super().__init__()
        self.head_dim = head_dim
        self.n_qubits = n_qubits
        self.measurement_weights = nn.Parameter(torch.randn(head_dim, n_qubits) * 0.1)
        self.quantum_aggregation = nn.Linear(n_qubits, head_dim)
        
    def quantum_measurement_simulation(self, value_states, attention_probs):
        classical_context = torch.matmul(attention_probs, value_states)
        value_quantum = torch.matmul(value_states, self.measurement_weights)
        quantum_measurement = torch.matmul(attention_probs, value_quantum)
        
        batch_size, num_heads, seq_len, n_qubits = quantum_measurement.shape
        quantum_flat = quantum_measurement.reshape(-1, n_qubits)
        quantum_enhancement_flat = self.quantum_aggregation(quantum_flat)
        quantum_enhancement = quantum_enhancement_flat.reshape(batch_size, num_heads, seq_len, self.head_dim)
        
        enhanced_context = classical_context + 0.1 * quantum_enhancement
        return enhanced_context
    
    def forward(self, attention_probs, value_layer):
        quantum_context = self.quantum_measurement_simulation(value_layer, attention_probs)
        return quantum_context


##############################################
# Quantum-Enhanced Self-Attention
##############################################

class CustomLastLayerSelfAttention(BertSelfAttention):
    """
    Quantum-enhanced self-attention for the LAST layer.
    """
    
    def __init__(self, config, n_qubits=4, enable_quantum_features=True, use_quantum_simulator=False):
        super().__init__(config)
        
        self.n_qubits = n_qubits
        self.enable_quantum_features = enable_quantum_features
        self.use_quantum_simulator = use_quantum_simulator
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        if enable_quantum_features:
            print(f"\n🔬 Initializing Quantum-Enhanced Attention:")
            print(f"   - Qubits: {n_qubits}")
            print(f"   - Quantum Simulator: {use_quantum_simulator}")
            
            # --- FIX #2: DIMENSION MISMATCH FIX ---
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

    # --- FIX #3: MANUALLY ADD HELPER METHOD (Stops AttributeError) ---
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
        **kwargs # --- FIX #4: ADD KWARGS (Stops past_key_values crash) ---
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
        
        # QUANTUM ENHANCEMENT 1: Quantum feature map encoding
        if self.enable_quantum_features:
            batch_size, num_heads, seq_len, head_dim = query_layer.shape
            
            query_flat = query_layer.reshape(-1, head_dim)
            key_flat = key_layer.reshape(-1, head_dim)
            value_flat = value_layer.reshape(-1, head_dim)
            
            query_quantum = self.quantum_query_encoder(query_flat).reshape(
                batch_size, num_heads, seq_len, head_dim
            )
            key_quantum = self.quantum_key_encoder(key_flat).reshape(
                batch_size, num_heads, seq_len, head_dim
            )
            value_quantum = self.quantum_value_encoder(value_flat).reshape(
                batch_size, num_heads, seq_len, head_dim
            )
            
            query_layer = query_layer + 0.1 * query_quantum
            key_layer = key_layer + 0.1 * key_quantum
            value_layer = value_layer + 0.1 * value_quantum

        # STEP 2: Attention Score Computation
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if self.enable_quantum_features:
            quantum_scores = self.quantum_similarity(query_layer, key_layer)
            attention_scores = attention_scores + 0.1 * quantum_scores

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # STEP 3: Attention Probabilities
        if self.enable_quantum_features:
            attention_probs = self.quantum_attention(attention_scores)
        else:
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # STEP 4: Context Aggregation
        if self.enable_quantum_features:
            context_layer = self.quantum_aggregation(attention_probs, value_layer)
        else:
            context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


# Replace only the last layer's attention
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


##############################################
# Helper Functions
##############################################

def interpret_quantum_outputs(quantum_values, verbose=True):
    if isinstance(quantum_values, list):
        quantum_values = torch.stack(quantum_values)
    
    quantum_values = quantum_values.flatten()
    mean_val = quantum_values.mean().item()
    std_val = quantum_values.std().item()
    min_val = quantum_values.min().item()
    max_val = quantum_values.max().item()
    
    abs_values = torch.abs(quantum_values)
    classical_like = (abs_values > 0.8).sum().item()
    quantum_like = (abs_values < 0.2).sum().item()
    mixed = len(quantum_values) - classical_like - quantum_like
    
    interpretation = {
        'mean': mean_val,
        'std': std_val,
        'range': (min_val, max_val),
        'classical_like_count': classical_like,
        'quantum_like_count': quantum_like,
        'mixed_count': mixed,
        'quantum_ratio': quantum_like / len(quantum_values) if len(quantum_values) > 0 else 0
    }
    
    if verbose:
        print("\n" + "="*60)
        print("QUANTUM OUTPUT INTERPRETATION")
        print("="*60)
        print(f"Mean value: {mean_val:.4f} (0 = strong quantum, ±1 = classical)")
        print(f"Std deviation: {std_val:.4f} (higher = more variation)")
        print(f"Range: [{min_val:.4f}, {max_val:.4f}]")
        print(f"\nQuantum behavior breakdown:")
        print(f"  Classical-like (|value| > 0.8): {classical_like}/{len(quantum_values)}")
        print(f"  Quantum-like (|value| < 0.2):   {quantum_like}/{len(quantum_values)}")
        print(f"  Mixed behavior:                 {mixed}/{len(quantum_values)}")
        print(f"\nQuantum ratio: {interpretation['quantum_ratio']:.2%}")
        print("="*60 + "\n")
    
    return interpretation


def compare_quantum_vs_classical(model, input_ids, attention_mask=None):
    model.eval()
    with torch.no_grad():
        outputs_quantum = model(input_ids=input_ids, attention_mask=attention_mask)
        original_state = model.bert.encoder.layer[-1].attention.self.enable_quantum_features
        model.bert.encoder.layer[-1].attention.self.enable_quantum_features = False
        outputs_classical = model(input_ids=input_ids, attention_mask=attention_mask)
        model.bert.encoder.layer[-1].attention.self.enable_quantum_features = original_state
    
    logits_quantum = outputs_quantum.logits
    logits_classical = outputs_classical.logits
    
    diff = (logits_quantum - logits_classical).abs().mean().item()
    max_diff = (logits_quantum - logits_classical).abs().max().item()
    
    print("\n" + "="*60)
    print("QUANTUM vs CLASSICAL COMPARISON")
    print("="*60)
    print(f"Mean absolute difference in logits: {diff:.6f}")
    print(f"Max absolute difference: {max_diff:.6f}")
    print("="*60 + "\n")
    
    return {
        'mean_diff': diff,
        'max_diff': max_diff,
        'quantum_logits': logits_quantum,
        'classical_logits': logits_classical
    }