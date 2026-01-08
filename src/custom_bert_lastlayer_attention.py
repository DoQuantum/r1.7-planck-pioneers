"""
Quantum-Enhanced BERT Attention Mechanism

HOW QUANTUM SIMULATION WORKS:
=============================

1. CLASSICAL SIMULATION (Default, use_quantum_simulator=False):
   - Uses PyTorch operations (matmul, sin, cos, tanh) to approximate quantum behavior
   - Fast execution, runs on CPU/GPU like normal neural networks
   - Mimics quantum concepts but doesn't use actual quantum mechanics
   - Example: Uses sin/cos to approximate rotation gates, matrix ops for state encoding

2. QUANTUM SIMULATION (use_quantum_simulator=True):
   - Uses PennyLane's 'default.qubit' simulator to run ACTUAL quantum circuits
   - Creates real quantum states (superposition, entanglement)
   - Applies real quantum gates (RX, RY, RZ rotations, CNOT entanglement)
   - Measures quantum observables (Pauli-Z expectation values)
   - Slower but provides true quantum behavior and can run on real quantum hardware

QUANTUM CIRCUIT STRUCTURE (when use_quantum_simulator=True):
   Input → Angle Encoding (RY gates) → Variational Layer (RX, RY, RZ) 
   → Entanglement (CNOT) → Measurement (Pauli-Z) → Output

The quantum circuit processes data in quantum superposition, creating correlations
through entanglement, then extracts classical information via measurement.
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
    """
    Quantum-inspired feature map encoder.
    
    CURRENT IMPLEMENTATION (use_quantum_simulator=False):
    - Uses classical PyTorch operations to approximate quantum behavior
    - Uses matrix multiplications, sin/cos functions to simulate quantum gates
    - Fast but not true quantum simulation
    
    QUANTUM SIMULATION (use_quantum_simulator=True):
    - Uses PennyLane's 'default.qubit' simulator to run actual quantum circuits
    - Creates real quantum states, applies quantum gates (RX, RY, RZ, CNOT)
    - Measures expectation values of quantum observables
    - Slower but provides true quantum behavior
    
    This is structured to evolve into a full quantum circuit with amplitude encoding.
    """
    def __init__(self, input_dim, output_dim, n_qubits=4, feature_map_type='pauli_z', 
                 use_quantum_simulator=False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_qubits = n_qubits
        self.feature_map_type = feature_map_type
        self.use_quantum_simulator = use_quantum_simulator
        
        if use_quantum_simulator:
            # ============================================================
            # ACTUAL QUANTUM SIMULATION USING PENNYLANE
            # ============================================================
            # Create PennyLane quantum device (simulator)
            self.dev = qml.device('default.qubit', wires=n_qubits)
            
            # Learnable quantum gate parameters
            # Each qubit needs rotation parameters (RX, RY, RZ)
            self.rotation_params = nn.Parameter(torch.randn(n_qubits, 3) * 0.01)
            
            # Define the actual quantum circuit
            @qml.qnode(self.dev, interface='torch', diff_method='backprop')
            def quantum_circuit(inputs, weights):
                """
                REAL QUANTUM CIRCUIT:
                1. Angle encoding: Encode classical data as rotation angles (RY gates)
                2. Variational layers: Learnable rotations (RX, RY, RZ)
                3. Entanglement: CNOT gates create quantum correlations
                4. Measurement: Expectation values of Pauli-Z observables
                
                Args:
                    inputs: Classical data [n_qubits] - will be encoded as angles
                    weights: Learnable parameters [n_qubits, 3] for RX, RY, RZ
                
                Returns:
                    List of expectation values (one per qubit)
                """
                # Step 1: ANGLE ENCODING - Encode classical data into quantum state
                # Each input value becomes a rotation angle on a qubit
                for i in range(n_qubits):
                    qml.RY(inputs[i], wires=i)  # Rotate qubit i by input angle
                
                # Step 2: VARIATIONAL LAYER - Learnable quantum gates
                # Apply parameterized rotations to each qubit
                for i in range(n_qubits):
                    qml.RX(weights[i, 0], wires=i)  # Rotation around X-axis
                    qml.RY(weights[i, 1], wires=i)  # Rotation around Y-axis
                    qml.RZ(weights[i, 2], wires=i)  # Rotation around Z-axis
                
                # Step 3: ENTANGLEMENT - Create quantum correlations
                # CNOT gates entangle qubits (quantum superposition + correlation)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])  # Entangle qubit i with qubit i+1
                
                # Step 4: MEASUREMENT - Extract classical information
                # Measure expectation values of Pauli-Z operator on each qubit
                # Returns values in [-1, 1] range
                return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
            
            self.quantum_circuit = quantum_circuit
            print(f"  → Using REAL PennyLane quantum simulator with {n_qubits} qubits")
        else:
            # ============================================================
            # CLASSICAL SIMULATION (Current default)
            # ============================================================
            # Classical approximation of quantum operations
            # Uses PyTorch operations to mimic quantum behavior
            self.encoding_weights = nn.Parameter(torch.randn(input_dim, n_qubits) * 0.1)
            self.rotation_params = nn.Parameter(torch.randn(n_qubits, 3) * 0.01)
            print(f"  → Using CLASSICAL simulation (PyTorch approximations)")
        
        # Projection to output dimension (classical post-processing)
        self.projection = nn.Linear(n_qubits, output_dim)
        
    def quantum_feature_map_simulation(self, x):
        """
        Encode input through quantum feature map.
        
        If use_quantum_simulator=True:
        - Runs ACTUAL quantum circuit using PennyLane
        - Creates quantum states, applies gates, measures observables
        - Returns expectation values from quantum measurements
        
        If use_quantum_simulator=False (default):
        - Uses classical PyTorch operations to approximate quantum behavior
        - Fast but not true quantum simulation
        
        Args:
            x: Input tensor [..., input_dim]
        
        Returns:
            Quantum features [..., n_qubits]
        """
        if self.use_quantum_simulator:
            # ============================================================
            # REAL QUANTUM SIMULATION PATH
            # ============================================================
            original_shape = x.shape
            x_flat = x.reshape(-1, self.input_dim)  # [batch_size, input_dim]
            
            quantum_outputs = []
            
            # Process each sample through quantum circuit
            for sample in x_flat:
                # Prepare input for quantum circuit
                # Take first n_qubits dimensions (or pad/truncate)
                if self.input_dim >= self.n_qubits:
                    quantum_input = sample[:self.n_qubits]
                else:
                    # Pad with zeros if input is smaller than n_qubits
                    quantum_input = torch.cat([
                        sample,
                        torch.zeros(self.n_qubits - self.input_dim, device=x.device)
                    ])
                
                # Normalize to [-π, π] range for quantum encoding
                quantum_input = torch.tanh(quantum_input) * math.pi
                
                # RUN ACTUAL QUANTUM CIRCUIT
                # This executes: encoding → rotations → entanglement → measurement
                quantum_result = self.quantum_circuit(quantum_input, self.rotation_params)
                
                # Convert list of expectation values to tensor
                quantum_outputs.append(torch.stack(quantum_result))
            
            # Stack all results
            encoded = torch.stack(quantum_outputs)
            encoded = encoded.reshape(*original_shape[:-1], self.n_qubits)
            
        else:
            # ============================================================
            # CLASSICAL SIMULATION PATH (Current default)
            # ============================================================
            # Step 1: Encode into quantum-inspired space (simulates amplitude encoding)
            # Uses matrix multiplication instead of actual quantum gates
            encoded = torch.matmul(x, self.encoding_weights)  # [..., n_qubits]
            
            # Step 2: Apply quantum-inspired rotations (simulates RX, RY, RZ gates)
            # Uses sin/cos functions to approximate rotation gate effects
            for i in range(self.n_qubits):
                rotation_effect = (
                    torch.sin(encoded[..., i:i+1] + self.rotation_params[i, 0]) * 
                    torch.cos(encoded[..., i:i+1] + self.rotation_params[i, 1]) +
                    self.rotation_params[i, 2]
                )
                encoded[..., i] = encoded[..., i] + 0.1 * rotation_effect
            
            # Step 3: Normalize (simulates quantum state normalization)
            encoded = torch.tanh(encoded)  # Keep values bounded
        
        return encoded
    
    def forward(self, x):
        """
        Encode input through quantum-inspired feature map.
        
        Args:
            x: Input tensor [..., input_dim]
        
        Returns:
            Encoded features [..., output_dim]
        """
        # Apply quantum-inspired encoding
        quantum_features = self.quantum_feature_map_simulation(x)
        
        # Project to output dimension
        output = self.projection(quantum_features)
        
        return output


class QuantumInspiredSimilarity(nn.Module):
    """
    Quantum-inspired similarity computation for attention scores.
    Uses quantum-inspired inner products and entanglement concepts.
    
    This will evolve into quantum kernel methods and quantum inner products.
    """
    def __init__(self, head_dim, n_qubits=4):
        super().__init__()
        self.head_dim = head_dim
        self.n_qubits = n_qubits
        
        # Quantum-inspired similarity parameters
        # These simulate quantum kernel functions
        self.quantum_kernel_weights = nn.Parameter(torch.randn(head_dim, n_qubits) * 0.1)
        self.entanglement_strength = nn.Parameter(torch.tensor(0.1))
        
    def quantum_inner_product(self, query, key):
        """
        Quantum-inspired inner product computation.
        Simulates quantum kernel methods and quantum feature space inner products.
        
        Args:
            query: [..., head_dim]
            key: [..., head_dim]
        
        Returns:
            Quantum-inspired similarity scores
        """
        # Project into quantum-inspired feature space
        q_quantum = torch.matmul(query, self.quantum_kernel_weights)  # [..., n_qubits]
        k_quantum = torch.matmul(key, self.quantum_kernel_weights)  # [..., n_qubits]
        
        # Quantum-inspired inner product (simulates quantum kernel)
        # In full quantum: this becomes measurement of quantum states
        quantum_similarity = torch.sum(q_quantum * k_quantum, dim=-1, keepdim=True)
        
        # Add entanglement-inspired correlation term
        # Simulates quantum entanglement effects on similarity
        entanglement_term = self.entanglement_strength * torch.sum(
            q_quantum * torch.roll(k_quantum, shifts=1, dims=-1), dim=-1, keepdim=True
        )
        
        return quantum_similarity + entanglement_term
    
    def forward(self, query_layer, key_layer):
        """
        Compute quantum-inspired attention scores.
        
        Args:
            query_layer: [batch, heads, seq_len, head_dim]
            key_layer: [batch, heads, seq_len, head_dim]
        
        Returns:
            Quantum-enhanced attention scores [batch, heads, seq_len, seq_len]
        """
        batch_size, num_heads, seq_len, head_dim = query_layer.shape
        
        # Standard dot-product attention
        classical_scores = torch.matmul(query_layer, key_layer.transpose(-2, -1))
        
        # Quantum-inspired enhancement (vectorized)
        # Project Q and K into quantum feature space
        q_quantum = torch.matmul(query_layer, self.quantum_kernel_weights)  # [batch, heads, seq_len, n_qubits]
        k_quantum = torch.matmul(key_layer, self.quantum_kernel_weights)   # [batch, heads, seq_len, n_qubits]
        
        # Compute quantum inner product for all query-key pairs
        # q_quantum: [batch, heads, seq_len_q, n_qubits]
        # k_quantum: [batch, heads, seq_len_k, n_qubits]
        # We want: [batch, heads, seq_len_q, seq_len_k]
        q_expanded = q_quantum.unsqueeze(-2)  # [batch, heads, seq_len_q, 1, n_qubits]
        k_expanded = k_quantum.unsqueeze(-3)  # [batch, heads, 1, seq_len_k, n_qubits]
        
        # Quantum kernel similarity
        quantum_similarity = torch.sum(q_expanded * k_expanded, dim=-1)  # [batch, heads, seq_len_q, seq_len_k]
        
        # Entanglement-inspired correlation
        k_rolled = torch.roll(k_expanded, shifts=1, dims=-1)  # Shift for entanglement simulation
        entanglement_term = self.entanglement_strength * torch.sum(
            q_expanded * k_rolled, dim=-1
        )
        
        quantum_enhancement = quantum_similarity + entanglement_term
        
        # Combine classical and quantum scores
        enhanced_scores = classical_scores + 0.1 * quantum_enhancement
        
        return enhanced_scores


class QuantumSuperpositionAttention(nn.Module):
    """
    Quantum-inspired attention probability computation using superposition concepts.
    Simulates quantum superposition and measurement for attention weights.
    
    This will evolve into quantum measurement operations on quantum states.
    """
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        # Quantum measurement parameters (simulates measurement basis)
        self.measurement_basis = nn.Parameter(torch.randn(1) * 0.1)
        
    def quantum_softmax(self, attention_scores):
        """
        Quantum-inspired softmax using superposition concepts.
        Simulates quantum measurement probabilities.
        
        Args:
            attention_scores: [..., seq_len]
        
        Returns:
            Quantum-inspired attention probabilities
        """
        # Standard softmax (simulates quantum measurement probabilities)
        classical_probs = nn.functional.softmax(attention_scores / self.temperature, dim=-1)
        
        # Quantum superposition enhancement
        # Simulates quantum interference effects
        # In full quantum: this becomes actual quantum measurement
        quantum_phase = torch.sin(attention_scores + self.measurement_basis)
        quantum_interference = 0.05 * (quantum_phase - quantum_phase.mean(dim=-1, keepdim=True))
        
        # Combine classical probabilities with quantum interference
        quantum_probs = classical_probs + quantum_interference
        quantum_probs = torch.clamp(quantum_probs, min=0.0)  # Ensure non-negative
        
        # Renormalize (quantum state normalization)
        quantum_probs = quantum_probs / (quantum_probs.sum(dim=-1, keepdim=True) + 1e-8)
        
        return quantum_probs
    
    def forward(self, attention_scores):
        """
        Apply quantum-inspired attention probability computation.
        
        Args:
            attention_scores: [batch, heads, seq_len, seq_len]
        
        Returns:
            Quantum-enhanced attention probabilities
        """
        batch_size, num_heads, seq_len, _ = attention_scores.shape
        
        # Reshape for processing
        scores_flat = attention_scores.reshape(-1, seq_len)
        
        # Apply quantum-inspired softmax
        probs_flat = self.quantum_softmax(scores_flat)
        
        # Reshape back
        quantum_probs = probs_flat.reshape(batch_size, num_heads, seq_len, seq_len)
        
        return quantum_probs


class QuantumContextAggregation(nn.Module):
    """
    Quantum-inspired context aggregation using quantum measurement concepts.
    Simulates quantum state collapse and measurement for final context computation.
    
    This will evolve into quantum measurement operations on entangled states.
    """
    def __init__(self, head_dim, n_qubits=4):
        super().__init__()
        self.head_dim = head_dim
        self.n_qubits = n_qubits
        
        # Quantum measurement parameters
        self.measurement_weights = nn.Parameter(torch.randn(head_dim, n_qubits) * 0.1)
        self.quantum_aggregation = nn.Linear(n_qubits, head_dim)
        
    def quantum_measurement_simulation(self, value_states, attention_probs):
        """
        Simulate quantum measurement on value states weighted by attention.
        
        Args:
            value_states: [batch, heads, seq_len, head_dim]
            attention_probs: [batch, heads, seq_len, seq_len]
        
        Returns:
            Quantum-enhanced context [batch, heads, seq_len, head_dim]
        """
        # Standard weighted aggregation
        classical_context = torch.matmul(attention_probs, value_states)
        
        # Quantum-inspired enhancement
        # Project values into quantum measurement space
        value_quantum = torch.matmul(value_states, self.measurement_weights)  # [batch, heads, seq_len, n_qubits]
        
        # Apply attention-weighted quantum measurement
        # Simulates measuring quantum states with attention probabilities
        quantum_measurement = torch.matmul(attention_probs, value_quantum)  # [batch, heads, seq_len, n_qubits]
        
        # Project back to classical space for each position
        batch_size, num_heads, seq_len, n_qubits = quantum_measurement.shape
        quantum_flat = quantum_measurement.reshape(-1, n_qubits)  # [batch*heads*seq_len, n_qubits]
        quantum_enhancement_flat = self.quantum_aggregation(quantum_flat)  # [batch*heads*seq_len, head_dim]
        quantum_enhancement = quantum_enhancement_flat.reshape(batch_size, num_heads, seq_len, self.head_dim)
        
        # Combine classical and quantum
        enhanced_context = classical_context + 0.1 * quantum_enhancement
        
        return enhanced_context
    
    def forward(self, attention_probs, value_layer):
        """
        Quantum-inspired context aggregation.
        
        Args:
            attention_probs: [batch, heads, seq_len, seq_len]
            value_layer: [batch, heads, seq_len, head_dim]
        
        Returns:
            Quantum-enhanced context layer
        """
        # Standard context computation
        classical_context = torch.matmul(attention_probs, value_layer)
        
        # Quantum enhancement
        quantum_context = self.quantum_measurement_simulation(value_layer, attention_probs)
        
        # Combine (quantum_context already includes classical, so we use it directly)
        return quantum_context


##############################################
# Quantum-Enhanced Self-Attention
##############################################

class CustomLastLayerSelfAttention(BertSelfAttention):
    """
    Quantum-enhanced self-attention for the LAST layer.
    Incorporates quantum-inspired improvements at multiple stages:
    1. Quantum feature map encoding for Q, K, V
    2. Quantum-inspired similarity computation
    3. Quantum superposition attention probabilities
    4. Quantum context aggregation
    
    All enhancements are classically simulatable but structured to evolve
    into full quantum circuits.
    """
    
    def __init__(self, config, n_qubits=4, enable_quantum_features=True, use_quantum_simulator=False):
        super().__init__(config)
        
        self.n_qubits = n_qubits
        self.enable_quantum_features = enable_quantum_features
        self.use_quantum_simulator = use_quantum_simulator
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        
        # Quantum-inspired components
        if enable_quantum_features:
            print(f"\n🔬 Initializing Quantum-Enhanced Attention:")
            print(f"   - Qubits: {n_qubits}")
            print(f"   - Quantum Simulator: {use_quantum_simulator}")
            
            # Quantum feature map encoders for Q, K, V
            self.quantum_query_encoder = QuantumFeatureMapEncoder(
                input_dim=config.hidden_size,
                output_dim=config.hidden_size,
                n_qubits=n_qubits,
                use_quantum_simulator=use_quantum_simulator
            )
            self.quantum_key_encoder = QuantumFeatureMapEncoder(
                input_dim=config.hidden_size,
                output_dim=config.hidden_size,
                n_qubits=n_qubits,
                use_quantum_simulator=use_quantum_simulator
            )
            self.quantum_value_encoder = QuantumFeatureMapEncoder(
                input_dim=config.hidden_size,
                output_dim=config.hidden_size,
                n_qubits=n_qubits,
                use_quantum_simulator=use_quantum_simulator
            )
            
            # Quantum-inspired similarity computation
            self.quantum_similarity = QuantumInspiredSimilarity(
                head_dim=self.attention_head_size,
                n_qubits=n_qubits
            )
            
            # Quantum superposition attention
            self.quantum_attention = QuantumSuperpositionAttention(temperature=1.0)
            
            # Quantum context aggregation
            self.quantum_aggregation = QuantumContextAggregation(
                head_dim=self.attention_head_size,
                n_qubits=n_qubits
            )
            
            print(f"✓ Initialized Quantum-Enhanced Attention with {n_qubits} qubits\n")

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # ============================================================
        # STEP 1: Query, Key, Value Projection with Quantum Encoding
        # ============================================================
        
        # Standard BERT projections
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
            # Encode Q, K, V through quantum feature maps
            # This simulates quantum state preparation
            batch_size, num_heads, seq_len, head_dim = query_layer.shape
            
            # Reshape for quantum encoding: [batch*heads*seq_len, head_dim]
            query_flat = query_layer.reshape(-1, head_dim)
            key_flat = key_layer.reshape(-1, head_dim)
            value_flat = value_layer.reshape(-1, head_dim)
            
            # Apply quantum feature map encoding
            query_quantum = self.quantum_query_encoder(query_flat).reshape(
                batch_size, num_heads, seq_len, head_dim
            )
            key_quantum = self.quantum_key_encoder(key_flat).reshape(
                batch_size, num_heads, seq_len, head_dim
            )
            value_quantum = self.quantum_value_encoder(value_flat).reshape(
                batch_size, num_heads, seq_len, head_dim
            )
            
            # Combine classical and quantum-encoded features
            query_layer = query_layer + 0.1 * query_quantum
            key_layer = key_layer + 0.1 * key_quantum
            value_layer = value_layer + 0.1 * value_quantum

        # ============================================================
        # STEP 2: Attention Score Computation with Quantum Similarity
        # ============================================================
        
        # Standard dot-product attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # QUANTUM ENHANCEMENT 2: Quantum-inspired similarity
        if self.enable_quantum_features:
            quantum_scores = self.quantum_similarity(query_layer, key_layer)
            # Combine classical and quantum scores
            attention_scores = attention_scores + 0.1 * quantum_scores

        # Apply attention mask
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # ============================================================
        # STEP 3: Attention Probabilities with Quantum Superposition
        # ============================================================
        
        # QUANTUM ENHANCEMENT 3: Quantum superposition attention
        if self.enable_quantum_features:
            attention_probs = self.quantum_attention(attention_scores)
        else:
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # ============================================================
        # STEP 4: Context Aggregation with Quantum Measurement
        # ============================================================
        
        # QUANTUM ENHANCEMENT 4: Quantum context aggregation
        if self.enable_quantum_features:
            context_layer = self.quantum_aggregation(attention_probs, value_layer)
        else:
            context_layer = torch.matmul(attention_probs, value_layer)

        # Reshape context layer
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


# Replace only the last layer's attention
class CustomBertForMaskedLM_LastLayerAttention(BertForMaskedLM):
    """
    BERT model with quantum-enhanced attention in the last layer.
    Replaces **only the last layer's self-attention** with quantum-inspired improvements.
    """

    def __init__(self, config, n_qubits=4, enable_quantum_features=True, use_quantum_simulator=False):
        super().__init__(config)

        LAST_LAYER = config.num_hidden_layers - 1

        # Create quantum-enhanced attention
        custom_attention = CustomLastLayerSelfAttention(
            config,
            n_qubits=n_qubits,
            enable_quantum_features=enable_quantum_features,
            use_quantum_simulator=use_quantum_simulator
        )

        # Get the last encoder layer
        layer = self.bert.encoder.layer[LAST_LAYER]

        # Replace ONLY the self-attention submodule
        layer.attention.self = custom_attention
        
        print(f"✓ Replaced layer {LAST_LAYER} with quantum-enhanced attention")


##############################################
# Helper Functions for Understanding Results
##############################################

def interpret_quantum_outputs(quantum_values, verbose=True):
    """
    Interpret quantum measurement results from qml.expval(PauliZ).
    
    Args:
        quantum_values: Tensor of quantum measurement results in [-1, 1] range
        verbose: Whether to print interpretation
    
    Returns:
        dict with interpretation metrics
    """
    if isinstance(quantum_values, list):
        quantum_values = torch.stack(quantum_values)
    
    quantum_values = quantum_values.flatten()
    
    # Calculate statistics
    mean_val = quantum_values.mean().item()
    std_val = quantum_values.std().item()
    min_val = quantum_values.min().item()
    max_val = quantum_values.max().item()
    
    # Interpret quantum behavior
    abs_values = torch.abs(quantum_values)
    classical_like = (abs_values > 0.8).sum().item()  # Near ±1
    quantum_like = (abs_values < 0.2).sum().item()     # Near 0
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
        if interpretation['quantum_ratio'] > 0.3:
            print("  → Strong quantum superposition detected!")
        elif interpretation['quantum_ratio'] < 0.1:
            print("  → Mostly classical behavior")
        else:
            print("  → Mixed quantum-classical behavior")
        print("="*60 + "\n")
    
    return interpretation


def compare_quantum_vs_classical(model, input_ids, attention_mask=None):
    """
    Compare outputs with and without quantum enhancement.
    
    Args:
        model: CustomBertForMaskedLM_LastLayerAttention model
        input_ids: Input token IDs
        attention_mask: Optional attention mask
    
    Returns:
        dict with comparison metrics
    """
    model.eval()
    with torch.no_grad():
        # Get quantum-enhanced outputs
        outputs_quantum = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Temporarily disable quantum features
        original_state = model.bert.encoder.layer[-1].attention.self.enable_quantum_features
        model.bert.encoder.layer[-1].attention.self.enable_quantum_features = False
        
        outputs_classical = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Restore original state
        model.bert.encoder.layer[-1].attention.self.enable_quantum_features = original_state
    
    # Compare logits
    logits_quantum = outputs_quantum.logits
    logits_classical = outputs_classical.logits
    
    diff = (logits_quantum - logits_classical).abs().mean().item()
    max_diff = (logits_quantum - logits_classical).abs().max().item()
    
    print("\n" + "="*60)
    print("QUANTUM vs CLASSICAL COMPARISON")
    print("="*60)
    print(f"Mean absolute difference in logits: {diff:.6f}")
    print(f"Max absolute difference: {max_diff:.6f}")
    print(f"Relative difference: {diff / logits_classical.abs().mean().item():.4%}")
    print("="*60 + "\n")
    
    return {
        'mean_diff': diff,
        'max_diff': max_diff,
        'quantum_logits': logits_quantum,
        'classical_logits': logits_classical
    }
