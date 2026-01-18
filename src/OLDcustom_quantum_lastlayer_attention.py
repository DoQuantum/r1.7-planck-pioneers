import math
import torch
from torch import nn
from transformers import BertForMaskedLM
from transformers.models.bert.modeling_bert import (
    BertSelfAttention,
    BertSelfOutput,
)
import pennylane as qml
import numpy as np

##############################################
# 1. Quantum Circuit for Attention
##############################################
class QuantumAttentionCircuit(nn.Module):
    """
    Quantum circuit that processes attention scores using PennyLane simulator.
    """
    def __init__(self, n_qubits=8, n_layers=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Create quantum device (simulator)
        self.dev = qml.device('default.qubit', wires=n_qubits)
        
        # Learnable quantum parameters
        # Each layer needs rotation parameters for each qubit
        weight_shape = (n_layers, n_qubits, 3)  # 3 rotations (RX, RY, RZ) per qubit
        self.weights = nn.Parameter(torch.randn(weight_shape) * 0.01)
        
        # Define the quantum circuit
        @qml.qnode(self.dev, interface='torch', diff_method='backprop')
        def quantum_circuit(inputs, weights):
            """
            Variational quantum circuit for processing attention features.
            
            Args:
                inputs: Classical data to encode (n_qubits values)
                weights: Learnable quantum gate parameters
            """
            # 1. Amplitude encoding of input data
            for i in range(self.n_qubits):
                qml.RY(inputs[i], wires=i)
            
            # 2. Variational layers with entanglement
            for layer in range(self.n_layers):
                # Rotation gates (parameterized)
                for i in range(self.n_qubits):
                    qml.RX(weights[layer, i, 0], wires=i)
                    qml.RY(weights[layer, i, 1], wires=i)
                    qml.RZ(weights[layer, i, 2], wires=i)
                
                # Entanglement layer (creates quantum correlations)
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                
                # Ring connection for full connectivity
                if self.n_qubits > 2:
                    qml.CNOT(wires=[self.n_qubits - 1, 0])
            
            # 3. Measurement: expectation values of Pauli-Z
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        self.quantum_circuit = quantum_circuit
        
        # Classical post-processing layer
        self.post_process = nn.Linear(n_qubits, 1)
    
    def forward(self, attention_scores):
        """
        Process attention scores through quantum circuit.
        
        Args:
            attention_scores: [batch, heads, seq_len, seq_len]
        
        Returns:
            Modified attention scores with same shape
        """
        batch_size, num_heads, seq_len, _ = attention_scores.shape
        
        # Flatten for processing
        scores_flat = attention_scores.reshape(-1, seq_len)
        
        quantum_outputs = []
        
        # Process each attention score row through quantum circuit
        for i in range(scores_flat.shape[0]):
            # Take first n_qubits positions (or pad/slice as needed)
            if seq_len >= self.n_qubits:
                inputs = scores_flat[i, :self.n_qubits]
            else:
                # Pad if sequence is shorter than n_qubits
                inputs = torch.cat([
                    scores_flat[i],
                    torch.zeros(self.n_qubits - seq_len, device=scores_flat.device)
                ])
            
            # Normalize inputs to [-π, π] range for quantum encoding
            inputs_normalized = torch.tanh(inputs) * math.pi
            
            # Run quantum circuit
            quantum_output = self.quantum_circuit(inputs_normalized, self.weights)
            quantum_output = torch.stack(quantum_output)
            
            # Post-process to get scalar modulation
            modulation = self.post_process(quantum_output.unsqueeze(0))
            quantum_outputs.append(modulation)
        
        # Stack and reshape
        quantum_modulation = torch.cat(quantum_outputs, dim=0)
        quantum_modulation = quantum_modulation.reshape(batch_size, num_heads, seq_len, 1)
        
        # Apply quantum modulation to attention scores
        # This adds quantum-processed information to classical attention
        modified_scores = attention_scores + 0.1 * quantum_modulation
        
        return modified_scores


##############################################
# 2. Quantum-Enhanced Self-Attention
##############################################
class QuantumEnhancedSelfAttention(BertSelfAttention):
    """
    Self-attention with quantum circuit processing in the last layer.
    """
    def __init__(self, config, n_qubits=8, n_layers=2):
        super().__init__(config)
        
        # Initialize quantum circuit
        self.quantum_circuit = QuantumAttentionCircuit(
            n_qubits=n_qubits,
            n_layers=n_layers
        )
        
        print(f"✓ Initialized Quantum Circuit with {n_qubits} qubits and {n_layers} layers")
    
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
        # Standard BERT attention computations
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
        
        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # ---- QUANTUM ENHANCEMENT ----
        # Process attention scores through quantum circuit
        attention_scores = self.quantum_circuit(attention_scores)
        # ----------------------------
        
        # Apply mask
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Softmax
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        
        # Mask heads if needed
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        
        # Weighted sum
        context_layer = torch.matmul(attention_probs, value_layer)
        
        # Reshape
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


##############################################
# 3. Custom BERT with Quantum Last Layer
##############################################
class QuantumBertForMaskedLM(BertForMaskedLM):
    """
    BERT model with quantum-enhanced attention in the last layer.
    """
    def __init__(self, config, n_qubits=8, n_quantum_layers=2):
        super().__init__(config)
        
        LAST_LAYER = config.num_hidden_layers - 1
        
        # Create quantum-enhanced attention
        quantum_attention = QuantumEnhancedSelfAttention(
            config,
            n_qubits=n_qubits,
            n_layers=n_quantum_layers
        )
        quantum_attention_output = BertSelfOutput(config)
        
        # Get the last encoder layer
        layer = self.bert.encoder.layer[LAST_LAYER]
        
        # Replace ONLY the last layer's self-attention
        layer.attention.self = quantum_attention
        layer.attention.output = quantum_attention_output
        
        print(f"✓ Replaced layer {LAST_LAYER} with quantum-enhanced attention")


##############################################
# 4. Example Usage
##############################################
if __name__ == "__main__":
    from transformers import BertConfig, BertTokenizer
    
    print("=" * 60)
    print("Quantum-Enhanced BERT Demo")
    print("=" * 60)
    
    # Create a small BERT config for testing
    config = BertConfig(
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=128,
    )
    
    # Initialize quantum BERT
    print("\n1. Initializing Quantum BERT...")
    model = QuantumBertForMaskedLM(
        config,
        n_qubits=8,  # Number of qubits in quantum circuit
        n_quantum_layers=2  # Depth of quantum circuit
    )
    
    # Create dummy input
    print("\n2. Creating dummy input...")
    batch_size = 2
    seq_length = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    
    # Forward pass
    print("\n3. Running forward pass...")
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    print(f"\n✓ Output shape: {outputs.logits.shape}")
    print(f"✓ Successfully processed input through quantum-enhanced BERT!")
    
    # Show trainable parameters
    print("\n4. Model parameters:")
    total_params = sum(p.numel() for p in model.parameters())
    quantum_params = sum(p.numel() for n, p in model.named_parameters() if 'quantum' in n)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Quantum parameters: {quantum_params:,}")
    
    print("\n" + "=" * 60)