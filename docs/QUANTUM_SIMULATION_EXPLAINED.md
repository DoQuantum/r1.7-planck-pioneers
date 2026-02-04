# Quantum Simulation: What "Running" Means and How to Interpret Results

## What Does "Running" the Quantum Simulation Entail?

When you call `model.forward()` or `model(input_ids)`, here's what happens step-by-step:

### 1. **Input Preparation** (Classical → Quantum Interface)

```python
# Your input: hidden states from BERT [batch, seq_len, hidden_size]
hidden_states = model.bert.encoder(...)  # Standard BERT processing

# For each token position, we extract features:
query_features = hidden_states  # [batch, seq_len, 768] for BERT-base
```

**What happens:**
- Classical neural network outputs (floating point numbers)
- These get normalized to `[-π, π]` range for quantum encoding
- Each feature dimension becomes a rotation angle for a qubit

### 2. **Quantum Circuit Execution** (The "Running" Part)

For each sample in your batch, the quantum circuit runs:

#### Step A: **Quantum State Preparation (Encoding)**
```python
# Classical input: [0.5, -0.3, 0.8, 0.2] (4 features)
# ↓ Converted to angles: [1.57, -0.94, 2.51, 0.63] radians
# ↓ Applied to quantum circuit:

for i in range(4):
    qml.RY(angles[i], wires=i)  # Rotate qubit i by angle[i]
```

**What this means:**
- Each qubit starts in state |0⟩
- RY gate rotates it to: `cos(angle/2)|0⟩ + sin(angle/2)|1⟩`
- Creates a **quantum superposition** - the qubit is in both |0⟩ and |1⟩ simultaneously
- The probability of measuring |0⟩ vs |1⟩ depends on the angle

**Visual representation:**
```
Qubit 0: |0⟩ ──[RY(1.57)]──> 0.707|0⟩ + 0.707|1⟩  (50/50 superposition)
Qubit 1: |0⟩ ──[RY(-0.94)]──> 0.866|0⟩ + 0.5|1⟩   (75/25 superposition)
Qubit 2: |0⟩ ──[RY(2.51)]──> 0.5|0⟩ + 0.866|1⟩    (25/75 superposition)
Qubit 3: |0⟩ ──[RY(0.63)]──> 0.953|0⟩ + 0.303|1⟩  (91/9 superposition)
```

#### Step B: **Variational Quantum Layer (Learning)**
```python
# Learnable parameters (trained during backpropagation)
weights = [[0.1, 0.2, -0.1],   # Qubit 0: RX, RY, RZ rotations
           [0.05, -0.1, 0.15],  # Qubit 1
           [-0.2, 0.1, 0.05],   # Qubit 2
           [0.15, -0.05, 0.1]]  # Qubit 3

for i in range(4):
    qml.RX(weights[i, 0], wires=i)  # Additional rotation around X-axis
    qml.RY(weights[i, 1], wires=i)  # Additional rotation around Y-axis
    qml.RZ(weights[i, 2], wires=i)  # Additional rotation around Z-axis
```

**What this means:**
- These rotations transform the quantum state
- The parameters are **learnable** - they get updated during training
- This is where the model learns quantum patterns in your data

#### Step C: **Entanglement (Quantum Correlations)**
```python
# CNOT gates create quantum entanglement
qml.CNOT(wires=[0, 1])  # Entangle qubit 0 and 1
qml.CNOT(wires=[1, 2])  # Entangle qubit 1 and 2
qml.CNOT(wires=[2, 3])  # Entangle qubit 2 and 3
qml.CNOT(wires=[3, 0])  # Ring connection
```

**What this means:**
- Before: Each qubit is independent
- After: Qubits become **entangled** - measuring one affects the others
- Creates quantum correlations that classical computers can't efficiently simulate
- This is where quantum advantage potentially comes from

**Example:**
```
Before: Qubit 0 and 1 are independent
After:  If you measure Qubit 0 = |1⟩, then Qubit 1 MUST be |1⟩ (correlated)
```

#### Step D: **Measurement (Quantum → Classical)**
```python
# Measure expectation values of Pauli-Z operator
results = [qml.expval(qml.PauliZ(i)) for i in range(4)]
# Returns: [-0.707, 0.5, -0.866, 0.906]  (values in [-1, 1] range)
```

**What this means:**
- `expval(PauliZ)` measures the "spin" of the qubit along the Z-axis
- Returns a value between -1 and +1:
  - **+1**: Qubit is definitely in state |0⟩
  - **-1**: Qubit is definitely in state |1⟩
  - **0**: Qubit is in perfect superposition (50/50)
  - **Other values**: Partial superposition

**Interpretation:**
- These are **expectation values**, not single measurements
- They represent the average outcome if you measured many times
- They capture quantum information in a classical format

### 3. **Integration Back into Classical Model**

```python
# Quantum outputs: [batch, n_qubits] with values in [-1, 1]
quantum_features = quantum_circuit(inputs)  # e.g., [-0.7, 0.5, -0.9, 0.9]

# Project back to original dimension
enhanced_features = projection_layer(quantum_features)  # [batch, hidden_size]

# Combine with classical features
final_features = classical_features + 0.1 * enhanced_features
```

**What this means:**
- Quantum outputs are **added** to classical features (with 0.1 weight)
- The quantum enhancement provides additional information
- The model learns to use both classical and quantum information

## How to Interpret the Results

### 1. **Quantum Measurement Values** (from `qml.expval(PauliZ)`)

```python
quantum_output = [-0.707, 0.5, -0.866, 0.906]
```

**Interpretation:**
- **Values close to +1**: Qubit strongly in |0⟩ state (classical-like)
- **Values close to -1**: Qubit strongly in |1⟩ state (classical-like)
- **Values close to 0**: Qubit in strong superposition (quantum-like)
- **Intermediate values**: Partial quantum behavior

**What this tells you:**
- If all values are near ±1: Quantum circuit is behaving classically
- If values are near 0: Strong quantum superposition is present
- Mixed values: Quantum circuit is using both classical and quantum effects

### 2. **Attention Scores After Quantum Enhancement**

```python
# Before quantum enhancement
classical_scores = attention_scores  # Standard BERT attention

# After quantum enhancement
quantum_scores = quantum_similarity(query, key)
enhanced_scores = classical_scores + 0.1 * quantum_scores
```

**Interpretation:**
- Quantum enhancement **modifies** attention scores by ~10%
- If quantum scores are large: Quantum features found strong patterns
- If quantum scores are small: Quantum features didn't add much
- The model learns when quantum enhancement helps vs. hurts

### 3. **Final Model Outputs**

```python
outputs = model(input_ids)
logits = outputs.logits  # [batch, seq_len, vocab_size]
```

**Interpretation:**
- These are standard BERT outputs (token predictions)
- Quantum enhancement affects the **last layer's attention**
- Compare quantum vs. classical model:
  - **Same performance**: Quantum didn't help (or hurt)
  - **Better performance**: Quantum found useful patterns
  - **Worse performance**: Quantum noise or overfitting

### 4. **During Training: What to Monitor**

```python
# Monitor these metrics:
loss = criterion(logits, labels)
quantum_params = [p for n, p in model.named_parameters() if 'quantum' in n]

# Check if quantum parameters are learning:
print(quantum_params[0].grad)  # Should be non-zero if learning
```

**What to look for:**
- **Quantum parameters changing**: Model is learning quantum patterns
- **Quantum parameters stuck**: Quantum circuit not contributing
- **Loss decreasing**: Quantum enhancement is helping
- **Loss increasing**: Quantum might be adding noise

## Example: Tracing a Single Forward Pass

```python
# Input
input_ids = torch.tensor([[101, 2023, 2003, 1037, 3231, 102]])  # "The cat is a [MASK]"

# Step 1: BERT processes input (classical)
hidden_states = bert_encoder(input_ids)  # [1, 6, 768]

# Step 2: Last layer attention (with quantum)
# For each of 6 tokens, quantum circuit runs:
for token_idx in range(6):
    token_features = hidden_states[0, token_idx, :]  # [768] features
    
    # Quantum encoding (if use_quantum_simulator=True):
    # - Take first 4 features: [0.5, -0.3, 0.8, 0.2]
    # - Normalize: [1.57, -0.94, 2.51, 0.63] radians
    # - Run quantum circuit:
    #   * Prepare: RY gates create superposition
    #   * Transform: RX, RY, RZ rotations
    #   * Entangle: CNOT gates
    #   * Measure: expval(PauliZ) → [-0.7, 0.5, -0.9, 0.9]
    
    quantum_output = quantum_circuit(token_features[:4])
    # Returns: tensor([-0.707, 0.5, -0.866, 0.906])
    
    # Project back: [4] → [768]
    enhanced_features = projection(quantum_output)  # [768]
    
    # Combine: classical + 0.1 * quantum
    final_features = token_features + 0.1 * enhanced_features

# Step 3: Attention computation uses enhanced features
attention_scores = compute_attention(enhanced_queries, enhanced_keys)

# Step 4: Final predictions
logits = model.lm_head(final_hidden_states)  # [1, 6, 30522]
predictions = torch.argmax(logits, dim=-1)  # Predicted token IDs
```

## Key Takeaways

1. **"Running" means**: Executing quantum gates on quantum states, then measuring
2. **Quantum outputs**: Expectation values in [-1, 1] range
3. **Integration**: Quantum features are added to classical features (10% weight)
4. **Interpretation**: 
   - Values near 0 = strong quantum behavior
   - Values near ±1 = classical behavior
   - Mixed = hybrid quantum-classical
5. **Final results**: Standard BERT outputs, but influenced by quantum enhancement

## Debugging Tips

```python
# Check if quantum circuit is actually running
if use_quantum_simulator:
    # Add print statements in quantum_feature_map_simulation
    print(f"Quantum input: {quantum_input}")
    print(f"Quantum output: {quantum_result}")
    print(f"Output range: [{quantum_result.min():.3f}, {quantum_result.max():.3f}]")

# Check quantum parameter gradients
for name, param in model.named_parameters():
    if 'quantum' in name and param.grad is not None:
        print(f"{name}: grad_norm = {param.grad.norm():.6f}")
```

This helps verify:
- Quantum circuit is executing
- Outputs are in expected range [-1, 1]
- Quantum parameters are learning (non-zero gradients)
