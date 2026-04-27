import torch
import time
import math
import os
from tqdm import tqdm
from datasets import load_dataset
from torch.optim import AdamW

# IMPORTS FROM YOUR FILES
from functions_MLM import (
    get_BertMaskedLM_BertTokenizer_MLM,
    prepare_wikitext_MLM,
    compute_mlm_accuracy_MLM
)
from custom_bert_lastlayer_attention import CustomBertForMaskedLM_LastLayerAttention

# ==============================================================================
# CONFIGURATION (Aligned to 2025 Paper + 8GB GPU Survival)
# ==============================================================================
EPOCHS = 15
LEARNING_RATE = 3e-5
BATCH_SIZE = 8  
GRADIENT_ACCUMULATION_STEPS = 8 
SEQUENCE_LENGTH = 120
N_QUBITS = 4

# ==============================================================================
# 1. SETUP & DATA LOADING
# ==============================================================================
print("Loading Tokenizer...")
_, tokenizer = get_BertMaskedLM_BertTokenizer_MLM()

print("Loading WikiText-2 Dataset...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

print("Preparing Data Chunks & Train/Val split...")
train_loader, val_loader = prepare_wikitext_MLM(
    data=dataset,
    tokenizer=tokenizer,
    batch_size=BATCH_SIZE,
    block_size=SEQUENCE_LENGTH
)

# ==============================================================================
# 2. DEVICE SETUP
# ==============================================================================
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU DETECTED: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("NO GPU DETECTED.")

# ==============================================================================
# 3. INITIALIZE MODEL & FREEZE LAYERS
# ==============================================================================
print(f"Initializing Quantum Model with {N_QUBITS} qubits...")
model = CustomBertForMaskedLM_LastLayerAttention.from_pretrained(
    'bert-base-uncased',
    n_qubits=N_QUBITS,
    use_quantum_simulator=True
)
model.to(device)

print("FORCE-LOADING ENTIRE TEACHER STATE (Body + Head)...")
from transformers import BertForMaskedLM
teacher = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.load_state_dict(teacher.state_dict(), strict=False)

# --- VRAM SAVER: FREEZE LOWER LAYERS ---
print("Freezing Classical BERT Embeddings and Layers 0-4 to save VRAM...")
for param in model.bert.embeddings.parameters():
    param.requires_grad = False
for i in range(5): # Freeze classical layers 0 through 4
    for param in model.bert.encoder.layer[i].parameters():
        param.requires_grad = False

# Only parameters with requires_grad=True (Layer 5, Layer 6 Quantum, Layers 7-11, and output head) will optimize
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, eps=1e-6)

# ==============================================================================
# 4. MAIN TRAINING LOOP
# ==============================================================================
print("\n" + "#"*60)
print(f"STARTING TRAINING: {EPOCHS} Epochs")
print("#"*60)

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    
    # --- TRAINING ---
    model.train()
    total_loss = 0
    train_start = time.time()
    
    optimizer.zero_grad()
    loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Training E{epoch+1}")
    
    for step, batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}
        
        outputs = model(**batch)
        loss = outputs.loss
        
        # Scale the loss for Gradient Accumulation
        loss = loss / GRADIENT_ACCUMULATION_STEPS
        
        if torch.isnan(loss):
            print(f"!!! NAN DETECTED at Epoch {epoch+1} !!! Skipping step.")
            optimizer.zero_grad() 
            continue
            
        loss.backward()

        # Update weights only every GRADIENT_ACCUMULATION_STEPS
        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (step + 1) == len(train_loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            
        # Multiply back by accumulation steps for accurate logging
        total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
        loop.set_postfix(loss=(loss.item() * GRADIENT_ACCUMULATION_STEPS))

    avg_train_loss = total_loss / len(train_loader)
    train_time = (time.time() - train_start) / 60
    print(f"   -> Avg Train Loss: {avg_train_loss:.4f} (Time: {train_time:.1f} min)")

    # --- VALIDATION ---
    model.eval()
    val_loss = 0
    total_acc = 0
    print("   -> Running Validation...")
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            val_loss += outputs.loss.item()
            total_acc += compute_mlm_accuracy_MLM(outputs.logits, batch["labels"])
    
    avg_val_loss = val_loss / len(val_loader)
    avg_val_acc = total_acc / len(val_loader)
    perplexity = math.exp(min(avg_val_loss, 20)) 
    
    print(f"   -> Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.4f} | Perplexity: {perplexity:.4f}")
    
    save_path = f"./QUANTUM_WIKI_4__epoch{epoch+1}"
    print(f"   Saving checkpoint to {save_path}...")
    model.save_pretrained(save_path)

print("\n" + "="*50)
print("FULL EXPERIMENT COMPLETE")
print("="*50)