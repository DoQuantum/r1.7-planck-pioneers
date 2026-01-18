import torch
import time
import math
from tqdm import tqdm
from datasets import load_dataset
from torch.optim import AdamW

# IMPORTS FROM YOUR FILES
from functions_MLM import (
    get_BertMaskedLM_BertTokenizer_MLM,
    prepare_data_kfold_MLM
)
from custom_bert_lastlayer_attention import CustomBertForMaskedLM_LastLayerAttention

# ==============================================================================
# 1. SETUP & DATA LOADING
# ==============================================================================
print("Loading Tokenizer...")
_, tokenizer = get_BertMaskedLM_BertTokenizer_MLM()

print("Loading FULL IMDb Dataset...")
dataset = load_dataset("imdb")

print("Splitting into 5 Folds (We will only run Fold 1)...")
folds = prepare_data_kfold_MLM(
    data=dataset,
    tokenizer=tokenizer,
    n_splits=5 
)

# GRAB ONLY FOLD 1
# folds[0] is likely a tuple: (train_loader, val_loader)
train_loader, val_loader = folds[0]

# ==============================================================================
# 2. INITIALIZE MODEL (QUANTUM)
# ==============================================================================
# --- FORCE CPU (Safe Mode) ---
print("Forcing usage of CPU to avoid conflicts...")
device = torch.device("cpu") 
# -----------------------------

print("Initializing Quantum Model...")
model = CustomBertForMaskedLM_LastLayerAttention.from_pretrained(
    'bert-base-uncased',
    use_quantum_simulator=True 
)
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

# ==============================================================================
# 3. TRAINING LOOP (JUST ONE FOLD)
# ==============================================================================
print("\n" + "="*40)
print("STARTING TIMING RUN: FOLD 1 ONLY")
print(f"Training Samples: {len(train_loader.dataset)}")
print("="*40)

start_time = time.time()
epochs = 1 

for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    # Progress bar
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    
    for batch in loop:
        # Move batch to CPU
        batch = {k: v.to(device) for k, v in batch.items()}
        
        outputs = model(**batch)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    # Calculate Epoch Time
    epoch_end = time.time()
    elapsed = epoch_end - start_time
    print(f"Epoch {epoch+1} finished in {elapsed/60:.2f} minutes.")
    
    # Validation
    model.eval()
    val_loss = 0
    print("Running Validation...")
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            val_loss += outputs.loss.item()
    
    avg_val = val_loss / len(val_loader)
    print(f"Validation Loss: {avg_val:.4f} | Perplexity: {math.exp(min(avg_val, 20)):.4f}")

print("\n" + "="*40)
print("TIMING COMPLETE")
print(f"Total time for 1 Epoch: {(time.time() - start_time)/60:.2f} minutes")
print("="*40)