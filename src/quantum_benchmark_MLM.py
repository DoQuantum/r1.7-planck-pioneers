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
    prepare_data_kfold_MLM
)
from custom_bert_lastlayer_attention import CustomBertForMaskedLM_LastLayerAttention

# ==============================================================================
# CONFIGURATION
# ==============================================================================
N_FOLDS = 5
EPOCHS = 3
LEARNING_RATE = 1e-5
BATCH_SIZE = 8  # Adjust this depending on your GPU memory (try 16 if 8 is easy)

# ==============================================================================
# 1. SETUP & DATA LOADING
# ==============================================================================
print("Loading Tokenizer...")
_, tokenizer = get_BertMaskedLM_BertTokenizer_MLM()

print("Loading FULL IMDb Dataset...")
# This loads the full 25k train / 25k test dataset
dataset = load_dataset("imdb")

print(f"Splitting into {N_FOLDS} Folds...")
# This will handle the 80/20 split logic for us
folds = prepare_data_kfold_MLM(
    data=dataset,
    tokenizer=tokenizer,
    n_splits=N_FOLDS,
    batch_size=BATCH_SIZE
)

# ==============================================================================
# 2. DEVICE SETUP (GPU ENABLED)
# ==============================================================================
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU DETECTED: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("NO GPU DETECTED. Training will be extremely slow.")

# ==============================================================================
# 3. MAIN TRAINING LOOP (ALL FOLDS)
# ==============================================================================
results = []

for fold_idx, (train_loader, val_loader) in enumerate(folds):
    fold_num = fold_idx + 1

    # ==========================================
    # SKIP ALREADY COMPLETED FOLDS (1 & 2)
    # ==========================================
    if fold_num < 5: 
        print(f"Skipping Fold {fold_num} (Already verified safe)...")
        continue
    # ==========================================

    print("\n" + "#"*60)
    print(f"STARTING FOLD {fold_num}/{N_FOLDS}")
    print("#"*60)

    # --- A. INITIALIZE FRESH MODEL FOR THIS FOLD ---
    # We must reload from scratch so Fold 2 doesn't start with Fold 1's weights
    print(f"Initializing Fresh Quantum Model for Fold {fold_num}...")
    model = CustomBertForMaskedLM_LastLayerAttention.from_pretrained(
        'bert-base-uncased',
        use_quantum_simulator=True 
    )
    model.to(device)
    
    # optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-6)

    # --- B. EPOCH LOOP ---
    for epoch in range(EPOCHS):
        print(f"\nFold {fold_num} - Epoch {epoch+1}/{EPOCHS}")
        
        # --- TRAINING ---
        model.train()
        total_loss = 0
        train_start = time.time()
        
        loop = tqdm(train_loader, desc=f"Training F{fold_num}-E{epoch+1}")
        
        for batch in loop:
            # Move batch to GPU
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss

            # 1. NAN AUTO-SKIP (The Eject Button)
            if torch.isnan(loss):
                print(f"!!! NAN DETECTED at Epoch {epoch+1} !!! Skipping batch.")
                optimizer.zero_grad() 
                continue
            
            loss.backward()

            # ====================================================
            # 2. NEW: NAN GRADIENT CHECK (The "Silent Killer" Fix)
            # ====================================================
            valid_gradients = True
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        valid_gradients = False
                        break
            
            if not valid_gradients:
                print(f"!!! NAN GRADIENTS DETECTED at Epoch {epoch+1} !!! Skipping step.")
                optimizer.zero_grad()
                continue
            # ====================================================

            # 2. GRADIENT CLIPPING (The Circuit Breaker)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / len(train_loader)
        train_time = (time.time() - train_start) / 60
        print(f"   -> Avg Train Loss: {avg_train_loss:.4f} (Time: {train_time:.1f} min)")

        # --- VALIDATION ---
        model.eval()
        val_loss = 0
        print("   -> Running Validation...")
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        perplexity = math.exp(min(avg_val_loss, 20)) # Cap at 20 to avoid overflow
        
        print(f"   -> Val Loss: {avg_val_loss:.4f} | Perplexity: {perplexity:.4f}")
        
        save_path = f"./QUANTUM_FULL_fold{fold_num}_epoch{epoch+1}"
        print(f"   Saving checkpoint to {save_path}...")
        model.save_pretrained(save_path)

    # Log simple result
    results.append({
        "fold": fold_num,
        "final_loss": avg_val_loss,
        "final_perplexity": perplexity
    })

print("\n" + "="*50)
print("FULL EXPERIMENT COMPLETE")
print("="*50)
for r in results:
    print(f"Fold {r['fold']}: Perplexity = {r['final_perplexity']:.4f}")