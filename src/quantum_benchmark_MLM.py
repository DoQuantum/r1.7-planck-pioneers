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
START_FOLD = 3   
EPOCHS = 3
LEARNING_RATE = 1e-5
BATCH_SIZE = 8  

# ==============================================================================
# 1. SETUP & DATA LOADING
# ==============================================================================
print("Loading Tokenizer...")
_, tokenizer = get_BertMaskedLM_BertTokenizer_MLM()

print("Loading FULL IMDb Dataset...")
dataset = load_dataset("imdb")

print(f"Splitting into {N_FOLDS} Folds...")
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

    # --- RESTART LOGIC: SKIP COMPLETED FOLDS ---
    if fold_num < START_FOLD:
        print(f"⏩ SKIPPING FOLD {fold_num} (Already Completed)")
        continue

    print("\n" + "#"*60)
    print(f"STARTING FOLD {fold_num}/{N_FOLDS}")
    print("#"*60)

    # --- A. INITIALIZE FRESH MODEL FOR THIS FOLD ---
    print(f"Initializing Fresh Quantum Model for Fold {fold_num}...")
    model = CustomBertForMaskedLM_LastLayerAttention.from_pretrained(
        'bert-base-uncased',
        n_qubits=4,
        use_quantum_simulator= True
    )
    model.to(device)
    
    print("FORCE-LOADING ENTIRE TEACHER STATE (Body + Head)...")
    from transformers import BertForMaskedLM
    
    teacher = BertForMaskedLM.from_pretrained('bert-base-uncased')
    teacher_state = teacher.state_dict()
    
    missing_keys, unexpected_keys = model.load_state_dict(teacher_state, strict=False)
    
    print(f"   - Missing Keys (Should be 0 for standard BERT parts): {len([k for k in missing_keys if 'quantum' not in k])}")
    print(f"   - Unexpected Keys (Should be all your Quantum stuff): {len(unexpected_keys)}")
    
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
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss

            # 1. NAN AUTO-SKIP
            if torch.isnan(loss):
                print(f"!!! NAN DETECTED at Epoch {epoch+1} !!! Skipping batch.")
                optimizer.zero_grad() 
                continue
            
            loss.backward()

            # 2. NAN GRADIENT CHECK
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

            # 3. GRADIENT CLIPPING
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
        perplexity = math.exp(min(avg_val_loss, 20)) 
        
        print(f"   -> Val Loss: {avg_val_loss:.4f} | Perplexity: {perplexity:.4f}")
        
        save_path = f"./QUANTUM_BASE_fold{fold_num}_epoch{epoch+1}"
        print(f"   Saving checkpoint to {save_path}...")
        model.save_pretrained(save_path)

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