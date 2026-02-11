import torch
import math
import json
import os
import sys
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    BertTokenizer,
    BertForMaskedLM,
    DataCollatorForLanguageModeling
)

# --- 1. PATH SETUP (CRITICAL FIX) ---
# This tells Python to look inside 'src' for your custom files
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# NOW we import your custom model
try:
    from custom_bert_lastlayer_attention import CustomBertForMaskedLM_LastLayerAttention
except ImportError:
    # Fallback if file is in root
    from custom_bert_lastlayer_attention import CustomBertForMaskedLM_LastLayerAttention

# ---------------------------------------------------------
# 2. MLM Accuracy Function
# ---------------------------------------------------------
def compute_mlm_accuracy(logits, labels):
    predictions = torch.argmax(logits, dim=-1)
    mask = labels != -100
    if mask.sum() == 0:
        return 0
    correct = (predictions[mask] == labels[mask]).sum().item()
    total = mask.sum().item()
    return correct / total

# ---------------------------------------------------------
# 3. Load and evaluate a single model
# ---------------------------------------------------------
def evaluate_model(model_path, test_loader, device):
    print(f"\n{'='*60}")
    print(f"🧐 EVALUATING: {model_path}")
    print(f"{'='*60}")

    # --- INTELLIGENT LOADING LOGIC ---
    try:
        if "QUANTUM" in model_path:
            print("   >>> ⚛️ Detected QUANTUM Checkpoint. Loading Custom Class...")
            # We must use the simulation flag for evaluation too
            model = CustomBertForMaskedLM_LastLayerAttention.from_pretrained(
                model_path,
                use_quantum_simulator=True 
            )
        else:
            print("   >>> 🤖 Detected CLASSICAL/STANDARD Checkpoint. Loading Standard BERT...")
            model = BertForMaskedLM.from_pretrained(model_path)
    except Exception as e:
        print(f"!!! CRITICAL ERROR loading {model_path}: {e}")
        return None

    model.to(device)
    model.eval()

    total_loss = 0
    total_acc = 0
    total_batches = 0

    # Progress bar
    loop = tqdm(test_loader, desc="Testing")
    
    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        logits = outputs.logits
        
        total_loss += loss.item()
        total_acc += compute_mlm_accuracy(logits, batch["labels"])
        total_batches += 1
        
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / total_batches
    avg_acc = total_acc / total_batches
    perplexity = math.exp(min(avg_loss, 20)) 

    print(f"\n>>> 🏁 FINAL RESULT: Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f} | PPL: {perplexity:.4f}")

    return {
        "model": model_path,
        "loss": avg_loss,
        "accuracy": avg_acc,
        "perplexity": perplexity
    }

# ---------------------------------------------------------
# 4. Prepare Test Data
# ---------------------------------------------------------
def prepare_test_loader(tokenizer, batch_size=16):
    print("\nLoading IMDb Test Split...")
    imdb = load_dataset("imdb")
    test_data = imdb["test"]

    print("Tokenizing Test Data (This might take a minute)...")
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_special_tokens_mask=True
        )

    tokenized = test_data.map(tokenize, batched=True, num_proc=4)
    
    # Keep only torch-compatible columns
    keep_cols = ["input_ids", "attention_mask", "labels"] # 'labels' is created by DataCollator, but we keep raw cols first
    tokenized = tokenized.remove_columns(
        [col for col in tokenized.column_names if col not in ["input_ids", "attention_mask"]]
    )

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    from torch.utils.data import DataLoader
    return DataLoader(tokenized, batch_size=batch_size, shuffle=False, collate_fn=collator)

# ---------------------------------------------------------
# 5. Main Runner
# ---------------------------------------------------------
def run_evaluation():
    # Detect GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    test_loader = prepare_test_loader(tokenizer)

    # --- GENERATE FILE LIST ---
    model_paths = []

    # 1. BASELINE (Standard BERT)
    # model_paths.append("bert-base-uncased") # Uncomment if you want to test raw BERT too

    # 2. YOUR QUANTUM MODELS (The Fix is Here: 'QUANTUM_BASE')
    for fold in range(1, 6):  # Folds 1 to 5
        for epoch in range(1, 4): # Epochs 1 to 3
            path = f"./QUANTUM_BASE_fold{fold}_epoch{epoch}"
            
            if os.path.exists(path):
                model_paths.append(path)
            else:
                print(f"⚠️ Warning: Checkpoint not found: {path}")

    print(f"\n✅ Found {len(model_paths)} models to evaluate.")

    results = []

    for mp in model_paths:
        r = evaluate_model(mp, test_loader, device)
        if r:
            results.append(r)

    # Save to JSON
    output_file = "Quantum_Baseline_Evaluation.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*40)
    print("=== 🏆 EXPERIMENT COMPLETE ===")
    print(f"Saved results to {output_file}")
    print("="*40)

if __name__ == "__main__":
    run_evaluation()