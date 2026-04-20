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

# --- 1. PATH SETUP ---
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from custom_bert_lastlayer_attention import CustomBertForMaskedLM_LastLayerAttention
except ImportError:
    from custom_bert_lastlayer_attention import CustomBertForMaskedLM_LastLayerAttention

# =========================================================
# CONFIGURATION: THE BATCH EVALUATION QUEUE
# =========================================================
BATCH_SIZE = 8
SEQUENCE_LENGTH = 120

# CAREFULLY UPDATED: Matching the exact prefixes of your new stress tests
# All of these runs utilize the 4-Qubit architecture.
EVAL_QUEUE = [
    {"prefix": "./QUANTUM_WIKI_4_100residlayer_epoch", "n_qubits": 4, "out_file": "Eval_4Q_100_percent.json"},
    {"prefix": "./QUANTUM_WIKI_4_75%_epoch",           "n_qubits": 4, "out_file": "Eval_4Q_75_percent.json"},
    {"prefix": "./QUANTUM_WIKI_4_11layer_epoch",       "n_qubits": 4, "out_file": "Eval_4Q_Layer11.json"},
    {"prefix": "./QUANTUM_WIKI_4_2layer_epoch",        "n_qubits": 4, "out_file": "Eval_4Q_Layer2.json"},
    {"prefix": "./QUANTUM_WIKI_4_depthtest_epoch",     "n_qubits": 4, "out_file": "Eval_4Q_DepthTest.json"}
]

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
def evaluate_model(model_path, test_loader, device, n_qubits):
    print(f"\n{'='*60}")
    print(f"🧐 EVALUATING: {model_path} (Expecting {n_qubits} Qubits)")
    print(f"{'='*60}")

    try:
        model = CustomBertForMaskedLM_LastLayerAttention.from_pretrained(
            model_path,
            n_qubits=n_qubits,
            use_quantum_simulator=True 
        )
    except Exception as e:
        print(f"!!! CRITICAL ERROR loading {model_path}: {e}")
        return None

    model.to(device)
    model.eval()

    total_loss = 0
    total_acc = 0
    total_batches = 0

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
# 4. Prepare WikiText Test Data (Chunked for VRAM)
# ---------------------------------------------------------
def prepare_test_loader(tokenizer, batch_size=8, block_size=120):
    print("\nLoading WikiText-2 Test Split...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    test_data = dataset["test"]

    print("Tokenizing Test Data...")
    def tokenize_function(examples):
        return tokenizer(examples["text"], return_special_tokens_mask=True)

    tokenized_datasets = test_data.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

    print(f"Chunking Test Data into blocks of {block_size} tokens...")
    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    lm_datasets = tokenized_datasets.map(group_texts, batched=True, batch_size=1000, num_proc=4)

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    from torch.utils.data import DataLoader
    return DataLoader(lm_datasets, batch_size=batch_size, shuffle=False, collate_fn=collator)

# ---------------------------------------------------------
# 5. Main Runner
# ---------------------------------------------------------
def run_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    test_loader = prepare_test_loader(tokenizer, batch_size=BATCH_SIZE, block_size=SEQUENCE_LENGTH)

    # LOOP THROUGH EVERY CONFIGURATION IN THE QUEUE
    for config in EVAL_QUEUE:
        prefix = config["prefix"]
        n_qubits = config["n_qubits"]
        out_file = config["out_file"]
        
        print("\n" + "*"*70)
        print(f"🚀 STARTING JOB: {prefix} ({n_qubits} Qubits)")
        print("*"*70)

        model_paths = []

        # Search for all saved epochs for this specific configuration
        for epoch in range(1, 16): 
            path = f"{prefix}{epoch}"
            if os.path.exists(path):
                model_paths.append(path)

        if len(model_paths) == 0:
            print(f"⚠️ Warning: No checkpoints found starting with {prefix}. Skipping...")
            continue

        print(f"✅ Found {len(model_paths)} models. Beginning evaluation...")

        results = []

        for mp in model_paths:
            # Pass the correct n_qubits into the evaluation function
            r = evaluate_model(mp, test_loader, device, n_qubits)
            if r:
                results.append(r)

        # Save to this configuration's specific JSON file
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"🎉 Finished job. Saved results to {out_file}")

    print("\n" + "="*50)
    print("=== 🏆 ALL JOBS IN QUEUE COMPLETE ===")
    print("="*50)

if __name__ == "__main__":
    run_evaluation()