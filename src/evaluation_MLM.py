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
# CONFIGURATION (Match your training run)
# =========================================================
N_QUBITS = 4  # Change this to 6 when you evaluate your 6-qubit run!
BATCH_SIZE = 8
SEQUENCE_LENGTH = 120

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

    try:
        if "QUANTUM" in model_path:
            print(f"   >>> ⚛️ Detected QUANTUM Checkpoint. Loading with {N_QUBITS} qubits...")
            model = CustomBertForMaskedLM_LastLayerAttention.from_pretrained(
                model_path,
                n_qubits=N_QUBITS,
                use_quantum_simulator=True 
            )
        else:
            print("   >>> 🤖 Detected CLASSICAL Checkpoint. Loading Standard BERT...")
            model = BertForMaskedLM.from_pretrained(model_path)
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
    return DataLoader(lm_datasets["test" if "test" in lm_datasets else "validation"], batch_size=batch_size, shuffle=False, collate_fn=collator)

# ---------------------------------------------------------
# 5. Main Runner
# ---------------------------------------------------------
def run_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    test_loader = prepare_test_loader(tokenizer, batch_size=BATCH_SIZE, block_size=SEQUENCE_LENGTH)

    model_paths = []

    # Search for all your saved WikiText epochs
    for epoch in range(1, 16): 
        path = f"./QUANTUM_WIKI_epoch{epoch}"
        if os.path.exists(path):
            model_paths.append(path)

    print(f"\n✅ Found {len(model_paths)} models to evaluate.")

    results = []

    for mp in model_paths:
        r = evaluate_model(mp, test_loader, device)
        if r:
            results.append(r)

    # Save to JSON
    output_file = "Quantum_WikiText_Evaluation.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*40)
    print("=== 🏆 EXPERIMENT COMPLETE ===")
    print(f"Saved results to {output_file}")
    print("="*40)

if __name__ == "__main__":
    run_evaluation()