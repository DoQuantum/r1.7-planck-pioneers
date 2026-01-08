import torch
import math
import json
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    BertTokenizer,
    BertForMaskedLM,
    DataCollatorForLanguageModeling
)

# ---------------------------------------------------------
# 1. MLM Accuracy Function (correct)
# ---------------------------------------------------------

def compute_mlm_accuracy(logits, labels):
    """
    Compute MLM accuracy:
    - Only evaluate positions where labels != -100 (masked positions)
    """
    predictions = torch.argmax(logits, dim=-1)

    mask = labels != -100  # only evaluate masked positions
    if mask.sum() == 0:
        return 0  # nothing to compare

    correct = (predictions[mask] == labels[mask]).sum().item()
    total = mask.sum().item()

    return correct / total


# ---------------------------------------------------------
# 2. Load and evaluate a single model
# ---------------------------------------------------------

def evaluate_model(model_path, test_loader, device):
    print(f"\nLoading model: {model_path}")
    model = BertForMaskedLM.from_pretrained(model_path).to(device)
    model.eval()

    total_loss = 0
    total_acc = 0
    batches = 0

    print("Running evaluation...")
    for batch in tqdm(test_loader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

        loss = outputs.loss
        logits = outputs.logits
        
        total_loss += loss.item()
        total_acc += compute_mlm_accuracy(logits, labels)
        batches += 1

    avg_loss = total_loss / batches
    avg_acc = total_acc / batches
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")

    return {
        "model": model_path,
        "loss": avg_loss,
        "accuracy": avg_acc,
        "perplexity": perplexity
    }


# ---------------------------------------------------------
# 3. Prepare IMDb test data with correct MLM masking
# ---------------------------------------------------------

def prepare_test_loader(tokenizer, batch_size=8):
    print("Loading dataset...")
    imdb = load_dataset("imdb")

    test_data = imdb["test"]

    print("Tokenizing...")
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding=False,
            max_length=512,
            return_special_tokens_mask=True
        )

    tokenized = test_data.map(tokenize, batched=True, num_proc=4)

    # Important: keep only useful columns
    tokenized = tokenized.remove_columns(
        [col for col in tokenized.column_names if col not in ["input_ids", "attention_mask", "special_tokens_mask"]]
    )

    # ---- Correct masking: exactly same as training ----
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    from torch.utils.data import DataLoader
    test_loader = DataLoader(
        tokenized,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator
    )

    return test_loader


# ---------------------------------------------------------
# 4. Main evaluation runner
# ---------------------------------------------------------

def run_evaluation():
    device = torch.device("cpu")
    print("Using device:", device)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    test_loader = prepare_test_loader(tokenizer)

    model_paths = [
        "bert-base-uncased", 
    ]

    '''"./bert_mlm_finetuned_epoch3",
        "./bert_mlm_finetuned_fold1_epoch3",
        "./bert_mlm_finetuned_fold2_epoch3",
        "./bert_mlm_finetuned_fold3_epoch3",
        "./bert_mlm_finetuned_fold4_epoch3",
        "./bert_mlm_finetuned_fold5_epoch3",'''

    results = []

    for mp in model_paths:
        try:
            r = evaluate_model(mp, test_loader, device)
            results.append(r)
            print(f"\n{mp} RESULTS:")
            print(f"Loss: {r['loss']:.4f}")
            print(f"Accuracy: {r['accuracy']:.4f}")
            print(f"Perplexity: {r['perplexity']:.4f}")

        except Exception as e:
            print(f"Failed to evaluate {mp}: {e}")

    # Save to JSON
    with open("evaluation_results2.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== DONE ===")
    print("Saved results to evaluation_results.json\n")


# ---------------------------------------------------------
# 5. Entry
# ---------------------------------------------------------

if __name__ == "__main__":
    run_evaluation()
