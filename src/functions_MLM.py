import numpy as np
from transformers import BertTokenizer, BertForMaskedLM, logging
import torch
from transformers import DataCollatorWithPadding, AutoModel, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
from torch.optim import AdamW
from tqdm.notebook import tqdm
import time
import math
from sklearn.model_selection import KFold
from custom_bert_lastlayer_attention import CustomBertForMaskedLM_LastLayerAttention

def tokenize_sentence_MLM(sentence: str, tokenizer) -> dict:
    """
    See how a sentence is tokenized by a given tokenizer (for Masked Language Modeling). 

    Args:
        sentence (str): Sentence to tokenize.
        tokenizer: Hugging Face tokenizer instance.

    Returns:
        dict: A dictionary containing tokens and their corresponding IDs.
    """

    # Tokenize the sentence (return PyTorch tensors)
    inputs = tokenizer(sentence, return_tensors='pt')

    # Convert token IDs to readable tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    print(f"\n🔹 Original sentence:\n{sentence}")
    print(f"🔹 Tokens:\n{tokens}")
    print(f"🔹 Token IDs:\n{inputs['input_ids'][0].tolist()}")

    return {"tokens": tokens, "ids": inputs['input_ids'][0].tolist()}

def get_BertMaskedLM_BertTokenizer_MLM() -> tuple:
    """
    Load the BERT model and tokenizer for Masked Language Modeling (MLM).

    Returns:
        tuple: (model, tokenizer)
            model (BertForMaskedLM): Pretrained BERT MLM model.
            tokenizer (BertTokenizer): Corresponding tokenizer.
    """

    # Silence "Some weights not used" warnings — expected for MLM fine-tuning
    logging.set_verbosity_error()

    # Load pretrained model configured for MLM
    model = BertForMaskedLM.from_pretrained(
        'bert-base-uncased',
        output_attentions=True
    )

    # Load matching tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Ensure tokenizer padding token is defined (important for custom datasets)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer

def tokenize_fn_MLM(batch, tokenizer: any) -> dict:
    """
    Tokenize a batch of text for Masked Language Modeling (MLM).

    Args:
        batch (dict): A batch from the dataset containing a "text" field.
        tokenizer (PreTrainedTokenizer): Tokenizer compatible with BERT.

    Returns:
        dict: Tokenized and padded input IDs and attention masks.
    """
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_special_tokens_mask=True  # useful for masking later
    )

def prepare_data_MLM(data: any, tokenizer: any) -> tuple:
    """
    Prepare dataset and dataloaders for Masked Language Modeling (MLM) fine-tuning.

    Args:
        data (DatasetDict): Dataset with 'train' and 'test' splits.
        tokenizer (PreTrainedTokenizer): Tokenizer for tokenization.

    Returns:
        tuple: (train_loader, test_loader)
    """

    small_train = data["train"].shuffle(seed=42)
    small_test = data["test"].shuffle(seed=42)

    # Tokenize text batches
    tokenized_train = small_train.map(lambda b: tokenize_fn_MLM(b, tokenizer), batched=True, num_proc=4)
    tokenized_test = small_test.map(lambda b: tokenize_fn_MLM(b, tokenizer), batched=True, num_proc=4)

    # Remove any unused columns (important for MLM fine-tuning)
    tokenized_train = tokenized_train.remove_columns([col for col in tokenized_train.column_names if col != "input_ids" and col != "attention_mask" and col != "special_tokens_mask"])
    tokenized_test = tokenized_test.remove_columns([col for col in tokenized_test.column_names if col != "input_ids" and col != "attention_mask" and col != "special_tokens_mask"])

    # Collator handles dynamic padding + random masking for MLM
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15  # 15% of tokens randomly masked
    )

    # Create dataloaders
    train_loader = DataLoader(tokenized_train, batch_size=4, shuffle=True, collate_fn=collator)
    test_loader = DataLoader(tokenized_test, batch_size=4, shuffle=False, collate_fn=collator)

    return train_loader, test_loader

def prepare_data_kfold_MLM(data: any, tokenizer: any, n_splits: int = 5) -> list:
    """
    Prepare dataset for k-fold cross-validation in Masked Language Modeling (MLM) fine-tuning.

    Args:
        data (DatasetDict): Dataset with 'train' split (we'll use this for k-fold).
        tokenizer (PreTrainedTokenizer): Tokenizer for tokenization.
        n_splits (int): Number of folds for cross-validation.

    Returns:
        list: List of tuples (train_loader, val_loader) for each fold.
    """
    # Use the train split for k-fold cross-validation
    dataset = data["train"].shuffle(seed=42)
    
    # Tokenize all data
    tokenized_dataset = dataset.map(lambda b: tokenize_fn_MLM(b, tokenizer), batched=True, num_proc=4)
    
    # Remove any unused columns
    tokenized_dataset = tokenized_dataset.remove_columns([
        col for col in tokenized_dataset.column_names 
        if col not in ["input_ids", "attention_mask", "special_tokens_mask"]
    ])
    
    # Initialize k-fold
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    folds = []
    dataset_size = len(tokenized_dataset)
    
    # Create train/val splits for each fold
    for fold, (train_indices, val_indices) in enumerate(kfold.split(range(dataset_size))):
        # Create subset datasets for train and validation
        train_subset = Subset(tokenized_dataset, train_indices)
        val_subset = Subset(tokenized_dataset, val_indices)
        
        # Collator handles dynamic padding + random masking for MLM
        collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=0.15  # 15% of tokens randomly masked
        )
        
        # Create dataloaders for this fold
        train_loader = DataLoader(train_subset, batch_size=4, shuffle=True, collate_fn=collator)
        val_loader = DataLoader(val_subset, batch_size=4, shuffle=False, collate_fn=collator)
        
        folds.append((train_loader, val_loader))
        print(f"Fold {fold+1}: Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")
    
    return folds

def compute_mlm_accuracy_MLM(logits, labels):
    """
    Compute token-level accuracy for Masked Language Modeling (MLM).

    Args:
        logits (torch.Tensor): Model predictions, shape [batch_size, seq_len, vocab_size]
        labels (torch.Tensor): Ground truth labels, shape [batch_size, seq_len], with -100 for unmasked tokens

    Returns:
        float: MLM accuracy (correct predictions / total masked tokens)
    """
    # Get predicted token IDs
    preds = logits.argmax(dim=-1)  # [B, L]

    # Only consider masked positions
    mask = labels != -100  # [B, L]

    if mask.sum(). item() == 0:
        return 0.0

    correct = (preds[mask] == labels[mask]).float().sum().item()
    total = mask.sum().item()

    return correct / total

def train_model_mlm_kfold_MLM(model_class, folds, epochs=3, lr=2e-5, n_splits=5):
    """
    Train model using k-fold cross-validation for Masked Language Modeling (MLM).

    Args:
        model_class: BERT model class to instantiate for each fold
        folds (list): List of (train_loader, val_loader) tuples for each fold
        epochs (int): Number of epochs to train
        lr (float): Learning rate
        n_splits (int): Number of folds (for saving models)

    Returns:
        dict: Dictionary containing results for each fold and overall statistics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_results = {}
    
    # Track overall metrics
    all_val_losses = []
    all_val_accuracies = []
    all_perplexities = []
    
    for fold_idx, (train_loader, val_loader) in enumerate(folds):
        print(f"\n{'='*50}")
        print(f"Training Fold {fold_idx+1}/{len(folds)}")
        print(f"{'='*50}")
        
        # Create a fresh model for each fold to ensure independence
        model = model_class.from_pretrained(
            'bert-base-uncased',
            output_attentions=True
        )
        model.to(device)
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        
        fold_train_losses = []
        fold_val_losses = []
        fold_val_accuracies = []
        
        for epoch in range(epochs):
            model.train()
            total_train_loss = 0.0
            t0 = time.time()
            
            # Training loop
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), 
                              desc=f"Fold {fold_idx+1} Epoch {epoch+1}", dynamic_ncols=True)
            
            for step, batch in progress_bar:
                # For MLM, batch must include *dynamically masked* inputs & labels
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                labels = batch["labels"].to(device)  # [B, L], -100 where not masked
                
                optimizer.zero_grad()
                # Let the model compute token-level loss
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out.loss  # scalar
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
                
                # Live display in tqdm
                progress_bar.set_postfix({"step_loss": f"{loss.item():.4f}"})
                
                if (step + 1) % 100 == 0:
                    print(f"Fold {fold_idx+1} | Epoch {epoch+1} | Step {step+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
            
            avg_train_loss = total_train_loss / max(1, len(train_loader))
            epoch_time = time.time() - t0
            fold_train_losses.append(avg_train_loss)
            
            # Validation loop
            model.eval()
            val_loss = 0.0
            total_acc = 0.0
            
            with torch.no_grad():
                t_eval0 = time.time()
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch.get("attention_mask", None)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(device)
                    labels = batch["labels"].to(device)
                    
                    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    val_loss += out.loss.item()
                    total_acc += compute_mlm_accuracy_MLM(out.logits, labels)
                val_wall = time.time() - t_eval0
            
            avg_val_loss = val_loss / max(1, len(val_loader))
            avg_val_acc = total_acc / max(1, len(val_loader))
            perplexity = math.exp(avg_val_loss) if avg_val_loss < 20 else float("inf")
            
            fold_val_losses.append(avg_val_loss)
            fold_val_accuracies.append(avg_val_acc)
            
            # Store fold results
            print({
                "fold": fold_idx + 1,
                "epoch": epoch + 1,
                "train_loss": round(avg_train_loss, 4),
                "val_loss": round(avg_val_loss, 4),
                "perplexity": "inf" if perplexity == float("inf") else round(perplexity, 4),
                "mlm_accuracy": round(avg_val_acc, 4),
                "epoch_time_s": round(epoch_time, 2),
                "val_wall_time_s": round(val_wall, 2),
            })
            
            # Save model checkpoint for this fold and epoch
            epoch_save_path = f"./bert_mlm_finetuned_fold{fold_idx+1}_epoch{epoch+1}"
            model.save_pretrained(epoch_save_path)
            print(f"Model checkpoint for fold {fold_idx+1}, epoch {epoch+1} saved to {epoch_save_path}")
        
        # Store results for this fold
        fold_results[f"fold_{fold_idx+1}"] = {
            "train_losses": fold_train_losses,
            "val_losses": fold_val_losses,
            "val_accuracies": fold_val_accuracies,
            "best_val_loss": min(fold_val_losses),
            "best_val_accuracy": max(fold_val_accuracies)
        }
        
        # Add to overall metrics
        all_val_losses.append(min(fold_val_losses))
        all_val_accuracies.append(max(fold_val_accuracies))
        all_perplexities.append(math.exp(min(fold_val_losses)) if min(fold_val_losses) < 20 else float("inf"))
    
    # Calculate overall statistics
    results = {
        "fold_results": fold_results,
        "overall": {
            "mean_val_loss": np.mean(all_val_losses),
            "std_val_loss": np.std(all_val_losses),
            "mean_val_accuracy": np.mean(all_val_accuracies),
            "std_val_accuracy": np.std(all_val_accuracies),
            "mean_perplexity": np.mean([p for p in all_perplexities if p != float("inf")]),
            "std_perplexity": np.std([p for p in all_perplexities if p != float("inf")])
        }
    }
    
    # Print overall results
    print(f"\n{'='*50}")
    print(f"K-Fold Cross-Validation Results (k={len(folds)})")
    print(f"{'='*50}")
    print(f"Mean Validation Loss: {results['overall']['mean_val_loss']:.4f} ± {results['overall']['std_val_loss']:.4f}")
    print(f"Mean Validation Accuracy: {results['overall']['mean_val_accuracy']:.4f} ± {results['overall']['std_val_accuracy']:.4f}")
    print(f"Mean Perplexity: {results['overall']['mean_perplexity']:.4f} ± {results['overall']['std_perplexity']:.4f}")
    
    return results

def train_model_mlm_MLM(model, train_loader, val_loader, epochs=3, lr=2e-5, save_path="./bert_mlm_finetuned"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    global_step = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}", dynamic_ncols=True)
        for step, batch in progress_bar:

            # For MLM, batch must include *dynamically masked* inputs & labels from a DataCollatorForLanguageModeling
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
                labels = batch["labels"].to(device)  # [B, L], -100 where not masked

            optimizer.zero_grad()
            # Let the model compute token-level loss
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss # scalar
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            global_step += 1

        # live display in tqdm
            progress_bar.set_postfix({"step_loss": f"{loss.item():.4f}", "global_step": global_step})

            if (step + 1) % 100 == 0:
                print(f"Epoch {epoch+1} | Step {step+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / max(1, len(train_loader))
        epoch_time = time.time() - t0

        # ----- Evaluation -----
        model.eval()
        val_loss = 0.0
        total_acc = 0.0
        with torch.no_grad():
            t_eval0 = time.time()
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                    labels = batch["labels"].to(device)

                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += out.loss.item()
                total_acc += compute_mlm_accuracy_MLM(out.logits, labels)
            val_wall = time.time() - t_eval0

        avg_val_loss = val_loss / max(1, len(val_loader))
        avg_val_acc = total_acc / max(1, len(val_loader))
        perplexity = math.exp(avg_val_loss) if avg_val_loss < 20 else float("inf")

        # Optional: peak memory (GPU)
        peak_mem_gb = None
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            # One extra tiny eval to populate peak
            _ = model(input_ids=input_ids[:1], attention_mask=attention_mask[:1] if attention_mask is not None else None, labels=labels[:1])
            peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)

        print({
            "epoch": epoch + 1,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "perplexity": "inf" if perplexity == float("inf") else round(perplexity, 4),
            "mlm_accuracy": round(avg_val_acc, 4),
            "epoch_time_s": round(epoch_time, 2),
            "val_wall_time_s": round(val_wall, 2),
            "peak_memory_gb": None if peak_mem_gb is None else round(peak_mem_gb, 3),
        })

        epoch_save_path = f"{save_path}_epoch{epoch+1}"
        model.save_pretrained(epoch_save_path)
        print(f"Model checkpoint saved to {epoch_save_path}")
    
    model.save_pretrained(save_path)
    print(f"Final finetuned model saved to {save_path}")
