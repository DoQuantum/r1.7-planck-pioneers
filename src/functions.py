import numpy as np
from transformers import BertTokenizer, BertForMaskedLM, logging
import torch
from transformers import DataCollatorWithPadding, AutoModel, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import AdamW
from tqdm.notebook import tqdm
import time
import math


'''def tokenize_sentence(sentence: str, tokenizer: any) -> np.ndarray:
    """See how a sentence is tokenized by a specific tokenizer.

    Args:
        sentence (str): Sentence to tokenize.
        tokenizer (str): Tokenizer used on sentence.

    Returns:
        np.ndarray: Tokenized sentence
    """

    # Get tokenized sentence
    if tokenizer.name_or_path == 'bert-base-uncased':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Get tokenized sentence in a format BERT can use (can't use lists, can use tensors)
        inputs = tokenizer(sentence, return_tensors='pt')

        # return tokenized sentence in a readable format (list)
        return tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    else:
        return'''

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



'''def get_BertMaskedLM_BertTokenizer() -> tuple:
    """Get BERTMaskedLM model and BertTokenizer.

    Returns:
        tuple: A tuple containing the BERTMaskedLM model and BertTokenizer.
    """


    # Silence the notification that BERT has unused weights, it's expected
    # We only use masked language modeling (MLM) head, to predict unseen words
    # We don't use next sentence prediction (NSP), to predict next sentence
    logging.set_verbosity_error()

    # Load BERT model
    model = BertForMaskedLM.from_pretrained('bert-base-uncased', output_attentions=True)

    # Load Berttokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Return BERTMaskedLM model and BertTokenizer
    return model, tokenizer'''

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

'''def tokenize_fn(batch, tokenizer: any) -> dict:
    return tokenizer(batch["text"], truncation=True, padding=True, max_length=64)'''

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
        max_length=64,
        return_special_tokens_mask=True  # useful for masking later
    )

'''def prepare_data(data: any, tokenizer: any) -> tuple:
    """Prepare data for model training and evaluation.

    Args:
        data (any): Dataset to prepare.
        tokenizer (any): Tokenizer used on dataset.

    Returns:
        any: Prepared dataset.
    """
    # Initialize datasets
    small_train = data["train"].shuffle(seed=42).select(range(2000))
    small_test  = data["test"].shuffle(seed=42).select(range(2000))

    # Apply tokenization (pass tokenizer via a lambda so map receives a single-arg callable)
    tokenized_train = small_train.map(lambda b: tokenize_fn(b, tokenizer), batched=True)
    tokenized_test  = small_test.map(lambda b: tokenize_fn(b, tokenizer), batched=True)

    # Set format for PyTorch
    tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    tokenized_test.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Dynamic padding for batches
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    # Dataloaders
    train_loader = DataLoader(tokenized_train, batch_size=1024, shuffle=True, collate_fn=collator)
    test_loader  = DataLoader(tokenized_test, batch_size=1024, shuffle=False, collate_fn=collator)

    return train_loader, test_loader'''

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
    train_loader = DataLoader(tokenized_train, batch_size=16, shuffle=True, collate_fn=collator)
    test_loader = DataLoader(tokenized_test, batch_size=16, shuffle=False, collate_fn=collator)

    return train_loader, test_loader

'''class BertWithLastLayerAttentionClassifier(nn.Module):
    def __init__(self, model, num_classes=2):
        super().__init__()
        self.bert = model
        
        # Freeze all layers except the last encoder layer
        for name, param in self.bert.named_parameters():
            # Only unfreeze Layer 5 and classifier
            if "encoder.layer.5" not in name and "pooler" not in name:
                param.requires_grad = False
        
        # Classification head on [CLS]
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        # Get hidden states and attention maps
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            output_attentions=True)
        
        last_hidden_state = outputs.last_hidden_state  # (batch, seq_len, hidden)
        cls_embedding = last_hidden_state[:, 0, :]     # Take [CLS] token representation
        logits = self.classifier(cls_embedding)        # Binary classification logits

        return logits, outputs.attentions  # Also return attention weights'''

'''def compute_mlm_accuracy(logits, labels):
    # logits: [B, L, V], labels: [B, L] with -100 where not masked
    preds = logits.argmax(dim=-1)              # [B, L]
    mask = labels != -100                      # [B, L]
    if mask.sum().item() == 0:
        return 0.0
    correct = (preds[mask] == labels[mask]).float().sum().item()
    total = mask.sum().item()
    return correct / total'''

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

    if mask.sum().item() == 0:
        return 0.0

    correct = (preds[mask] == labels[mask]).float().sum().item()
    total = mask.sum().item()

    return correct / total

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

















































































