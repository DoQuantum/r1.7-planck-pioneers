import numpy as np
from transformers import BertTokenizer, BertForMaskedLM, logging
import torch
from transformers import DataCollatorWithPadding, AutoModel
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import AdamW
from tqdm.notebook import tqdm
import time
import math


def tokenize_sentence(sentence: str, tokenizer: any) -> np.ndarray:
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
        return


def get_BertMaskedLM_BertTokenizer() -> tuple:
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
    return model, tokenizer


def tokenize_fn(batch, tokenizer: any) -> dict:
    return tokenizer(batch["text"], truncation=True, padding=True, max_length=64)

def prepare_data(data: any, tokenizer: any) -> tuple:
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

    return train_loader, test_loader

class BertWithLastLayerAttentionClassifier(nn.Module):
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

        return logits, outputs.attentions  # Also return attention weights

def compute_mlm_accuracy(logits, labels):
    # logits: [B, L, V], labels: [B, L] with -100 where not masked
    preds = logits.argmax(dim=-1)              # [B, L]
    mask = labels != -100                      # [B, L]
    if mask.sum().item() == 0:
        return 0.0
    correct = (preds[mask] == labels[mask]).float().sum().item()
    total = mask.sum().item()
    return correct / total


def train_model_mlm(model, train_loader, val_loader, epochs=3, lr=2e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
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
                total_acc += compute_mlm_accuracy(out.logits, labels)
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

















































































