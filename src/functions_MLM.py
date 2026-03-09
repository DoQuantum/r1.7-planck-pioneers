import numpy as np
from transformers import BertTokenizer, BertForMaskedLM, logging
import torch
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
import math

def get_BertMaskedLM_BertTokenizer_MLM() -> tuple:
    """
    Load the BERT model and tokenizer for Masked Language Modeling (MLM).
    """
    logging.set_verbosity_error()
    model = BertForMaskedLM.from_pretrained('bert-base-uncased', output_attentions=True)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer

def prepare_wikitext_MLM(data: any, tokenizer: any, batch_size: int = 8, block_size: int = 120) -> tuple:
    """
    Prepare WikiText-2 dataset. Tokenizes all text, concatenates it, and chunks it 
    into exact blocks of 'block_size' (120) to match the paper's VRAM constraints.
    """
    # 1. Tokenize all text
    def tokenize_function(examples):
        return tokenizer(examples["text"], return_special_tokens_mask=True)

    print("Tokenizing raw WikiText-2 data...")
    tokenized_datasets = data.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

    # 2. Group texts into chunks of 120
    def group_texts(examples):
        # Concatenate all texts
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # Drop the remainder to ensure exact block sizes
        total_length = (total_length // block_size) * block_size
        # Split by chunks
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    print(f"Chunking data into blocks of {block_size} tokens...")
    lm_datasets = tokenized_datasets.map(group_texts, batched=True, batch_size=1000, num_proc=4)

    # 3. Dynamic Masking (15%)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15 
    )

    # 4. Create standard Train/Val dataloaders
    train_loader = DataLoader(lm_datasets["train"], batch_size=batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(lm_datasets["validation"], batch_size=batch_size, shuffle=False, collate_fn=collator)

    return train_loader, val_loader

def compute_mlm_accuracy_MLM(logits, labels):
    """
    Compute token-level accuracy for Masked Language Modeling (MLM).
    """
    preds = logits.argmax(dim=-1)  
    mask = labels != -100  

    if mask.sum().item() == 0:
        return 0.0

    correct = (preds[mask] == labels[mask]).float().sum().item()
    total = mask.sum().item()

    return correct / total