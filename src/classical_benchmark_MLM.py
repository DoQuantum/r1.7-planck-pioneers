from functions_MLM import (
    tokenize_sentence_MLM,
    get_BertMaskedLM_BertTokenizer_MLM,
    tokenize_fn_MLM,
    prepare_data_MLM,
    train_model_mlm_MLM,
    compute_mlm_accuracy_MLM
)
from datasets import load_dataset
import torch
import numpy as np

################
## Example Input / Hidden States for Inspection
################

# Load pretrained BERT MLM model and tokenizer
model, tokenizer = get_BertMaskedLM_BertTokenizer_MLM()
 
# Inspect tokenization of a sample sentence
tokens = tokenize_sentence_MLM(sentence="The cat is a fatty patty.", tokenizer=tokenizer)
print("Tokenized sentence:", tokens)

# Show model details (MLM head)
print(model)

################
## Data Preparation
################

# Load a dataset for MLM fine-tuning
# You can use "imdb" small subset, or "wikitext" for a more standard MLM dataset
dataset = load_dataset("imdb")

# Prepare data for MLM fine-tuning
train_loader, test_loader = prepare_data_MLM(dataset, tokenizer)

################
## MLM Fine-tuning
################

# Train model on dynamically masked tokens
train_model_mlm_MLM(model, train_loader, test_loader, epochs=3, lr=2e-5)

################
## Notes on evaluation
################
# Accuracy (MLM): compute_mlm_accuracy_MLM is used internally
# Perplexity: printed during training
# Efficiency: track CPU memory usage; FLOPS are optional
