from functions_MLM import (
    tokenize_sentence_MLM,
    get_BertMaskedLM_BertTokenizer_MLM,
    prepare_data_kfold_MLM,  # Updated function for k-fold
    train_model_mlm_kfold_MLM,  # New k-fold training function
    compute_mlm_accuracy_MLM
)
from datasets import load_dataset
import torch
import numpy as np
from transformers import BertForMaskedLM  # Required for model class

################
## Example Input / Hidden States for Inspection
################

# Load tokenizer and temporary model for demonstration
_, tokenizer = get_BertMaskedLM_BertTokenizer_MLM()
 
# Inspect tokenization of a sample sentence
tokens = tokenize_sentence_MLM(sentence="The cat is a fatty patty.", tokenizer=tokenizer)
print("Tokenized sentence:", tokens)

################
## Data Preparation for k-fold
################

# Load dataset - ONLY using 'train' split for k-fold
dataset = load_dataset("imdb")

# Prepare k-fold data loaders (5 folds by default)
folds = prepare_data_kfold_MLM(
    data=dataset,
    tokenizer=tokenizer,
    n_splits=5  # Number of folds
)

################
## MLM Fine-tuning with k-fold cross-validation
################

# Train using k-fold CV (creates fresh model for each fold)
results = train_model_mlm_kfold_MLM(
    model_class=BertForMaskedLM,  # Pass model class, not instance
    folds=folds,
    epochs=3,
    lr=2e-5,
    n_splits=5
)

print("\n" + "="*50)
print("K-FOLD CROSS-VALIDATION COMPLETE")
print("="*50)
print(f"Overall Mean Validation Accuracy: {results['overall']['mean_val_accuracy']:.4f}")
print(f"Overall Mean Perplexity: {results['overall']['mean_perplexity']:.4f}")