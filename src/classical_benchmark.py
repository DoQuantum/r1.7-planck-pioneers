from functions import (
    tokenize_sentence,
    prepare_data,
    BertWithLastLayerAttentionClassifier,
    train_model_classifier
)
from datasets import load_dataset
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import AutoModel, BertTokenizer

################
## Input: hidden_states (batch_size, seq_len, hidden_dim)
## Output: updated_hidden_states (same shape)
################
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using:", device)


# Load BERT base model and tokenizer
model_name = "bert-base-uncased"
model = AutoModel.from_pretrained(model_name, output_attentions=True)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Load IMDB sentiment dataset
dataset = load_dataset("imdb")

# Prepare tokenized data loaders
train_loader, test_loader = prepare_data(dataset, tokenizer)

# Build classifier on top of BERT
classifier = BertWithLastLayerAttentionClassifier(model, num_classes=2)

# Train classifier and evaluate accuracy
train_model_classifier(classifier, train_loader, test_loader, epochs=3)

# Replace Attention Mechanism with New Attention Mechanism
# Measure Accuracy and Efficiency Again
# Compare Results



# Accuracy (predicts the text): Perplexity / Masked Language Model Accuracy 


# Efficency: FLOPS / memory / time


