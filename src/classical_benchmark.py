from functions import tokenize_sentence, get_BertMaskedLM_BertTokenizer, prepare_data, BertWithLastLayerAttentionClassifier, train_model_mlm
from datasets import load_dataset
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


################
## Input: hidden_states (batch_size, seq_len, hidden_dim)
## Output: updated_hidden_states (same shape)
################

# Get Model and Tokenizer
model, tokenizer = get_BertMaskedLM_BertTokenizer()

# Show tokenized sentence
tokenize_sentence(sentence="The cat is a fatty patty.", tokenizer=tokenizer)

# Show model only uses masked language modeling head
print(model)






# Get Base Model
model, tokenizer = get_BertMaskedLM_BertTokenizer()
# Get Data to Finetune On
dataset = load_dataset("imdb")
# Prepare Data for Model
train_loader, test_loader = prepare_data(dataset, tokenizer)
# Finetune Model on Data
# Measure Accuracy and Efficiency
train_model_mlm(model, train_loader, test_loader, epochs=3)
# Replace Attention Mechanism with New Attention Mechanism
# Measure Accuracy and Efficiency Again
# Compare Results



# Accuracy (predicts the text): Perplexity / Masked Language Model Accuracy 


# Efficency: FLOPS / memory / time















































































































