from datasets import load_dataset
from transformers import BertTokenizerFast
import numpy as np
import matplotlib.pyplot as plt

# Load dataset and tokenizer
dataset = load_dataset("imdb")
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# Tokenize a sample (you can use a subset to save time)
def count_tokens(example):
    tokens = tokenizer(
        example["text"],
        truncation=False,  # don't cut them off
        add_special_tokens=True
    )
    return {"length": len(tokens["input_ids"])}

# Use a subset for speed (e.g., 5000 samples)
subset = dataset["train"].select(range(5000))
token_lengths = subset.map(count_tokens, num_proc=4)
lengths = np.array(token_lengths["length"])

# Show summary stats
print(f"Mean length: {lengths.mean():.2f}")
print(f"Median length: {np.median(lengths):.2f}")
print(f"95th percentile: {np.percentile(lengths, 95):.2f}")
print(f"99th percentile: {np.percentile(lengths, 99):.2f}")
print(f"Max length: {lengths.max()}")

# Plot histogram
plt.figure(figsize=(8,5))
plt.hist(lengths, bins=50, color="steelblue", alpha=0.7)
plt.xlabel("Token Length")
plt.ylabel("Number of Samples")
plt.title("IMDB Review Token Length Distribution")
plt.show()
