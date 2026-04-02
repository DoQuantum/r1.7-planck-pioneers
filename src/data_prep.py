import sys

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


def prepare_data(train_size=12500, test_size=5000):
    print(f"Loading IMDB dataset ({train_size} train, {test_size} test)...")
    dataset = load_dataset("imdb")

    train_data = dataset["train"].shuffle(seed=42).select(range(train_size))
    test_data = dataset["test"].shuffle(seed=42).select(range(test_size))

    tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=128
        )

    tokenized_train = train_data.map(tokenize_function, batched=True)
    tokenized_test = test_data.map(tokenize_function, batched=True)

    tokenized_train.set_format(
        type="torch", columns=["input_ids", "attention_mask", "label"]
    )
    tokenized_test.set_format(
        type="torch", columns=["input_ids", "attention_mask", "label"]
    )

    torch.save(tokenized_train, "train_subset.pt")
    torch.save(tokenized_test, "test_subset.pt")
    print("Data saved to .pt files!")


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        train_size = int(sys.argv[1])
        test_size = int(sys.argv[2])
    else:
        train_size = 12500
        test_size = 5000
    prepare_data(train_size, test_size)
