import torch
from datasets import load_dataset
from transformers import AutoTokenizer


def prepare_data(sample_size=1000):
    print(f"Loading IMDB dataset and taking {sample_size} samples...")
    dataset = load_dataset("imdb")

    # Shuffle and subset to save time
    train_data = dataset["train"].shuffle(seed=42).select(range(sample_size))
    test_data = dataset["test"].shuffle(seed=42).select(range(sample_size // 5))

    tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=128
        )

    tokenized_train = train_data.map(tokenize_function, batched=True)
    tokenized_test = test_data.map(tokenize_function, batched=True)

    # Set format for PyTorch
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
    prepare_data()
