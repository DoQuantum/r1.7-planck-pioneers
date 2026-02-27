import torch
from datasets import load_dataset
from transformers import AutoTokenizer


def prepare_data():
    # Half-full IMDB: 12,500 train, 5,000 test
    TRAIN_SIZE = 12500
    TEST_SIZE = 5000

    print(f"Loading IMDB dataset ({TRAIN_SIZE} train, {TEST_SIZE} test)...")
    dataset = load_dataset("imdb")

    train_data = dataset["train"].shuffle(seed=42).select(range(TRAIN_SIZE))
    test_data = dataset["test"].shuffle(seed=42).select(range(TEST_SIZE))

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
