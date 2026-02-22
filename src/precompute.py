import torch
from torch.utils.data import DataLoader
from transformers import AutoModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def process_file(model, filename, output_name):
    print(f"--- Processing {filename} ---")
    try:
        data = torch.load(filename, weights_only=False)
    except FileNotFoundError:
        print(f"Error: {filename} not found. Skipping.")
        return

    loader = DataLoader(data, batch_size=32, shuffle=False)
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            outputs = model(input_ids, attention_mask=attention_mask)
            # Extract [CLS] token
            cls_embeddings = outputs.last_hidden_state[:, 0, :]

            all_embeddings.append(cls_embeddings.cpu())
            all_labels.append(batch["label"].cpu())

    final_features = torch.cat(all_embeddings, dim=0)
    final_labels = torch.cat(all_labels, dim=0)

    torch.save({"features": final_features, "labels": final_labels}, output_name)
    print(f"Saved to {output_name} (Shape: {final_features.shape})")


def precompute_all():
    print("Loading base TinyBERT model...")
    bert = AutoModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D").to(DEVICE)
    bert.eval()

    # Process Training Set
    process_file(bert, "train_subset.pt", "precomputed_train.pt")

    # Process Testing Set
    process_file(bert, "test_subset.pt", "precomputed_test.pt")

    print("\nAll features extracted successfully.")


if __name__ == "__main__":
    precompute_all()
