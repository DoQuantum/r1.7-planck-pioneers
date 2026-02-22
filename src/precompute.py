import torch
from torch.utils.data import DataLoader
from transformers import AutoModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def precompute():
    print("Loading base TinyBERT model...")
    bert = AutoModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D").to(DEVICE)
    bert.eval()

    # FIX: added weights_only=False
    try:
        train_data = torch.load("train_subset.pt", weights_only=False)
    except FileNotFoundError:
        print("Error: train_subset.pt not found. Run data_prep.py first!")
        return

    loader = DataLoader(train_data, batch_size=32, shuffle=False)

    all_embeddings = []
    all_labels = []

    print("Extracting features (this may take a minute)...")
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            outputs = bert(input_ids, attention_mask=attention_mask)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]

            all_embeddings.append(cls_embeddings.cpu())
            all_labels.append(batch["label"].cpu())

    final_embeddings = torch.cat(all_embeddings, dim=0)
    final_labels = torch.cat(all_labels, dim=0)

    torch.save(
        {"features": final_embeddings, "labels": final_labels}, "precomputed_train.pt"
    )
    print(
        f"Success! Saved {final_embeddings.shape[0]} feature vectors to precomputed_train.pt"
    )


if __name__ == "__main__":
    precompute()
