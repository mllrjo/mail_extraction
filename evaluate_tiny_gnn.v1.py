# File: evaluate_tiny_gnn.py

import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import classification_report, accuracy_score, f1_score
from models.tinygnn_with_tags import TinyGNNWithTags
from utils import load_graphs_by_domain
import json

MODEL_PATH = "models/tinygnn_full.pth"
DATA_DIR = "gnn_data"
DOMAIN = "restaurant"

def evaluate():
    checkpoint = torch.load(MODEL_PATH)

    input_dim = checkpoint["input_dim"]
    out_dim = checkpoint["out_dim"]

    model = TinyGNNWithTags(input_dim=input_dim, out_dim=out_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    label2idx, idx2label = TinyGNNWithTags.load_label_map()

    graphs = load_graphs_by_domain(DATA_DIR, DOMAIN)
    loader = DataLoader(graphs, batch_size=1)

    all_true = []
    all_pred = []

    with torch.no_grad():
        for batch in loader:
            out = model(batch)
            mask = batch.y >= 0
            preds = out[mask].argmax(dim=1)
            all_true.extend(batch.y[mask].tolist())
            all_pred.extend(preds.tolist())

    acc = accuracy_score(all_true, all_pred)
    f1 = f1_score(all_true, all_pred, average="weighted")

    print("\n📊 Evaluation Results:")
    print(f"✅ Accuracy: {acc * 100:.2f}%")
    print(f"🎯 Weighted F1 Score: {f1:.3f}\n")

    target_names = [idx2label[i] for i in sorted(idx2label)]
    print("📋 Classification Report:")
    print(classification_report(all_true, all_pred, target_names=target_names, labels=sorted(idx2label)))

if __name__ == "__main__":
    evaluate()

