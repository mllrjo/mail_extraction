# File: evaluate_tiny_gnn.py
# Description: Evaluates TinyGNN performance on all labeled nodes in gnn_data/.

import os
import torch
from torch_geometric.data import Data
from utils import TinyGNN
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

# === CONFIG ===
GNN_DIR = "gnn_data"
MODEL_PATH = "models/tinygnn_model.pth"

# === LOAD MODEL + METADATA ===
print(f"📦 Loading model from {MODEL_PATH}")
checkpoint = torch.load(MODEL_PATH, map_location="cpu")
model = TinyGNN(
    input_dim=checkpoint["input_dim"],
    hidden_dim=64,
    output_dim=checkpoint["output_dim"]
)
model.load_state_dict(checkpoint["state_dict"])
model.eval()

label2idx = checkpoint["label2idx"]
idx2label = {v: k for k, v in label2idx.items()}

# === EVALUATE ON ALL LABELED NODES ===
all_true = []
all_pred = []

print(f"🔍 Evaluating graphs in {GNN_DIR}")
for fname in tqdm(os.listdir(GNN_DIR)):
    if not fname.endswith(".pt"):
        continue
    path = os.path.join(GNN_DIR, fname)
    data = torch.load(path)

    if not hasattr(data, "y") or data.y is None:
        continue

    x, edge_index, y = data.x, data.edge_index, data.y
    mask = (y != -1)
    if mask.sum().item() == 0:
        continue  # no labels

    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)

    all_true.extend(y[mask].cpu().tolist())
    all_pred.extend(pred[mask].cpu().tolist())

# === METRICS ===
if not all_true:
    print("❌ No labeled nodes found for evaluation.")
else:
    acc = accuracy_score(all_true, all_pred)
    f1 = f1_score(all_true, all_pred, average="weighted")

    print("\n📊 Evaluation Results:")
    print(f"✅ Accuracy: {acc*100:.2f}%")
    print(f"🎯 Weighted F1 Score: {f1:.3f}")

    print("\n📋 Classification Report:")
    labels = sorted(idx2label)
    target_names = [idx2label[i] for i in labels]
    print(classification_report(all_true, all_pred, labels=labels, target_names=target_names, zero_division=0))

