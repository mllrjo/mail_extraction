# File: evaluate_tiny_gnn.py

import torch
from torch_geometric.loader import DataLoader
from models.tinygnn_with_tags import TinyGNNWithTags
from utils import load_graphs_by_domain
import json
from sklearn.metrics import classification_report
import string
import numpy as np

GNN_DATA_DIR = "gnn_data"
DOMAIN = "restaurant"
MODEL_PATH = "models/tinygnn_full.pth"
LABEL_MAP_PATH = "label_map.json"

# Character-level embedding setup
CHAR_INDEX = {c: i for i, c in enumerate(string.ascii_lowercase + string.digits + " .,;:-_")}
CHAR_DIM = len(CHAR_INDEX)

def text_to_char_vector(text, max_len=100):
    vec = np.zeros(CHAR_DIM, dtype=np.float32)
    for c in text.lower()[:max_len]:
        if c in CHAR_INDEX:
            vec[CHAR_INDEX[c]] += 1
    if vec.sum() > 0:
        vec /= vec.sum()
    return vec

def augment_graph_with_char_features(graph):
    # Dummy example: average over all field label strings
    if hasattr(graph, 'field_labels'):
        all_text = ' '.join(graph.field_labels.keys())
        char_vec = text_to_char_vector(all_text)
        # Expand to match number of nodes and concatenate to node features
        repeated = np.tile(char_vec, (graph.x.shape[0], 1))
        char_tensor = torch.tensor(repeated, dtype=torch.float32)
        graph.x = torch.cat([graph.x, char_tensor], dim=1)
    return graph

def evaluate():
    # 🔹 Load test data
    graphs = load_graphs_by_domain(GNN_DATA_DIR, DOMAIN)
    graphs = [augment_graph_with_char_features(g) for g in graphs]
    loader = DataLoader(graphs, batch_size=1, shuffle=False)

    # 🔹 Load label map
    with open(LABEL_MAP_PATH, "r") as f:
        label2idx = json.load(f)
    idx2label = {v: k for k, v in label2idx.items()}
    out_dim = len(label2idx)

    # 🔹 Log inferred label space with field names
    print("\n🔍 Label map loaded (field names):")
    for field_name, idx in sorted(label2idx.items(), key=lambda x: x[1]):
        print(f"  {field_name:12s} → {idx}")

    # 🔹 Get input dim from sample
    input_dim = graphs[0].x.shape[1]

    # 🔹 Load model
    model = TinyGNNWithTags(input_dim=input_dim, out_dim=out_dim)
    checkpoint = torch.load(MODEL_PATH)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_true = []
    all_pred = []

    for batch in loader:
        with torch.no_grad():
            # 🧪 Sanity check: skip invalid batches
            if batch.y.numel() > 0 and batch.y.max() >= out_dim:
                print(f"⚠️ Skipping batch with invalid label index: max label = {batch.y.max().item()}, out_dim = {out_dim}")
                continue

            out = model(batch)
            pred = out.argmax(dim=1)
            mask = batch.y >= 0
            all_true.extend(batch.y[mask].tolist())
            all_pred.extend(pred[mask].tolist())

    if all_true and all_pred:
        used_labels = sorted(set(all_true + all_pred))
        target_names = [idx2label[i] if i in idx2label else f"label_{i}" for i in used_labels]
        print("\n📋 Classification Report:")
        print(classification_report(all_true, all_pred, target_names=target_names, labels=used_labels))
    else:
        print("⚠️ No valid predictions available. Evaluation skipped due to label mismatch.")

if __name__ == "__main__":
    evaluate()

