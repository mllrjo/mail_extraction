# File: baseline_tagonly.py
# Description: Simple tag-name-only baseline classifier

import os
import torch
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.multiclass import unique_labels
from torch_geometric.data import Data

# === CONFIG ===
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
GNN_DIR = os.path.join(PROJECT_DIR, "gnn_data")
INDEX_FILE = os.getenv("SWDE_INDEX", "swde_index_curated.json")
INDEX_FILE = os.path.join(PROJECT_DIR, INDEX_FILE)

def extract_tag_frequencies(graphs):
    tag2label_counts = {}
    label_counts = {}

    for data in graphs:
        if not hasattr(data, "tag_ids") or data.tag_ids.numel() == 0:
            continue
        tags = data.tag_ids.tolist()
        for i, tag in enumerate(tags):
            label = int(data.y[i])
            if label == -1:
                continue
            tag2label_counts.setdefault(tag, {})
            tag2label_counts[tag][label] = tag2label_counts[tag].get(label, 0) + 1
            label_counts[label] = label_counts.get(label, 0) + 1

    # Take the most frequent label for each tag
    tag2label = {}
    for tag, counts in tag2label_counts.items():
        tag2label[tag] = max(counts.items(), key=lambda x: x[1])[0]

    return tag2label, label_counts

def load_graphs():
    graphs = []
    for fname in os.listdir(GNN_DIR):
        if not fname.endswith(".pt"):
            continue
        path = os.path.join(GNN_DIR, fname)
        graph = torch.load(path)
        graphs.append(graph)
    return graphs

def main():
    print("🧪 Starting tag-only baseline...")

    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        index = json.load(f)
    if not index:
        raise ValueError("Index is empty")

    domain = index[0]["domain"]
    fields = sorted({field for item in index for field in item["groundtruth"].keys()})
    label2idx = {f: i for i, f in enumerate(fields)}
    idx2label = {v: k for k, v in label2idx.items()}

    graphs = load_graphs()
    train_graphs = graphs[:4]
    test_graphs = graphs[4:]

    tag2labels, label_counts = extract_tag_frequencies(train_graphs)

    print("✅ Tag frequency model ready")
    print(f"Tag2labels: {tag2labels}")

    all_true = []
    all_pred = []

    for data in test_graphs:
        if not hasattr(data, "tag_ids"):
            continue
        tags = data.tag_ids.tolist()
        for i, tag in enumerate(tags):
            true = int(data.y[i])
            if true == -1:
                continue
            pred = tag2labels.get(tag, -1)
            if pred != -1:
                all_true.append(true)
                all_pred.append(pred)

    if not all_true:
        print("⚠️ No labeled nodes found during evaluation.")
        return

    accuracy = accuracy_score(all_true, all_pred)
    f1 = f1_score(all_true, all_pred, average="weighted")
    print(f"✅ Accuracy: {accuracy:.2%}")
    print(f"🎯 Weighted F1 Score: {f1:.3f}")
    print("\n📋 Classification Report:")
    used_labels = sorted(set(all_true + all_pred))
    print(classification_report(
        all_true,
        all_pred,
        labels=used_labels,
        target_names=[idx2label[i] for i in used_labels]
    ))

if __name__ == "__main__":
    main()

