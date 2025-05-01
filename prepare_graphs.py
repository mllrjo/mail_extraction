# File: prepare_graphs.py

import os
import json
from utils import load_graphs_by_domain

GNN_DATA_DIR = "gnn_data"
DOMAIN = "restaurant"
LABEL_MAP_PATH = "label_map.json"


def infer_label_map_from_graphs(graphs):
    # Extract all field names present in graph field_labels
    field_names = sorted({field for g in graphs for field in getattr(g, 'field_labels', {}).keys()})
    return {field: idx for idx, field in enumerate(field_names)}


def main():
    print("🔍 Loading graphs to infer label map...")
    graphs = load_graphs_by_domain(GNN_DATA_DIR, DOMAIN)

    label2idx = infer_label_map_from_graphs(graphs)
    print(f"✅ Inferred label2idx: {label2idx}")

    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(label2idx, f, indent=2)
        print(f"💾 Saved label map to {LABEL_MAP_PATH}")


if __name__ == "__main__":
    main()

