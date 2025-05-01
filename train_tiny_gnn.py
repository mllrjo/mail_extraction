# File: train_tiny_gnn.py

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from models.tinygnn_with_tags import TinyGNNWithTags
from utils import load_graphs_by_domain, infer_label_map_from_graphs
import os
import json

GNN_DATA_DIR = "gnn_data"
DOMAIN = "restaurant"
MODEL_PATH = "models/tinygnn_full.pth"
LABEL_MAP_PATH = "label_map.json"


def train():
    # 🔹 Load training data
    graphs = load_graphs_by_domain(GNN_DATA_DIR, DOMAIN)
    loader = DataLoader(graphs, batch_size=1, shuffle=True)

    # 🔹 Infer label map from data
    label2idx = infer_label_map_from_graphs(graphs)
    idx2label = {v: k for k, v in label2idx.items()}
    out_dim = len(label2idx)

    # 🔹 Save label map to disk
    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(label2idx, f, indent=2)

    # 🔹 Get input dim from sample
    input_dim = graphs[0].x.shape[1]

    # 🔹 Initialize model
    model = TinyGNNWithTags(input_dim=input_dim, out_dim=out_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print("🚀 Starting training...")
    for epoch in range(1, 51):
        model.train()
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()

            # 🧪 Sanity check: skip invalid batches
            if batch.y.numel() > 0 and batch.y.max() >= out_dim:
                print(f"⚠️ Skipping batch with invalid label index: max label = {batch.y.max().item()}, out_dim = {out_dim}")
                continue

            out = model(batch)
            mask = batch.y >= 0
            loss = F.cross_entropy(out[mask], batch.y[mask])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"🧪 Epoch {epoch:02d}: Loss = {total_loss:.4f}")

    # 🔹 Save model checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "out_dim": out_dim
    }, MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()

