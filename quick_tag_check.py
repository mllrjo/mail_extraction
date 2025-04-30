# quick_tag_check.py
import os
import torch

GNN_DIR = "gnn_data"

print("🔍 Checking tag_ids in all .pt files...")
for fname in sorted(os.listdir(GNN_DIR)):
    if not fname.endswith(".pt"):
        continue
    path = os.path.join(GNN_DIR, fname)
    data = torch.load(path)
    ok = hasattr(data, "tag_ids") and data.tag_ids.shape[0] == data.x.shape[0]
    print(f"{fname:20} tag_ids OK? {ok}")

