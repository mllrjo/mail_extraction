# File: models/tinygnn_with_tags.py

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import json

class TinyGNNWithTags(torch.nn.Module):
    def __init__(self, input_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, 32)
        self.conv2 = GCNConv(32, out_dim)
        self.out_dim = out_dim

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # 🛡️ Sanity check: guard against invalid label indices during training
        if hasattr(data, 'y') and data.y.numel() > 0 and data.y.max() >= self.out_dim:
            msg = f"⚠️ Label {data.y.max().item()} out of bounds for output dim {self.out_dim}"
            with open("label_error.log", "a") as log:
                log.write(msg + "\n")

            # Detailed debug: write all labels and label map if available
            with open("label_debug.log", "a") as dbg:
                dbg.write("======== DEBUG INFO ========\n")
                dbg.write(f"Max label: {data.y.max().item()}, Out dim: {self.out_dim}\n")
                dbg.write(f"All labels: {data.y.tolist()}\n")
                if hasattr(data, 'label2idx'):
                    dbg.write(f"label2idx: {data.label2idx}\n")
                if hasattr(data, 'x'):
                    dbg.write(f"Input feature shape: {data.x.shape}\n")
                if hasattr(data, 'tag_ids'):
                    dbg.write(f"Tag IDs: {data.tag_ids.tolist()[:10]} ... (truncated)\n")
                dbg.write("===========================\n\n")

            raise ValueError("Invalid label index detected in batch.")

        # 🩹 Mask out unlabeled nodes
        mask = data.y >= 0
        assert mask.sum() > 0, "No labeled nodes in this batch!"

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

    def save_label_map(self, label2idx, path="label_map.json"):
        with open(path, "w") as f:
            json.dump(label2idx, f)

    @staticmethod
    def load_label_map(path="label_map.json"):
        with open(path, "r") as f:
            label2idx = json.load(f)
        idx2label = {v: k for k, v in label2idx.items()}
        return label2idx, idx2label

