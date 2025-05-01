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

        # 🔡️ Sanity check: guard against invalid label indices during training
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

        # 🦩 Mask out unlabeled nodes
        mask = data.y >= 0
        assert mask.sum() > 0, "No labeled nodes in this batch!"

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

    def save_label_map(self, label2idx, path="label_map.json"):
        # ✅ Patch: use raw field names as keys
        json.dump(label2idx, open(path, "w"), indent=2)

    @staticmethod
    def load_label_map(path="label_map.json"):
        with open(path, "r") as f:
            label2idx = json.load(f)
        idx2label = {v: k for k, v in label2idx.items()}
        return label2idx, idx2label

    def get_class_counts(self, dataset):
        """Returns a dictionary of class index → count from the dataset."""
        from collections import Counter
        counts = Counter()
        for data in dataset:
            labels = data.y[data.y >= 0].tolist()
            counts.update(labels)
        return dict(counts)

    def log_class_distribution(self, dataset, label_map_path="label_map.json"):
        """Logs a human-readable breakdown of class distributions using label_map."""
        class_counts = self.get_class_counts(dataset)
        label2idx, idx2label = self.load_label_map(label_map_path)
        with open("class_distribution.log", "w") as f:
            for idx in sorted(class_counts):
                # Map back from index to readable label
                label = idx2label.get(idx, f"class_{idx}")
                count = class_counts[idx]
                f.write(f"{label:12s} → {count} nodes\n")

    def load_and_log_class_distribution(self, gnn_dir, domain):
        from utils import load_graphs_by_domain
        graphs = load_graphs_by_domain(gnn_dir, domain)
        self.log_class_distribution(graphs)

        # Log confirmation to stdout
        print("📊 Class distribution logged to class_distribution.log")

    def evaluate_and_log_distribution(self, gnn_dir, domain):
        """Call during evaluation to log class distribution."""
        print("🧮 Logging class distribution during evaluation...")
        self.load_and_log_class_distribution(gnn_dir, domain)

    def predict_and_report(self, all_true, all_pred, label_map_path="label_map.json"):
        """Generate and print a classification report with readable labels."""
        from sklearn.metrics import classification_report
        label2idx, idx2label = self.load_label_map(label_map_path)
        target_names = [idx2label[i] for i in sorted(set(all_true + all_pred))]
        print("\n📋 Classification Report:")
        print(classification_report(all_true, all_pred, target_names=target_names))

