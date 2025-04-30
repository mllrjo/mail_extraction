# File: utils.py
# Description: Modular utilities for TinyGNN + SWDE pipeline
# Supports isolated test execution via env variables.

import torch
import torch.nn as nn
import torch.nn.functional as F
from bs4 import BeautifulSoup
import os
import re
import gzip
from torch_geometric.data import Data, Dataset

#####################################
from collections import defaultdict

def infer_domain_fields(gt_base):
    domain_fields = defaultdict(set)

    for domain in os.listdir(gt_base):
        domain_path = os.path.join(gt_base, domain)
        if not os.path.isdir(domain_path):
            continue

        for filename in os.listdir(domain_path):
            if not filename.endswith(".txt"):
                continue

            match = re.match(rf"{domain}-(.*?)-(.*)\.txt", filename)
            if match:
                website, field = match.group(1), match.group(2)
                domain_fields[domain].add(field)

    return domain_fields
########################################
# 1. Label Mapping Utilities
########################################

def build_label_mapping(groundtruth_fields):
    label2idx = {label: idx for idx, label in enumerate(sorted(set(groundtruth_fields)))}
    idx2label = {idx: label for label, idx in label2idx.items()}
    return label2idx, idx2label

if __name__ == "__main__" and os.environ.get("TEST_LABEL_MAPPING") == "1":
    test_labels = ["title", "author", "price", "author"]
    l2i, i2l = build_label_mapping(test_labels)
    assert i2l[l2i["price"]] == "price"
    print("[PASS] Label mapping test")

########################################
# 2. DOM Feature Extraction
########################################

def extract_dom_features(html_content):
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html_content, "html.parser")
    node_feats = []
    edges = []
    node_texts = []
    idx_counter = [0]
    parent_stack = []
    tag_names = []  # ✅ New: store HTML tag names

    def recurse(node, parent_idx=None):
        if isinstance(node, str):
            return

        tag_names.append(node.name if hasattr(node, "name") else "text")

        idx = idx_counter[0]
        idx_counter[0] += 1

        # Features
        tag = node.name or "text"
        depth = len(parent_stack)
        text_len = len(node.get_text(strip=True)) if hasattr(node, "get_text") else 0
        has_href = int(bool(node.get("href"))) if hasattr(node, "get") else 0

        feats = [depth, text_len, has_href]
        node_feats.append(feats)
        node_texts.append(node.get_text(strip=True) if hasattr(node, "get_text") else "")

        if parent_idx is not None:
            edges.append([parent_idx, idx])

        parent_stack.append(node)
        for child in getattr(node, "children", []):
            recurse(child, idx)
        parent_stack.pop()
       
    recurse(soup.body or soup)

    # ✅ Sanity check (optional)
    assert len(tag_names) == len(node_feats), f"Mismatch: {len(tag_names)} tags vs {len(node_feats)} nodes"
    if len(tag_names) != len(node_feats):
        print(f"⚠️ Tag mismatch: {len(tag_names)} tags vs {len(node_feats)} nodes")


    return node_feats, edges, node_texts, tag_names


if __name__ == "__main__" and os.environ.get("TEST_DOM_FEATURES") == "1":
    sample_html = "<html><body><div><a href='link'>Text</a></div></body></html>"
    feats, edges = extract_dom_features(sample_html)
    assert any(f["has_href"] == 1 for f in feats)
    assert len(edges) > 0
    print("[PASS] DOM feature extraction test")

########################################
# 3. Graph Construction
########################################
from torch_geometric.data import Data

def make_graph(node_feats, edges, labels, domain=None, tags=None):

    x = torch.tensor(node_feats, dtype=torch.float)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    y = torch.tensor(labels, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)

    if domain is not None:
        data.domain = domain

    if tags is not None and len(tags) == len(node_feats):
        tag_tensor = torch.tensor([hash(tag) % (2**16) for tag in tags], dtype=torch.long)
        data.tag_ids = tag_tensor

    return data


if __name__ == "__main__" and os.environ.get("TEST_GRAPH_MAKE") == "1":
    feats = [{"depth": 1, "text_len": 4, "has_href": 1},
             {"depth": 2, "text_len": 0, "has_href": 0}]
    edges = [(0, 1)]
    g = make_graph(feats, edges, [0, 1])
    assert g.x.shape[1] == 3
    assert g.edge_index.shape[1] == 1
    print("[PASS] Graph construction test")

########################################
# 4. TinyGNN Model
########################################
import torch.nn as nn
from torch_geometric.nn import GCNConv

class TinyGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        return x

if __name__ == "__main__" and os.environ.get("TEST_TINY_GNN") == "1":
    dummy_data = Data(
        x=torch.rand(5, 3),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    )
    model = TinyGNN(input_dim=3, hidden_dim=4, output_dim=2)
    out = model(dummy_data)
    assert out.shape == (5, 2)
    print("[PASS] TinyGNN test")

########################################
# 5. GNNDataset Loader (supports .pt and .pt.gz)
########################################
from torch_geometric.data import Dataset
import os
import torch

class GNNDataset(Dataset):
    def __init__(self, graph_dir):
        super().__init__()
        self.graph_files = sorted([
            os.path.join(graph_dir, f) for f in os.listdir(graph_dir) if f.endswith(".pt")
        ])

    def len(self):
        return len(self.graph_files)

    def get(self, idx):
        data = torch.load(self.graph_files[idx])

        # ❗ Strip non-tensor fields before batching
        if hasattr(data, 'domain'):
            del data.domain

        return data


if __name__ == "__main__" and os.environ.get("TEST_DATASET_GZ") == "1":
    import tempfile
    sample_data = Data(x=torch.rand(2, 3), edge_index=torch.tensor([[0, 1], [1, 0]]))

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = os.path.join(tmpdir, "sample.pt")
        gz_path = os.path.join(tmpdir, "sample.pt.gz")

        torch.save(sample_data, raw_path)
        with gzip.open(gz_path, "wb") as f:
            torch.save(sample_data, f)

        ds = GNNDataset(tmpdir)
        d1 = ds.get(0)
        d2 = ds.get(1)
        assert isinstance(d1, Data) and isinstance(d2, Data)
        print("[PASS] Dataset loading (.pt + .pt.gz) test")

if __name__ == "__main__" and os.environ.get("TEST_DATASET") == "1":
    import tempfile
    sample_data = Data(x=torch.rand(2, 3), edge_index=torch.tensor([[0, 1], [1, 0]]))

    with tempfile.TemporaryDirectory() as tmpdir:
        torch.save(sample_data, os.path.join(tmpdir, "sample.pt"))
        ds = GNNDataset(tmpdir)
        d = ds.get(0)
        assert isinstance(d, Data)
        print("[PASS] Dataset loading (.pt only) test")

