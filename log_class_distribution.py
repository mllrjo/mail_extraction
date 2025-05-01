# File: log_class_distribution.py

from models.tinygnn_with_tags import TinyGNNWithTags
from utils import load_graphs_by_domain
import torch

GNN_DIR = "gnn_data"
DOMAIN = "restaurant"
LABEL_MAP_PATH = "label_map.json"

# 🔁 Load graphs and model
graphs = load_graphs_by_domain(GNN_DIR, DOMAIN)
label2idx, _ = TinyGNNWithTags.load_label_map(LABEL_MAP_PATH)
model = TinyGNNWithTags(input_dim=3, out_dim=len(label2idx))  # adjust if input dim differs
model.log_class_distribution(graphs, label_map_path=LABEL_MAP_PATH)

print("📊 Logged class distribution to class_distribution.log")

