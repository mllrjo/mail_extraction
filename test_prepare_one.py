# File: test_prepare_one.py
# Description: Mini harness to test GNN graph extraction on one SWDE HTML + groundtruth pair

import os
import html
import torch
from utils import extract_dom_features, make_graph, build_label_mapping
my_root='/Users/jonathanmiller/Desktop/Dev/structHTML/ClaudeAPI/'

# === Config ===
HTML_PATH = my_root + "swde/html_data/book/book/book-barnesandnoble(2000)/0000.htm"
GT_PATH = my_root +"swde/html_data/groundtruth/groundtruth/book/book-barnesandnoble-title.txt"
PAGE_ID = "0000"
SAVE_PATH = "gnn_data/test_0000.pt"

LABEL_FIELDS = ["title"]
label2idx, idx2label = build_label_mapping(LABEL_FIELDS)

def load_groundtruth(filepath):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            if parts[0].strip() == PAGE_ID:
                return [parts[2].strip()]
    return []
import os
import html

def normalize_text(text):
    if not text:
        return ""
    text = html.unescape(text)
    text = text.lower()
    text = text.replace(" ", "").replace("\xa0", "")
    return text

def label_nodes(nodes, labels, label2idx):
    debug = os.environ.get("DEBUG_MATCH") == "1"
    y = [-1] * len(nodes)

    for label_text, field in labels:
        label_norm = normalize_text(label_text)
        matched = False
        for i, node in enumerate(nodes):
            node_text = normalize_text(node.get("text", ""))
            if label_norm in node_text:
                y[i] = label2idx[field]
                matched = True
                if debug:
                    print(f"[MATCH] Label '{label_text}' matched node {i} (text='{node.get('text', '')}')")
                break
        if not matched and debug:
            print(f"[NO MATCH] Label '{label_text}' did not match any node.")
    return y

def main():
    os.makedirs("gnn_data", exist_ok=True)

    # Read HTML
    with open(HTML_PATH, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()

    # Extract DOM features
    nodes, edges = extract_dom_features(html)

    # Read groundtruth labels
    gt_labels = load_groundtruth(GT_PATH)
    labeled = [(val, "title") for val in gt_labels]

    # Assign node labels
    y_indices = label_nodes(nodes, labeled, label2idx)

    # Build and save graph
    graph = make_graph(nodes, edges, y_indices)
    torch.save(graph, SAVE_PATH)

    # Report
    print(f"✅ Saved test graph to {SAVE_PATH}")
    print(f"  - Nodes: {graph.num_nodes}")
    print(f"  - Edges: {graph.num_edges}")
    labeled_nodes = (torch.tensor(y_indices) != -1).sum().item()
    print(f"  - Labeled nodes: {labeled_nodes}")

if __name__ == "__main__":
    main()

