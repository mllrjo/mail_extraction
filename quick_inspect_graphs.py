# File: quick_inspect_graphs.py
# Description: Quick check of graphs in gnn_data/ for labeled nodes and domains.

import os
import torch

GRAPH_DIR = "gnn_data"

def inspect_graphs(graph_dir):
    graphs = [f for f in os.listdir(graph_dir) if f.endswith(".pt")]
    if not graphs:
        print("❌ No graphs found.")
        return

    print(f"🔍 Inspecting {len(graphs)} graphs...\n")

    for fname in sorted(graphs):
        path = os.path.join(graph_dir, fname)
        data = torch.load(path)
        num_nodes = data.x.shape[0]
        num_edges = data.edge_index.shape[1]
        num_labeled = (data.y != -1).sum().item()
        domain = getattr(data, "domain", "unknown")

        print(f"🗂 {fname}")
        print(f"    Domain: {domain}")
        print(f"    Nodes: {num_nodes}, Edges: {num_edges}")
        print(f"    Labeled nodes: {num_labeled}")
        print("")

if __name__ == "__main__":
    inspect_graphs(GRAPH_DIR)

