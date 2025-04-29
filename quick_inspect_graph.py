# File: quick_inspect_graph.py
# Description: Quick inspection of saved GNN graph .pt files

import sys
import torch

def inspect_graph(path):
    data = torch.load(path)
    print(f"✅ Loaded graph: {path}")
    print(f"  - Num nodes: {data.num_nodes}")
    print(f"  - Num edges: {data.num_edges}")
    print(f"  - Node feature dim: {data.x.shape[1] if hasattr(data, 'x') else 'N/A'}")

    if hasattr(data, 'y') and data.y is not None:
        labeled = (data.y != -1).sum().item()
        print(f"  - Labeled nodes: {labeled}")
        print(f"  - Label dimension: {data.y.shape if hasattr(data, 'y') else 'N/A'}")
    else:
        print("  - No labels found.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python quick_inspect_graph.py <path_to_pt_file>")
        sys.exit(1)

    path = sys.argv[1]
    inspect_graph(path)

