# File: prepare_gnn_small_batch.py
# Description: Prepares GNN training data from SWDE index, with dynamic domain + fields.

import os
import json
import torch
import html
from tqdm import tqdm
from utils import extract_dom_features, make_graph, infer_domain_fields

# === CONFIG ===
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SWDE_ROOT = os.path.abspath(os.path.join(PROJECT_DIR, "..", "swde"))
INDEX_FILE = os.path.join(PROJECT_DIR, "swde_index.json")
SAVE_DIR = os.path.join(PROJECT_DIR, "gnn_data")
GT_BASE = os.path.join(SWDE_ROOT, "html_data", "groundtruth", "groundtruth")

MAX_FILES = 10  # Small batch for debugging

os.makedirs(SAVE_DIR, exist_ok=True)

# === MAIN PREP FUNCTION ===
def main():
    print(f"📦 Loading SWDE index from {INDEX_FILE}")
    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        index = json.load(f)

    if not index:
        raise ValueError("SWDE index is empty!")

    # Detect domain from first example
    domain = index[0]["domain"]
    print(f"✅ Detected domain: {domain}")

    # Infer available fields
    domain_fields = infer_domain_fields(GT_BASE)
    fields = sorted(domain_fields.get(domain, []))
    print(f"✅ Using fields: {fields}")

    for item in tqdm(index[:MAX_FILES], desc="Preparing small batch"):
        html_path = os.path.join(SWDE_ROOT, item["html"])
        if not os.path.exists(html_path):
            print(f"⚠️ HTML file missing: {html_path}")
            continue

        # Extract features from HTML
        try:
            with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
                html_content = f.read()
        except Exception as e:
            print(f"⚠️ Error reading HTML {html_path}: {e}")
            continue

        node_feats, edges, node_texts = extract_dom_features(html_content)

        # Build label mapping for this page
        label2idx = {field: i for i, field in enumerate(fields)}
        y_indices = [-1] * len(node_feats)

        for field in fields:
            if field not in item["groundtruth"]:
                continue  # Skip if this field missing for this page

            gt_value = item["groundtruth"][field]
            if not gt_value:
                continue  # Skip empty groundtruth

            gt_value = html.unescape(gt_value).strip().lower()

            # Try to find a matching node
            for idx, text in enumerate(node_texts):
                if not text:
                    continue
                if gt_value in text.lower():
                    y_indices[idx] = label2idx[field]
                    break

        # Save graph
        graph = make_graph(node_feats, edges, y_indices, domain=item["domain"])
        save_path = os.path.join(SAVE_DIR, f"sample_{item['pageID']}.pt")
        torch.save(graph, save_path)

    print(f"✅ Saved graphs to {SAVE_DIR}")

# === MAIN ENTRY ===
if __name__ == "__main__":
    main()

