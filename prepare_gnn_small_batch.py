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
INDEX_NAME = os.getenv("SWDE_INDEX", "swde_index.json")
INDEX_FILE = os.path.join(PROJECT_DIR, INDEX_NAME)
SAVE_DIR = os.path.join(PROJECT_DIR, "gnn_data")
GT_BASE = os.path.join(SWDE_ROOT, "html_data", "groundtruth", "groundtruth")

MAX_FILES = 5  # Small batch for debugging

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

        try:
            with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
                html_content = f.read()
        except Exception as e:
            print(f"⚠️ Error reading HTML {html_path}: {e}")
            continue

        node_feats, edges, node_texts, tag_names = extract_dom_features(html_content)

        label2idx = {field: i for i, field in enumerate(fields)}
        y_indices = [-1] * len(node_feats)

        for field in fields:
            if field not in item["groundtruth"]:
                continue

            gt_value = item["groundtruth"][field]
            if not gt_value:
                continue

            gt_value = html.unescape(gt_value).strip().lower()

            for idx, text in enumerate(node_texts):
                if not text:
                    continue
                text_norm = text.lower().replace(" ", "")
                gt_norm = gt_value.replace(" ", "")
                if gt_norm in text_norm or text_norm in gt_norm:
                    print(f"🕵️ Field '{field}' → GT: '{gt_value}' | MATCHED: '{text}'")
                    y_indices[idx] = label2idx[field]
                    break

        label_counts = {}
        for label in y_indices:
            if label != -1:
                label_name = fields[label]
                label_counts[label_name] = label_counts.get(label_name, 0) + 1
        print(f"Labels for page {item['pageID']}: {label_counts}")

        if len(node_feats) == 0:
            print(f"⚠️ Skipping page {item['pageID']} due to empty node_feats")
            continue

        graph = make_graph(node_feats, edges, y_indices, domain=item["domain"], tags=tag_names)
        print("Tag IDs present:", hasattr(graph, "tag_ids"), graph.tag_ids.shape if hasattr(graph, "tag_ids") else "N/A")

        save_path = os.path.join(SAVE_DIR, f"sample_{item['pageID']}.pt")
        torch.save(graph, save_path)

    print(f"✅ Saved graphs to {SAVE_DIR}")

# === MAIN ENTRY ===
if __name__ == "__main__":
    main()

