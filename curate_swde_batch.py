# File: curate_swde_batch.py
# Description: Build a clean mini SWDE batch with well-labeled pages.

import os
import json
from collections import defaultdict

# === CONFIG ===
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_FILE = os.getenv("SWDE_INDEX", "swde_index.json")
CURATED_INDEX_FILE = os.path.join(PROJECT_DIR, "swde_index_curated.json")

# Set minimum number of labeled fields you require
MIN_FIELDS = 3

def has_enough_labels(record, min_fields):
    gt = record.get("groundtruth", {})
    # Count non-empty groundtruths
    return sum(bool(v.strip()) for v in gt.values()) >= min_fields

def main():
    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        index = json.load(f)

    curated = []
    for record in index:
        if has_enough_labels(record, MIN_FIELDS):
            curated.append(record)

    print(f"✅ Selected {len(curated)} pages with at least {MIN_FIELDS} labeled fields.")

    with open(CURATED_INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(curated, f, indent=2)

    print(f"✅ Saved curated index to {CURATED_INDEX_FILE}")

if __name__ == "__main__":
    main()

