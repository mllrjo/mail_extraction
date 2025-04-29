# File: validate_swde_index.py
# Description: Quickly validate that swde_index.json has healthy groundtruth coverage.

import os
import json
from collections import defaultdict

# === CONFIG ===
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_FILE = os.path.join(PROJECT_DIR, "swde_index.json")

# === VALIDATOR ===
def validate_swde_index(index_file):
    if not os.path.exists(index_file):
        print(f"❌ Index file not found: {index_file}")
        return

    with open(index_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_pages = len(data)
    domain_counts = defaultdict(lambda: {"total": 0, "labeled": 0})

    for record in data:
        domain = record.get("domain", "unknown")
        domain_counts[domain]["total"] += 1

        groundtruth = record.get("groundtruth", {})
        if any(val.strip() for val in groundtruth.values()):
            domain_counts[domain]["labeled"] += 1

    print(f"\n📋 Validation Report for {index_file}")
    print("-------------------------------------------------")
    print(f"Total Pages Indexed: {total_pages}\n")

    for domain, counts in sorted(domain_counts.items()):
        total = counts["total"]
        labeled = counts["labeled"]
        percent = (labeled / total) * 100 if total > 0 else 0.0
        print(f"Domain: {domain}")
        print(f"    Pages Indexed: {total}")
        print(f"    Pages with Groundtruth: {labeled} ({percent:.1f}%)")

        if percent < 70.0:
            print(f"    ⚠️ Warning: Only {percent:.1f}% of pages have labels!")
        print("")

    print("✅ Validation complete.")

# === MAIN ENTRY ===
if __name__ == "__main__":
    validate_swde_index(INDEX_FILE)

