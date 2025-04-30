import os
import json
from collections import defaultdict, Counter

# Config
INDEX_FILE = os.getenv("SWDE_INDEX", "swde_index.json")

# Load
with open(INDEX_FILE, "r", encoding="utf-8") as f:
    index = json.load(f)

field_values = defaultdict(set)
field_counts = Counter()

for page in index:
    for field, value in page.get("groundtruth", {}).items():
        if value:
            norm_value = value.strip().lower()
            field_values[field].add(norm_value)
            field_counts[field] += 1

print(f"\n📊 Label Diversity Summary ({INDEX_FILE}):")
for field in sorted(field_values):
    print(f"  {field:10} → {len(field_values[field])} unique / {field_counts[field]} total values")

