import os
import json
from collections import defaultdict

project_dir = os.getcwd()
index_file = os.path.join(project_dir, "swde_index.json")
curated_path = os.path.join(project_dir, "swde_index_balanced.json")

MIN_PER_FIELD = 5

# Load index
with open(index_file, "r", encoding="utf-8") as f:
    index = json.load(f)

field_counters = defaultdict(int)
selected_pages = []

for item in index:
    gt = item.get("groundtruth", {})
    added = False
    for field, value in gt.items():
        if value.strip() and field_counters[field] < MIN_PER_FIELD:
            field_counters[field] += 1
            added = True
    if added:
        selected_pages.append(item)

with open(curated_path, "w", encoding="utf-8") as f:
    json.dump(selected_pages, f, indent=2)

print(f"✅ Saved balanced subset to: {curated_path}")
print(f"📊 Per-field coverage: {dict(field_counters)}")

