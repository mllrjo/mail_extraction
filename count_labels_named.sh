#!/bin/bash

INDEX_JSON="swde_index_curated.json"
GNN_DIR="gnn_data"

# Extract unique label names from index safely
LABEL_LIST=$(python3 -c "
import json
with open('$INDEX_JSON') as f:
    index = json.load(f)
fields = sorted(set(index[0]['groundtruth'].keys()))
print(fields)
")

echo "📊 Labeled Node Counts Per File (with Label Names):"
for file in "$GNN_DIR"/*.pt; do
    echo "$(basename "$file"):"
    python3 -c "
import torch
from collections import Counter
label_names = $LABEL_LIST
data = torch.load('$file')
if hasattr(data, 'y'):
    y = data.y.tolist()
    counter = Counter([label for label in y if label != -1])
    for label in sorted(counter):
        name = label_names[label] if label < len(label_names) else f'Unknown({label})'
        print(f'  {name}: {counter[label]} nodes')
else:
    print('  No labels found')
"
done

