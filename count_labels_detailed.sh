#!/bin/bash
echo "📊 Per-Label Labeled Node Counts Per File:"
for file in gnn_data/*.pt; do
    echo "$(basename "$file"):"
    python3 -c "
import torch
from collections import Counter
data = torch.load('$file')
if hasattr(data, 'y'):
    y = data.y.tolist()
    counter = Counter([label for label in y if label != -1])
    for label, count in sorted(counter.items()):
        print(f'  Label {label}: {count} nodes')
else:
    print('  No labels found')
"
done

