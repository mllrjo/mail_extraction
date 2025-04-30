#!/bin/bash
echo "📊 Labeled Node Counts Per File:"
for file in gnn_data/*.pt; do
    count=$(python3 -c "
import torch
data = torch.load('$file')
print(sum((data.y != -1).tolist()))
    ")
    echo "$(basename "$file"): $count labeled nodes"
done

