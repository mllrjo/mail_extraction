#!/bin/bash
# File: run_full_pipeline.sh
# Description: Runs full SWDE → GNN pipeline cleanly.

set -e  # Exit immediately if any command fails

echo "🔄 Step 1: Rebuilding swde_index.json..."
python swde_indexer.py

echo "🔄 Step 2: Preparing small batch of GNN graphs..."
python prepare_gnn_small_batch.py

echo "🔄 Step 3: Validating index coverage..."
python validate_swde_index.py

echo "🔄 Step 4: Training TinyGNN model..."
python train_tiny_gnn.py

echo "🔄 Step 5: Evaluating TinyGNN model..."
python evaluate_tiny_gnn.py

echo "✅ Full pipeline completed successfully!"

