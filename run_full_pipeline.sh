#!/bin/bash
# File: run_full_pipeline.sh
# Description: End-to-end pipeline for TinyGNN + baselines using SWDE

set -e

echo "🔄 Step 0: Curating balanced subset..."
python curate_swde_balanced.py

export SWDE_INDEX=swde_index_balanced.json

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

