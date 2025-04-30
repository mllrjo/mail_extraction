#!/bin/bash
# File: run_pipeline.sh
# Usage: bash run_pipeline.sh [curated|full]
# Default: full

MODE=${1:-full}

if [ "$MODE" = "curated" ]; then
    export SWDE_INDEX=swde_index_curated.json
    echo "🔄 Running pipeline with curated SWDE index: $SWDE_INDEX"
else
    unset SWDE_INDEX
    echo "🔄 Running pipeline with full SWDE index."
fi

# Call your existing full pipeline runner
./run_full_pipeline.sh

