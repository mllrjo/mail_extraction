#!/bin/bash

# Usage: ./run_pipeline_mode.sh [full|curated]
MODE=${1:-full}
LOGFILE="run_log.txt"
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

if [[ "$MODE" == "curated" ]]; then
    INDEX="swde_index_curated.json"
    echo "🔁 Running in CURATED mode..."
elif [[ "$MODE" == "full" ]]; then
    INDEX="swde_index.json"
    echo "🔁 Running in FULL mode..."
else
    echo "❌ Invalid mode: $MODE"
    echo "Usage: $0 [full|curated]"
    exit 1
fi

# Export the index env var so run_full_pipeline.sh picks it up
export SWDE_INDEX=$INDEX

# Run and tee output
{
    echo "=== [$TIMESTAMP] MODE: $MODE ==="
    ./run_full_pipeline.sh
    echo ""
} 2>&1 | tee -a "$LOGFILE"

