#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# run_finetune.sh — End-to-end LoRA fine-tuning pipeline
#
# Steps:
#   1. Export positive interactions from SQLite → JSONL
#   2. Train LoRA adapter on the exported data
#   3. (Optional) Merge adapter into base model
#
# This script is designed to run OUTSIDE the API process
# (background job, cron, manual trigger) so it never blocks
# the FastAPI event loop.
#
# Usage:
#   bash finetune/run_finetune.sh
#
# Environment variables:
#   MLML_DB_PATH — SQLite database path (default: data/mlml.db)
#   MAX_STEPS    — Training steps (default: 200)
#   MERGE        — Set to "1" to merge adapter after training
# ──────────────────────────────────────────────────────────────
set -euo pipefail

DB_PATH="${MLML_DB_PATH:-data/mlml.db}"
MAX_STEPS="${MAX_STEPS:-200}"
MERGE="${MERGE:-0}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
DATA_PATH="finetune/data/train_${TIMESTAMP}.jsonl"
ADAPTER_PATH="adapters/v_${TIMESTAMP}"
MERGED_PATH="models/merged_${TIMESTAMP}"

echo "=== Step 1: Export positive interactions ==="
python -m finetune.prepare_data \
    --db-path "${DB_PATH}" \
    --output "${DATA_PATH}" \
    --min-feedback 1

# Check if export produced any data
if [ ! -s "${DATA_PATH}" ]; then
    echo "No training data exported. Aborting."
    exit 0
fi

echo ""
echo "=== Step 2: LoRA fine-tuning ==="
python -m finetune.train_lora \
    --data "${DATA_PATH}" \
    --output "${ADAPTER_PATH}" \
    --max-steps "${MAX_STEPS}" \
    --batch-size 1 \
    --grad-accum 4

echo ""
echo "Adapter saved to: ${ADAPTER_PATH}"

if [ "${MERGE}" = "1" ]; then
    echo ""
    echo "=== Step 3: Merge adapter ==="
    python -m finetune.merge_adapters \
        --adapter "${ADAPTER_PATH}" \
        --output "${MERGED_PATH}"
    echo "Merged model saved to: ${MERGED_PATH}"
fi

echo ""
echo "=== Fine-tuning pipeline complete ==="
echo "To load the new adapter in the running API, set:"
echo "  MLML_ADAPTER_DIR=${ADAPTER_PATH}"
echo "and restart, or call the adapter reload endpoint."
