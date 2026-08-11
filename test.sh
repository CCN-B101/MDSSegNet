#!/usr/bin/env bash
set -euo pipefail

# Run after train.sh has produced all 15 best_mIoU checkpoints.

CHECKPOINT_ROOT="checkpoints/statistical"
RESULT_ROOT="results/statistical"

python test.py \
    --data-root "/home/test/CCN/Crack500/test" \
    --dataset "Crack500" \
    --checkpoint-root "${CHECKPOINT_ROOT}" \
    --result-root "${RESULT_ROOT}" \
    --seeds 0 1 2 3 4 \
    --crop-size 448 448 \
    --num-workers 0

python test.py \
    --data-root "/home/test/CCN/DeepCrack/test" \
    --dataset "DeepCrack" \
    --checkpoint-root "${CHECKPOINT_ROOT}" \
    --result-root "${RESULT_ROOT}" \
    --seeds 0 1 2 3 4 \
    --crop-size 448 448 \
    --num-workers 0

python test.py \
    --data-root "/home/test/CCN/CTC444/test" \
    --dataset "CTC" \
    --checkpoint-root "${CHECKPOINT_ROOT}" \
    --result-root "${RESULT_ROOT}" \
    --seeds 0 1 2 3 4 \
    --crop-size 448 448 \
    --num-workers 0

echo ""
echo "All three statistical evaluations are complete."
echo "See results/statistical/<dataset>/statistical_summary.csv"
