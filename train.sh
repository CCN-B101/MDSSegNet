#!/usr/bin/env bash
set -euo pipefail

# Run this script from the project root.
# Five independent seeds are used for each dataset.
SEEDS=(0 1 2 3 4)

CHECKPOINT_ROOT="checkpoints/statistical"
RESULT_ROOT="results/statistical"

run_dataset () {
    DATASET_NAME="$1"
    DATA_ROOT="$2"

    echo "============================================================"
    echo "Training ${DATASET_NAME}"
    echo "Data root: ${DATA_ROOT}"
    echo "============================================================"

    for SEED in "${SEEDS[@]}"; do
        echo ""
        echo ">>> ${DATASET_NAME} | independent run seed=${SEED}"

        PYTHONHASHSEED="${SEED}" \
        CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        python train.py \
            --data-root "${DATA_ROOT}" \
            --dataset "${DATASET_NAME}" \
            --seed "${SEED}" \
            --epochs 50 \
            --crop-size 448 448 \
            --batch-size 8 \
            --lr 0.001 \
            --weight-decay 0.0001 \
            --num-workers 0 \
            --checkpoint-root "${CHECKPOINT_ROOT}" \
            --result-root "${RESULT_ROOT}"
    done
}

run_dataset "Crack500"  "/home/test/CCN/Crack500"
run_dataset "DeepCrack" "/home/test/CCN/DeepCrack"
run_dataset "CTC"       "/home/test/CCN/CTC444"

echo ""
echo "All 15 independent training runs are complete."
