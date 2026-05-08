#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODE_DIR="$ROOT_DIR/code"
LOG_DIR="$CODE_DIR/exp_logs"
mkdir -p "$LOG_DIR"

DATASET="yelp2018"
EPOCHS="300"
TOPKS="[20]"
MODEL="ngcf"

run_one() {
  local seed="$1"
  local tag="${MODEL}_${DATASET}_ep${EPOCHS}_s${seed}"
  echo "[RUN] ${tag}"
  (
    cd "$CODE_DIR"
    python main.py \
      --dataset "$DATASET" \
      --model "$MODEL" \
      --epochs "$EPOCHS" \
      --seed "$seed" \
      --topks "$TOPKS" \
      --recdim 64 \
      --layer 3 \
      --bpr_batch 4096 \
      --testbatch 2048 \
      --lr 0.001 \
      --decay 1e-4 \
      --multicore 1 \
      --tensorboard 0 \
      --test_interval 10 \
      --save_interval 300 \
      --comment "$tag"
  ) 2>&1 | tee "$LOG_DIR/${tag}.log"
}

run_one 2020
run_one 2021
run_one 2022

echo "Done. Logs at: $LOG_DIR"
