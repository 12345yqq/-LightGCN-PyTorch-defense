#!/usr/bin/env bash
set -euo pipefail

# Quick-screen settings (can be overridden by env vars)
DATASET="${DATASET:-yelp2018}"
EPOCHS="${EPOCHS:-150}"
SEED="${SEED:-2020}"
TOPKS="${TOPKS:-[20]}"
LAYER_AGG="${LAYER_AGG:-mean}"
TEST_INTERVAL="${TEST_INTERVAL:-10}"

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODE_DIR="$ROOT_DIR/code"
LOG_DIR="$ROOT_DIR/code/exp_logs"
mkdir -p "$LOG_DIR"

run_case () {
  local name="$1"
  shift
  echo "[RUN] $name"
  (
    cd "$CODE_DIR"
    python main.py \
      --dataset "$DATASET" \
      --model lgn \
      --layer_agg "$LAYER_AGG" \
      --epochs "$EPOCHS" \
      --seed "$SEED" \
      --topks "$TOPKS" \
      --test_interval "$TEST_INTERVAL" \
      --save_interval "$EPOCHS" \
      --comment "$name" \
      "$@"
  ) 2>&1 | tee "$LOG_DIR/${name}.log"
}

run_case "hnm_qs_baseline_${DATASET}_s${SEED}" \
  --hnm_mode none

run_case "hnm_qs_mix_${DATASET}_s${SEED}" \
  --hnm_mode mix \
  --hnm_ratio 0.4

run_case "hnm_qs_curr_${DATASET}_s${SEED}" \
  --hnm_mode curriculum \
  --hnm_curr_start 0.2 \
  --hnm_curr_end 0.6 \
  --hnm_warmup_epochs "$EPOCHS"

echo "All 3 HNM quick-screen runs finished. Logs in: $LOG_DIR"
