#!/usr/bin/env bash
set -euo pipefail

ROOT="/public/home/cqu_cs_205/LightGCN-PyTorch-master"
PY="$ROOT/.conda_env/bin/python"
MAIN="$ROOT/code/main.py"
OUTDIR="$ROOT/code/exp_logs/hnm_must"
LOGDIR="$OUTDIR/logs"
SUMMARY="$OUTDIR/summary.csv"

mkdir -p "$LOGDIR"

echo "run_name,group,seed,hnm_mode,hnm_ratio,hnm_curr_start,hnm_curr_end,hnm_warmup_epochs,epochs,test_interval,topks,elapsed_sec,start_ts,end_ts" > "$SUMMARY"

run_one() {
  local run_name="$1"
  local group_name="$2"
  local seed="$3"
  local hnm_mode="$4"
  local hnm_ratio="$5"
  local curr_start="$6"
  local curr_end="$7"
  local warmup="$8"

  local epochs=100
  local test_interval=10
  local topks='[20]'
  local log_file="$LOGDIR/${run_name}.log"

  local start_ts end_ts elapsed
  start_ts=$(date +%s)

  "$PY" "$MAIN" \
    --dataset gowalla \
    --model lgn \
    --seed "$seed" \
    --epochs "$epochs" \
    --test_interval "$test_interval" \
    --save_interval 100 \
    --topks "$topks" \
    --hnm_mode "$hnm_mode" \
    --hnm_ratio "$hnm_ratio" \
    --hnm_curr_start "$curr_start" \
    --hnm_curr_end "$curr_end" \
    --hnm_warmup_epochs "$warmup" \
    --comment "$run_name" \
    2>&1 | tee "$log_file"

  end_ts=$(date +%s)
  elapsed=$((end_ts - start_ts))

  echo "${run_name},${group_name},${seed},${hnm_mode},${hnm_ratio},${curr_start},${curr_end},${warmup},${epochs},${test_interval},${topks},${elapsed},${start_ts},${end_ts}" >> "$SUMMARY"
}

# Group 1: HNM negative-result robustness (must)
for seed in 2020 2021 2022; do
  run_one "g1_baseline_seed${seed}" "robustness" "$seed" "none" "0.0" "0.2" "0.6" "100"
  run_one "g1_mix04_seed${seed}" "robustness" "$seed" "mix" "0.4" "0.2" "0.6" "100"
  run_one "g1_curr_seed${seed}" "robustness" "$seed" "curriculum" "0.4" "0.2" "0.6" "100"
done

# Group 2: HNM strength sensitivity (must, single seed fast screen)
for ratio in 0.1 0.2 0.3 0.4; do
  run_one "g2_mix${ratio}_seed2020" "ratio_screen" "2020" "mix" "$ratio" "0.2" "0.6" "100"
done

echo "All runs completed. Summary: $SUMMARY"
