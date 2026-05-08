#!/bin/bash
set -e

run_root=code/exp_logs/paper_yelp2018_layer_sweep_300
mkdir -p "$run_root"
progress=$run_root/progress.log
summary=$run_root/summary.tsv

echo -e "layer\tdataset\tseed\tduration_sec\tprecision20\trecall20\tndcg20\tlog" > "$summary"

function run_one() {
  layer=$1
  seed=$2
  logf=$run_root/layer${layer}_seed${seed}.log
  echo "[START] $(date '+%F %T') dataset=yelp2018 seed=$seed layer=$layer epochs=300 runtime=python3(rocm)" | tee -a "$progress"
  start=$(date +%s)
  python3 code/main.py \
    --dataset yelp2018 \
    --model lgn \
    --layer "$layer" \
    --epochs 300 \
    --seed "$seed" \
    --tensorboard 0 \
    --multicore 1 \
    --cuda 1 \
    --gpu_id 0 \
    --comment "yelp2018_layer${layer}_seed${seed}_300" 2>&1 | tee "$logf"
  end=$(date +%s)
  dur=$((end-start))

  prec=$(grep -a "precision" "$logf" | tail -n1 | grep -oE "[0-9]+\.[0-9]+" | tail -n1 || echo nan)
  rec=$(grep -a "recall" "$logf" | tail -n1 | grep -oE "[0-9]+\.[0-9]+" | tail -n1 || echo nan)
  ndcg=$(grep -a "ndcg" "$logf" | tail -n1 | grep -oE "[0-9]+\.[0-9]+" | tail -n1 || echo nan)

  echo -e "$layer\tyelp2018\t$seed\t$dur\t$prec\t$rec\t$ndcg\t$logf" >> "$summary"
  echo "[DONE ] $(date '+%F %T') dataset=yelp2018 seed=$seed layer=$layer duration_sec=$dur" | tee -a "$progress"
}

# single-seed sweep to match the existing yelp2018 layer logs style
for layer in 1 2 3 4; do
  run_one "$layer" 2020
done

echo "[ALL_DONE] $(date '+%F %T')" | tee -a "$progress"