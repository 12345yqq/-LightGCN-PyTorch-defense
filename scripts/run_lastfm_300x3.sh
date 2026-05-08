#!/bin/bash
set -e

run_root=code/exp_logs/paper_lastfm_300x3
mkdir -p "$run_root"
progress=$run_root/progress.log
summary=$run_root/summary.tsv

echo -e "group\tdataset\tmodel\tseed\tcl_weight\tprecision20\trecall20\tndcg20\tduration_sec" > "$summary"

function run_one() {
  group=$1; dataset=$2; model=$3; seed=$4; clw=$5; comment=$6
  logf=$run_root/${group}_${dataset}_seed${seed}_cl${clw}.log
  echo "[START] $(date '+%F %T') group=$group dataset=$dataset seed=$seed cl_weight=$clw" | tee -a "$progress"
  start=$(date +%s)
  python3 code/main.py --dataset $dataset --model $model --epochs 300 --seed $seed --cl_weight $clw --gpu_id 0 --comment $comment 2>&1 | tee "$logf"
  end=$(date +%s)
  dur=$((end-start))

  # 尝试从日志中提取最后出现的 precision/recall/ndcg 的数值
  prec=$(grep -a "precision" "$logf" | tail -n1 | grep -oE "[0-9]+\.[0-9]+" | tail -n1 || echo nan)
  rec=$(grep -a "recall" "$logf" | tail -n1 | grep -oE "[0-9]+\.[0-9]+" | tail -n1 || echo nan)
  ndcg=$(grep -a "ndcg" "$logf" | tail -n1 | grep -oE "[0-9]+\.[0-9]+" | tail -n1 || echo nan)

  echo -e "$group\t$dataset\t$model\t$seed\t$clw\t$prec\t$rec\t$ndcg\t$dur" >> "$summary"
  echo "[DONE ] $(date '+%F %T') group=$group dataset=$dataset seed=$seed duration_sec=$dur" | tee -a "$progress"
}

# baseline (cl_weight=0)
for s in 2020 2021 2022; do
  run_one "lastfm_baseline" lastfm lgn $s 0.0 "lastfm_baseline_seed${s}"
done

# simgcl (使用已调优的 cl_weight=0.01)
for s in 2020 2021 2022; do
  run_one "lastfm_simgcl" lastfm lgn $s 0.01 "lastfm_simgcl_seed${s}"
done

echo "[ALL_DONE] $(date '+%F %T')" | tee -a "$progress"
