#!/bin/bash
set -e

run_root=code/exp_logs/paper_gowalla_best_300x3
mkdir -p "$run_root"
progress=$run_root/progress.log
summary=$run_root/summary.tsv

echo -e "group\tdataset\tmodel\tseed\tparams\tprecision20\trecall20\tndcg20\tduration_sec" > "$summary"

function run_one() {
  group=$1; dataset=$2; model=$3; seed=$4; params=$5; comment=$6
  logf=$run_root/${group}_${dataset}_seed${seed}.log
  echo "[START] $(date '+%F %T') group=$group dataset=$dataset seed=$seed params=$params" | tee -a "$progress"
  start=$(date +%s)
  python3 code/main.py --dataset $dataset --model $model --epochs 300 --seed $seed \
    --layer_agg learnable --layer_w_l2 1e-4 --layer_w_tau 0.5 \
    --cl_weight 0.01 --cl_temp 0.2 --gpu_id 0 --comment $comment 2>&1 | tee "$logf"
  end=$(date +%s)
  dur=$((end-start))

  prec=$(grep -a "precision" "$logf" | tail -n1 | grep -oE "[0-9]+\.[0-9]+" | tail -n1 || echo nan)
  rec=$(grep -a "recall" "$logf" | tail -n1 | grep -oE "[0-9]+\.[0-9]+" | tail -n1 || echo nan)
  ndcg=$(grep -a "ndcg" "$logf" | tail -n1 | grep -oE "[0-9]+\.[0-9]+" | tail -n1 || echo nan)

  echo -e "$group\t$dataset\t$model\t$seed\t$params\t$prec\t$rec\t$ndcg\t$dur" >> "$summary"
  echo "[DONE ] $(date '+%F %T') group=$group dataset=$dataset seed=$seed duration_sec=$dur" | tee -a "$progress"
}

params_desc="layer_agg=learnable;layer_w_l2=1e-4;layer_w_tau=0.5;cl=0.01,temp=0.2"
for s in 2020 2021 2022; do
  run_one "gowalla_best" gowalla lgn $s "$params_desc" "gowalla_best_seed${s}"
done

echo "[ALL_DONE] $(date '+%F %T')" | tee -a "$progress"
