#!/usr/bin/env bash
set -euo pipefail

cd /public/home/cqu_cs_205/LightGCN-PyTorch-master/code

L2=1e-3
LR=0.001
DECAY=5e-4
EPOCHS=150

run_one() {
  local tau="$1"
  local seed="$2"
  local tau_tag
  tau_tag=$(echo "$tau" | sed 's/\./p/g')
  local comment="yelp150_tau${tau_tag}_l2_1e-3_s${seed}"
  local ckpt_src="./checkpoints/lgn-yelp2018-3-64.pth.tar"
  local ckpt_dst="./checkpoints/lgn-yelp2018-3-64-yelp150_tau${tau_tag}_l2_1e-3_s${seed}.pth.tar"

  echo "===== [TRAIN] tau=${tau}, seed=${seed} ====="
  python main.py \
    --dataset yelp2018 --model lgn --layer 3 --recdim 64 \
    --layer_agg learnable --layer_w_l2 "${L2}" --layer_w_tau "${tau}" \
    --seed "${seed}" --epochs "${EPOCHS}" --load 0 \
    --lr "${LR}" --decay "${DECAY}" \
    --test_interval 10 --save_interval 50 \
    --tensorboard 1 --comment "${comment}" \
    --cuda 1 --gpu_id 0

  echo "===== [TEST] tau=${tau}, seed=${seed} ====="
  python test.py \
    --dataset yelp2018 --model lgn --layer 3 --recdim 64 \
    --layer_agg learnable --layer_w_l2 "${L2}" --layer_w_tau "${tau}" \
    --seed "${seed}" --topks "[10,20,50]" \
    --tensorboard 0 --multicore 0 --cuda 1 --gpu_id 0

  cp "${ckpt_src}" "${ckpt_dst}"
  echo "Saved checkpoint: ${ckpt_dst}"
}

for tau in 0.5 1.0 2.0; do
  for seed in 2020 2021 2022; do
    run_one "$tau" "$seed"
  done
done

python summarize_tau_results.py

echo "DONE: tau screen finished and summary generated."
