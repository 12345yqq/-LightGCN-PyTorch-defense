#!/usr/bin/env bash
set -euo pipefail

cd /public/home/cqu_cs_205/LightGCN-PyTorch-master/code

L2=1e-3
LR=0.001
DECAY=5e-4
EPOCHS=300

run_one() {
  local seed="$1"
  local tag="yelp300_lrn_l2_${L2}_s${seed}"
  local ckpt="./checkpoints/lgn-yelp2018-3-64.pth.tar"
  local out_ckpt="./checkpoints/lgn-yelp2018-3-64-${tag}.pth.tar"

  echo "===== [TRAIN] seed=${seed}, tag=${tag} ====="
  python main.py \
    --dataset yelp2018 --model lgn --layer 3 --recdim 64 \
    --layer_agg learnable --layer_w_l2 "${L2}" \
    --seed "${seed}" --epochs "${EPOCHS}" \
    --lr "${LR}" --decay "${DECAY}" \
    --test_interval 10 --save_interval 50 \
    --tensorboard 1 --comment "${tag}" \
    --cuda 1 --gpu_id 0

  echo "===== [TEST] seed=${seed}, tag=${tag} ====="
  python test.py \
    --dataset yelp2018 --model lgn --layer 3 --recdim 64 \
    --layer_agg learnable --layer_w_l2 "${L2}" \
    --seed "${seed}" --topks "[10,20,50]" \
    --tensorboard 0 --multicore 0 --cuda 1 --gpu_id 0

  cp "${ckpt}" "${out_ckpt}"
  echo "Saved checkpoint: ${out_ckpt}"
}

run_one 2020
run_one 2021
run_one 2022

echo "All learnable+l2(300ep,3seeds) runs finished."
