#!/usr/bin/env bash
set -euo pipefail

cd /public/home/cqu_cs_205/LightGCN-PyTorch-master/code

L2=1e-3
LR=0.001
DECAY=5e-4

train_resume_2021() {
  local seed=2021
  local tag="yelp300_lrn_l2_${L2}_s2021_resume"
  local ckpt="./checkpoints/lgn-yelp2018-3-64.pth.tar"
  local out_ckpt="./checkpoints/lgn-yelp2018-3-64-yelp300_lrn_l2_1e-3_s2021.pth.tar"

  echo "===== [RESUME TRAIN] seed=${seed}, tag=${tag} ====="
  # Resume from current checkpoint state; run remaining ~190 epochs to reach ~300 total updates.
  python main.py \
    --dataset yelp2018 --model lgn --layer 3 --recdim 64 \
    --layer_agg learnable --layer_w_l2 "${L2}" \
    --seed "${seed}" --epochs 190 --load 1 \
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

train_fresh_2022() {
  local seed=2022
  local tag="yelp300_lrn_l2_${L2}_s2022"
  local ckpt="./checkpoints/lgn-yelp2018-3-64.pth.tar"
  local out_ckpt="./checkpoints/lgn-yelp2018-3-64-yelp300_lrn_l2_1e-3_s2022.pth.tar"

  echo "===== [FRESH TRAIN] seed=${seed}, tag=${tag} ====="
  python main.py \
    --dataset yelp2018 --model lgn --layer 3 --recdim 64 \
    --layer_agg learnable --layer_w_l2 "${L2}" \
    --seed "${seed}" --epochs 300 --load 0 \
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

train_resume_2021
train_fresh_2022

python summarize_l2_results.py

echo "All resumed runs + summary finished."
