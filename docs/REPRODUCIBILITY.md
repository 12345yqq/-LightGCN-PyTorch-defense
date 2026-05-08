# Reproducibility Guide

This document captures the exact steps to reproduce experiments for this repository.

## 1) Code snapshot

- Tag: exp-snapshot-20260508
- Commit: b40a09749a5644a619fc17475a411f1974faec18

## 2) Environment

Choose one of the two methods below:

### Option A: pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Option B: conda

```bash
conda env create -f env.yml
conda activate <env_name>
```

## 3) Data

Place datasets under `data/` with the following structure:

- data/lastfm/
- data/gowalla/
- data/yelp2018/
- data/amazon-book/

Each dataset folder should contain the files expected by the original LightGCN implementation.

## 4) Training and evaluation

Main entry point: `code/main.py`

Example commands (adjust as needed):

### LightGCN (baseline)

```bash
python code/main.py --model lgn --dataset yelp2018 --seed 2020 --epochs 300 --layer 3 --recdim 64 --lr 0.001 --decay 1e-4 --bpr_batch 2048 --test_interval 10
```

### SimGCL

```bash
python code/main.py --model lgn --dataset yelp2018 --seed 2020 --epochs 300 --layer 3 --recdim 64 --lr 0.001 --decay 1e-4 --bpr_batch 2048 --cl_weight 0.1 --cl_temp 0.2 --test_interval 10
```

## 5) Outputs and logs

- Checkpoints: `code/checkpoints/`
- Logs: `code/exp_logs/`
- TensorBoard runs: `code/runs/`

## 6) Key parameters

These defaults are defined in `code/parse.py`:

- epochs: 1000 (override to 300 when needed)
- test_interval: 10
- layer: 3
- recdim: 64
- lr: 0.001
- decay: 1e-4
- bpr_batch: 2048
- seed: 2020

## 7) Notes

- If you need exact commands for a specific run, check the corresponding log in `code/exp_logs/` and copy the header configuration.
- For large artifacts, prefer Git LFS or external storage and document the download link here.
