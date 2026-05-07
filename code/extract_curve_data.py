#!/usr/bin/env python3
"""Extract convergence curve points from training logs.

The repo prints evaluation results every `test_interval` epochs in the form:
    {'precision': array([...]), 'recall': array([...]), 'ndcg': array([...])}

This script pairs each metric snapshot with the following `EPOCH[x/y]` line,
then writes a table that can be plotted directly.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from math import sqrt
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


METRIC_RE = re.compile(
    r"\{'precision': array\(\[([-+0-9.eE]+)\]\), 'recall': array\(\[([-+0-9.eE]+)\]\), 'ndcg': array\(\[([-+0-9.eE]+)\]\)\}"
)
EPOCH_RE = re.compile(r"EPOCH\[(\d+)/(\d+)\]")


def parse_log(log_path: Path) -> List[dict]:
    rows = []
    pending_metric: Optional[Tuple[float, float, float]] = None

    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            metric_match = METRIC_RE.search(line)
            if metric_match:
                pending_metric = (
                    float(metric_match.group(1)),
                    float(metric_match.group(2)),
                    float(metric_match.group(3)),
                )
                continue

            epoch_match = EPOCH_RE.search(line)
            if epoch_match and pending_metric is not None:
                train_epoch = int(epoch_match.group(1))
                rows.append(
                    {
                        "epoch": max(train_epoch - 1, 0),
                        "precision": pending_metric[0],
                        "recall": pending_metric[1],
                        "ndcg": pending_metric[2],
                    }
                )
                pending_metric = None

    return rows


def infer_seed(log_path: Path) -> str:
    match = re.search(r"s(\d{4})", log_path.name)
    return match.group(1) if match else log_path.stem


def infer_group(log_path: Path) -> str:
    stem = log_path.stem.lower()
    if "gowalla" in stem:
        return "gowalla_best"
    if "lastfm" in stem:
        return "lastfm"
    if "yelp" in stem:
        return "yelp2018"
    return "unknown"


def write_tsv(path: Path, rows: Iterable[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract curve data from repo logs")
    parser.add_argument(
        "--log-dir",
        type=Path,
        required=True,
        help="Directory containing training .log files",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for TSV files",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.log",
        help="Glob pattern relative to log-dir",
    )
    args = parser.parse_args()

    log_files = sorted(args.log_dir.glob(args.pattern))
    if not log_files:
        raise SystemExit(f"No log files matched {args.log_dir / args.pattern}")

    all_rows = []
    for log_path in log_files:
        series = parse_log(log_path)
        seed = infer_seed(log_path)
        group = infer_group(log_path)
        if not series:
            continue
        out_file = args.out_dir / f"{log_path.stem}_curve.tsv"
        write_tsv(
            out_file,
            (
                {
                    "group": group,
                    "seed": seed,
                    **row,
                }
                for row in series
            ),
            ["group", "seed", "epoch", "precision", "recall", "ndcg"],
        )
        for row in series:
            all_rows.append({"group": group, "seed": seed, **row})

    all_out = args.out_dir / "all_curves.tsv"
    write_tsv(all_out, all_rows, ["group", "seed", "epoch", "precision", "recall", "ndcg"])

    grouped = defaultdict(list)
    for row in all_rows:
        grouped[row["epoch"]].append(row)

    mean_rows = []
    std_rows = []
    for epoch in sorted(grouped):
        epoch_rows = grouped[epoch]
        count = float(len(epoch_rows))
        for metric in ("precision", "recall", "ndcg"):
            values = [float(r[metric]) for r in epoch_rows]
            mean = sum(values) / count
            var = sum((x - mean) ** 2 for x in values) / count
            if metric == "precision":
                precision_mean, precision_std = mean, sqrt(var)
            elif metric == "recall":
                recall_mean, recall_std = mean, sqrt(var)
            else:
                ndcg_mean, ndcg_std = mean, sqrt(var)
        mean_rows.append(
            {
                "epoch": epoch,
                "precision": precision_mean,
                "recall": recall_mean,
                "ndcg": ndcg_mean,
            }
        )
        std_rows.append(
            {
                "epoch": epoch,
                "precision": precision_std,
                "recall": recall_std,
                "ndcg": ndcg_std,
            }
        )

    write_tsv(args.out_dir / "mean_curve.tsv", mean_rows, ["epoch", "precision", "recall", "ndcg"])
    write_tsv(args.out_dir / "std_curve.tsv", std_rows, ["epoch", "precision", "recall", "ndcg"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())