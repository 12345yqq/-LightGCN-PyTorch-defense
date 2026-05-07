#!/usr/bin/env python3
"""Extract per-epoch training loss from logs and combine HNM/non-HNM runs for comparison."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import List, Optional


EPOCH_LOSS_RE = re.compile(r"EPOCH\[(\d+)/(\d+)\].*?loss\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)")
HNM_MODE_RE = re.compile(r"'hnm_mode':\s*'([^']+)'")


def extract_from_log(path: Path) -> List[dict]:
    rows = []
    hnm_mode: Optional[str] = None
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if hnm_mode is None:
                m = HNM_MODE_RE.search(line)
                if m:
                    hnm_mode = m.group(1)
            m2 = EPOCH_LOSS_RE.search(line)
            if m2:
                epoch = int(m2.group(1))
                total = int(m2.group(2))
                loss = float(m2.group(3))
                rows.append({"epoch": epoch, "total_epochs": total, "loss": loss, "hnm_mode": hnm_mode or "unknown"})
    return rows


def write_tsv(out: Path, rows: List[dict]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "seed", "epoch", "loss", "hnm_mode"], delimiter="\t")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def infer_seed(path: Path) -> str:
    m = re.search(r"s(\d{4})", path.name)
    return m.group(1) if m else path.stem


def infer_method(path: Path) -> str:
    name = path.parent.name.lower()
    if "best" in name or "gowalla_best" in path.stem:
        return "LLA"
    if "combo" in name or "hnm" in name:
        return "HNM_combo"
    if "gpu_seed" in name or "baseline" in path.stem:
        return "mean"
    return path.stem


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-paths", nargs="+", type=Path, required=True, help="One or more log files to parse")
    parser.add_argument("--out", type=Path, required=True, help="Output TSV for combined losses")
    args = parser.parse_args()

    combined = []
    for p in args.log_paths:
        if not p.exists():
            print(f"skip missing: {p}")
            continue
        rows = extract_from_log(p)
        method = infer_method(p)
        seed = infer_seed(p)
        for r in rows:
            combined.append({"method": method, "seed": seed, "epoch": r["epoch"], "loss": r["loss"], "hnm_mode": r["hnm_mode"]})

    write_tsv(args.out, combined)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
