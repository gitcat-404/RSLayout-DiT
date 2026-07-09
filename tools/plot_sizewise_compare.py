#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_csv(path: Path):
    with path.open() as f:
        return {row["size"]: row for row in csv.DictReader(f)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metric", choices=["map50", "map5095"], default="map5095")
    args = parser.parse_args()

    before = load_csv(args.before)
    after = load_csv(args.after)
    sizes = ["small", "medium", "large"]
    x = np.arange(len(sizes))
    width = 0.38

    plt.figure(figsize=(5.8, 3.8))
    plt.bar(x - width / 2, [float(before[s][args.metric]) for s in sizes], width, label="w/o object-aware")
    plt.bar(x + width / 2, [float(after[s][args.metric]) for s in sizes], width, label="RSLayout-DiT")
    plt.xticks(x, sizes)
    plt.ylabel(args.metric.upper())
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.legend(frameon=False)
    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=300)


if __name__ == "__main__":
    main()
