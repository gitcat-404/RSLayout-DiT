#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
from safetensors import safe_open


FLOAT_RE = r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?"


def tensor_numel_from_safetensors(path: Path) -> int:
    total = 0
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            shape = handle.get_slice(key).get_shape()
            total += math.prod(shape)
    return total


def count_safetensors_under(root: Path) -> int:
    return sum(tensor_numel_from_safetensors(path) for path in root.rglob("*.safetensors"))


def bytes_under(root: Path) -> int:
    if root.is_file():
        return root.stat().st_size
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def parse_yolo_log(log_path: Path):
    rows = []
    pattern = re.compile(
        rf"^\s*(?P<class>\S+)\s+(?P<images>\d+)\s+(?P<instances>\d+)\s+"
        rf"(?P<p>{FLOAT_RE})\s+(?P<r>{FLOAT_RE})\s+"
        rf"(?P<map50>{FLOAT_RE})\s+(?P<map5095>{FLOAT_RE})\s*$"
    )
    for raw_line in log_path.read_text(errors="ignore").splitlines():
        clean = re.sub(r"\x1b\[[0-9;]*m", "", raw_line)
        match = pattern.match(clean)
        if not match:
            continue
        item = match.groupdict()
        if item["class"] == "all":
            continue
        rows.append(
            {
                "class": item["class"],
                "images": int(item["images"]),
                "instances": int(item["instances"]),
                "precision": float(item["p"]),
                "recall": float(item["r"]),
                "map50": float(item["map50"]),
                "map5095": float(item["map5095"]),
            }
        )
    if not rows:
        raise RuntimeError(f"No per-class YOLO rows found in {log_path}")
    return rows


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_per_class(before_rows, after_rows, out_path: Path, metric: str):
    before = {row["class"]: row for row in before_rows}
    after = {row["class"]: row for row in after_rows}
    classes = [c for c in after if c in before]
    classes.sort(key=lambda c: after[c][metric] - before[c][metric], reverse=True)

    before_values = [before[c][metric] for c in classes]
    after_values = [after[c][metric] for c in classes]
    x = list(range(len(classes)))
    width = 0.38

    plt.figure(figsize=(13, 4.8))
    plt.bar([i - width / 2 for i in x], before_values, width, label="w/o object-aware")
    plt.bar([i + width / 2 for i in x], after_values, width, label="RSLayout-DiT")
    plt.xticks(x, classes, rotation=55, ha="right", fontsize=8)
    plt.ylabel(metric.upper())
    plt.ylim(0, max(max(before_values), max(after_values)) * 1.18)
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.legend(frameon=False)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, default=Path("/path/to/FLUX-RS"))
    parser.add_argument("--before-ckpt", type=Path, required=True)
    parser.add_argument("--after-ckpt", type=Path, required=True)
    parser.add_argument("--before-yolo-log", type=Path, required=True)
    parser.add_argument("--after-yolo-log", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("paper_artifacts"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    base_params = count_safetensors_under(args.base_model)
    before_lora = args.before_ckpt / "lora.safetensors"
    after_lora = args.after_ckpt / "lora.safetensors"
    before_params = tensor_numel_from_safetensors(before_lora)
    after_params = tensor_numel_from_safetensors(after_lora)

    param_rows = [
        {
            "model": "Frozen FLUX-RS",
            "total_params": base_params,
            "trainable_params": 0,
            "trainable_percent": 0.0,
            "checkpoint_mb": 0.0,
        },
        {
            "model": "RSLayout-DiT w/o object-aware training",
            "total_params": base_params + before_params,
            "trainable_params": before_params,
            "trainable_percent": before_params / (base_params + before_params) * 100,
            "checkpoint_mb": bytes_under(before_lora) / 1024**2,
        },
        {
            "model": "RSLayout-DiT",
            "total_params": base_params + after_params,
            "trainable_params": after_params,
            "trainable_percent": after_params / (base_params + after_params) * 100,
            "checkpoint_mb": bytes_under(after_lora) / 1024**2,
        },
    ]
    write_csv(args.out_dir / "parameter_efficiency.csv", param_rows)
    (args.out_dir / "parameter_efficiency.json").write_text(json.dumps(param_rows, indent=2))

    before_rows = parse_yolo_log(args.before_yolo_log)
    after_rows = parse_yolo_log(args.after_yolo_log)
    write_csv(args.out_dir / "per_class_before.csv", before_rows)
    write_csv(args.out_dir / "per_class_after.csv", after_rows)
    plot_per_class(before_rows, after_rows, args.out_dir / "per_class_map50.png", "map50")
    plot_per_class(before_rows, after_rows, args.out_dir / "per_class_map5095.png", "map5095")

    print(f"Wrote artifacts to {args.out_dir}")
    print(f"Base params: {base_params:,}")
    print(f"LoRA params: {after_params:,}")


if __name__ == "__main__":
    main()
