#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
from pathlib import Path

import torch
from safetensors import safe_open


def numel_from_safetensors(path: Path):
    total = 0
    examples = []
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            shape = handle.get_slice(key).get_shape()
            n = math.prod(shape)
            total += n
            if len(examples) < 8:
                examples.append((key, shape, n))
    return total, examples


def flatten_tensors(obj, prefix=""):
    if torch.is_tensor(obj):
        yield prefix, tuple(obj.shape), obj.numel()
    elif isinstance(obj, dict):
        for k, v in obj.items():
            child = f"{prefix}.{k}" if prefix else str(k)
            yield from flatten_tensors(v, child)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            child = f"{prefix}.{i}" if prefix else str(i)
            yield from flatten_tensors(v, child)


def top_level_summary(obj):
    if not isinstance(obj, dict):
        return {"type": type(obj).__name__}
    out = {}
    for k, v in obj.items():
        if torch.is_tensor(v):
            out[k] = {"type": "tensor", "shape": list(v.shape), "numel": v.numel()}
        elif isinstance(v, dict):
            count = sum(1 for _ in flatten_tensors(v))
            numel = sum(n for _, _, n in flatten_tensors(v))
            out[k] = {"type": "dict", "tensor_count": count, "numel": numel}
        else:
            out[k] = {"type": type(v).__name__}
    return out


def count_torch_checkpoint(path: Path, prefer_key=None):
    obj = torch.load(path, map_location="cpu")
    summary = top_level_summary(obj)
    target = obj
    used_key = None
    if isinstance(obj, dict):
        candidates = []
        if prefer_key and prefer_key in obj:
            candidates.append(prefer_key)
        candidates += ["state_dict", "model", "module", "model_state_dict", "net", "ema"]
        for key in candidates:
            if key in obj and isinstance(obj[key], dict):
                target = obj[key]
                used_key = key
                break
    tensors = list(flatten_tensors(target))
    total = sum(n for _, _, n in tensors)
    examples = tensors[:8]
    return total, examples, summary, used_key


def count_path(path: Path, prefer_key=None):
    if path.suffix == ".safetensors":
        total, examples = numel_from_safetensors(path)
        return total, examples, {"format": "safetensors"}, None
    return count_torch_checkpoint(path, prefer_key=prefer_key)


def human_m(n):
    return round(n / 1_000_000, 3)


def size_mb(path: Path):
    if path.is_file():
        return path.stat().st_size / 1024**2
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1024**2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("paper_artifacts/model_param_counts"))
    parser.add_argument("--item", action="append", nargs="+", help="name path [prefer_key]")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    details = {}
    for item in args.item or []:
        if len(item) < 2:
            raise ValueError("--item requires name and path")
        name = item[0]
        path = Path(item[1])
        prefer_key = item[2] if len(item) > 2 else None
        total, examples, summary, used_key = count_path(path, prefer_key=prefer_key)
        rows.append(
            {
                "model": name,
                "path": str(path),
                "counted_key": used_key or "",
                "params": total,
                "params_m": human_m(total),
                "file_size_mb": round(size_mb(path), 3),
            }
        )
        details[name] = {
            "path": str(path),
            "counted_key": used_key,
            "top_level_summary": summary,
            "examples": [(k, list(shape), n) for k, shape, n in examples],
        }

    with (args.out / "model_param_counts.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "model_param_counts.json").write_text(json.dumps(rows, indent=2))
    (args.out / "model_param_count_details.json").write_text(json.dumps(details, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
