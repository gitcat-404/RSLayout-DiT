#!/usr/bin/env python
import argparse
import json
import re
from pathlib import Path


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def last_float(pattern: str, text: str):
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    return float(matches[-1]) if matches else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--baseline-map5095", type=float, default=0.2617600859797639)
    parser.add_argument("--baseline-map50", type=float, default=0.4686028887640507)
    parser.add_argument("--baseline-fid", type=float, default=29.41145)
    parser.add_argument("--baseline-clip-local", type=float, default=0.213758)
    parser.add_argument("--baseline-clip-global", type=float, default=0.257097)
    args = parser.parse_args()

    logs = args.eval_dir / "logs"
    clip_log = read_text(logs / "clipscore.log")
    fid_log = read_text(logs / "fid.log")
    yolo_log = read_text(logs / "yoloscore.log")

    metrics = {
        "clip_local": last_float(r"local_similarity:\s*([0-9.]+)", clip_log),
        "clip_global": last_float(r"global_similarity'?:\s*([0-9.]+)", clip_log),
        "fid": last_float(r"frechet_inception_distance:\s*([0-9.]+)", fid_log)
        or last_float(r"Frechet Inception Distance:\s*([0-9.]+)", fid_log),
        "yolo_map50": last_float(r"YOLOSCORE_MAP50\s+([0-9.]+)", yolo_log),
        "yolo_map5095": last_float(r"YOLOSCORE_MAP50_95\s+([0-9.]+)", yolo_log),
    }
    baseline = {
        "clip_local": args.baseline_clip_local,
        "clip_global": args.baseline_clip_global,
        "fid": args.baseline_fid,
        "yolo_map50": args.baseline_map50,
        "yolo_map5095": args.baseline_map5095,
    }
    deltas = {
        key: (metrics[key] - value if metrics[key] is not None else None)
        for key, value in baseline.items()
    }
    summary = {
        "metrics": metrics,
        "baseline_objloss_v2": baseline,
        "delta_vs_objloss_v2": deltas,
        "recommendation": "start_multiscale_condition"
        if metrics["yolo_map5095"] is not None and metrics["yolo_map5095"] >= 0.255
        else "review_objloss_v3_before_continuing",
    }
    out = args.eval_dir / "summary_metrics.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
