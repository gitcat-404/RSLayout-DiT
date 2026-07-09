#!/usr/bin/env python
"""Open-vocabulary detector evaluation for zero-shot layout generation."""

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rslayout_utils.prompting import display_class


GENERATED_RE = re.compile(r"^(?:(?P<index>\d+)_)?(?P<stem>.+?)(?:_generated)?\.png$")


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def valid_box(box: Iterable[float]) -> bool:
    x0, y0, x1, y1 = [float(v) for v in box]
    return x1 > x0 and y1 > y0


def normalized_to_pixel(box: Iterable[float], width: int, height: int) -> List[float]:
    x0, y0, x1, y1 = [float(v) for v in box]
    return [x0 * width, y0 * height, x1 * width, y1 * height]


def box_iou(a: Iterable[float], b: Iterable[float]) -> float:
    ax0, ay0, ax1, ay1 = [float(v) for v in a]
    bx0, by0, bx1, by1 = [float(v) for v in b]
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def center_error(a: Iterable[float], b: Iterable[float], width: int, height: int) -> float:
    ax0, ay0, ax1, ay1 = [float(v) for v in a]
    bx0, by0, bx1, by1 = [float(v) for v in b]
    acx = 0.5 * (ax0 + ax1)
    acy = 0.5 * (ay0 + ay1)
    bcx = 0.5 * (bx0 + bx1)
    bcy = 0.5 * (by0 + by1)
    diag = math.sqrt(width * width + height * height)
    return math.sqrt((acx - bcx) ** 2 + (acy - bcy) ** 2) / diag


def mean(values: List[float]) -> Optional[float]:
    return float(np.mean(values)) if values else None


def build_generated_index(generated_dir: Path) -> Dict[str, Path]:
    by_stem: Dict[str, Path] = {}
    for path in sorted(generated_dir.glob("*.png")):
        if any(token in path.stem for token in ("_layout", "_labeled", "_condition")):
            continue
        match = GENERATED_RE.match(path.name)
        if not match:
            continue
        stem = match.group("stem")
        by_stem.setdefault(stem, path)
    return by_stem


def open_image(path: Path) -> Image.Image:
    try:
        return Image.open(path).convert("RGB")
    except FileNotFoundError:
        if path.is_symlink():
            target = Path(path.readlink())
            target_text = str(target)
            if target_text.startswith("/data1/"):
                swapped = Path("/data2" + target_text[len("/data1") :])
                return Image.open(swapped).convert("RGB")
        raise


def make_queries(label: str) -> List[str]:
    name = display_class(label)
    # Keep the first text label closest to the object name for post-processing.
    queries = [name, f"aerial photo of {name}", f"satellite image of {name}"]
    deduped = []
    for query in queries:
        if query not in deduped:
            deduped.append(query)
    return deduped


def post_process(processor, outputs, target_sizes, threshold: float, queries: List[str]):
    if hasattr(processor, "post_process_grounded_object_detection"):
        try:
            return processor.post_process_grounded_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=threshold,
                text_labels=[queries],
            )[0]
        except TypeError:
            return processor.post_process_grounded_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=threshold,
            )[0]
    return processor.post_process_object_detection(
        outputs=outputs,
        target_sizes=target_sizes,
        threshold=threshold,
    )[0]


def result_to_detections(result: dict) -> List[dict]:
    boxes = result.get("boxes", [])
    scores = result.get("scores", [])
    labels = result.get("labels", [])
    text_labels = result.get("text_labels", [None] * len(boxes))
    detections = []
    for idx, box in enumerate(boxes):
        label_value = labels[idx].item() if hasattr(labels[idx], "item") else labels[idx]
        score_value = scores[idx].item() if hasattr(scores[idx], "item") else float(scores[idx])
        detections.append(
            {
                "box": [float(v) for v in box.tolist()],
                "score": float(score_value),
                "label": label_value,
                "text_label": text_labels[idx] if idx < len(text_labels) else None,
            }
        )
    return detections


def summarize(rows: List[dict]) -> dict:
    def group_summary(items: List[dict]) -> dict:
        return {
            "n": len(items),
            "detected_rate": mean([float(r["detected"]) for r in items]),
            "recall_iou30": mean([float(r["best_iou"] >= 0.30) for r in items]),
            "recall_iou50": mean([float(r["best_iou"] >= 0.50) for r in items]),
            "best_iou": mean([r["best_iou"] for r in items]),
            "center_error": mean([r["center_error"] for r in items if r["center_error"] is not None]),
            "det_score": mean([r["best_score"] for r in items if r["best_score"] is not None]),
        }

    per_class = defaultdict(list)
    per_difficulty = defaultdict(list)
    for row in rows:
        per_class[row["class"]].append(row)
        per_difficulty[row["difficulty"]].append(row)
    return {
        "overall": group_summary(rows),
        "per_class": {key: group_summary(value) for key, value in sorted(per_class.items())},
        "per_difficulty": {key: group_summary(value) for key, value in sorted(per_difficulty.items())},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", type=str, default="google/owlvit-base-patch32")
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model_id)
    model = model.to(device).eval()

    metadata = load_jsonl(args.metadata)
    if args.max_samples is not None:
        metadata = metadata[: args.max_samples]
    generated_by_stem = build_generated_index(args.generated_dir)

    sample_rows = []
    missing = []
    detection_cache: Dict[Tuple[str, str], List[dict]] = {}

    for row in tqdm(metadata, desc=f"Open-det {args.generated_dir.name}"):
        stem = Path(row["file_name"]).stem
        image_path = generated_by_stem.get(stem)
        if image_path is None:
            missing.append(row["file_name"])
            continue

        image = open_image(image_path)
        width, height = image.size
        labels = [label for label in row["caption"][1:] if label]
        boxes = row["bndboxes"][: len(labels)]
        difficulty = row.get("zero_shot_difficulty", "unknown")

        for obj_idx, (label, box) in enumerate(zip(labels, boxes)):
            if not valid_box(box):
                continue
            queries = make_queries(label)
            cache_key = (str(image_path), label)
            if cache_key not in detection_cache:
                inputs = processor(images=image, text=[queries], return_tensors="pt")
                inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
                target_sizes = torch.tensor([[height, width]], device=device)
                with torch.no_grad():
                    outputs = model(**inputs)
                result = post_process(processor, outputs, target_sizes, args.threshold, queries)
                detections = result_to_detections(result)
                detection_cache[cache_key] = detections
            else:
                detections = detection_cache[cache_key]

            gt = normalized_to_pixel(box, width, height)
            best = None
            for det in detections:
                iou = box_iou(gt, det["box"])
                if best is None or iou > best["iou"]:
                    best = {**det, "iou": iou}

            if best is None:
                best_iou = 0.0
                best_score = None
                c_err = None
                best_box = None
            else:
                best_iou = float(best["iou"])
                best_score = float(best["score"])
                c_err = float(center_error(gt, best["box"], width, height))
                best_box = best["box"]

            sample_rows.append(
                {
                    "file_name": image_path.name,
                    "metadata_file": row["file_name"],
                    "object_index": obj_idx,
                    "class": label,
                    "difficulty": difficulty,
                    "detected": best is not None,
                    "num_detections": len(detections),
                    "best_iou": best_iou,
                    "recall_iou30": best_iou >= 0.30,
                    "recall_iou50": best_iou >= 0.50,
                    "center_error": c_err,
                    "best_score": best_score,
                    "gt_box": gt,
                    "best_box": best_box,
                }
            )

    summary = summarize(sample_rows)
    summary.update(
        {
            "num_metadata": len(metadata),
            "num_objects": len(sample_rows),
            "num_missing_generated": len(missing),
            "missing_generated": missing[:20],
            "model_id": args.model_id,
            "threshold": args.threshold,
        }
    )

    (args.output_dir / "open_detector_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (args.output_dir / "open_detector_samples.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "file_name",
            "metadata_file",
            "object_index",
            "class",
            "difficulty",
            "detected",
            "num_detections",
            "best_iou",
            "recall_iou30",
            "recall_iou50",
            "center_error",
            "best_score",
            "gt_box",
            "best_box",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sample_rows)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
