#!/usr/bin/env python
"""Evaluate CLIP Local/Global for zero-shot layout generations without FID."""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rslayout_utils.prompting import display_class


GENERATED_RE = re.compile(r"^(?P<index>\d+)_(?P<stem>.+)_generated\.png$")


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def valid_box(box):
    x0, y0, x1, y1 = [float(v) for v in box]
    return x1 > x0 and y1 > y0


def crop_box(image: Image.Image, box):
    x0, y0, x1, y1 = [float(v) for v in box]
    width, height = image.size
    left = max(0, min(width - 1, int(round(x0 * width))))
    top = max(0, min(height - 1, int(round(y0 * height))))
    right = max(left + 1, min(width, int(round(x1 * width))))
    bottom = max(top + 1, min(height, int(round(y1 * height))))
    return image.crop((left, top, right, bottom))


def mean_or_nan(values):
    return float(np.mean(values)) if values else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--prompt-metadata", type=Path, required=True)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", type=str, default="ViT-L-14")
    parser.add_argument("--pretrained", type=str, default="laion2b-s32b-b82k")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(args.metadata)
    prompt_rows = load_jsonl(args.prompt_metadata)
    prompts_by_file = {row["file_name"]: row for row in prompt_rows}

    generated_by_stem = {}
    for path in args.generated_dir.glob("*_generated.png"):
        match = GENERATED_RE.match(path.name)
        if match:
            generated_by_stem[match.group("stem")] = path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(args.model_name, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model_name)
    model = model.to(device).eval()

    text_cache = {}

    def text_features(text: str):
        if text not in text_cache:
            with torch.no_grad():
                tokens = tokenizer([text]).to(device)
                feats = model.encode_text(tokens)
                text_cache[text] = F.normalize(feats, dim=-1)
        return text_cache[text]

    def image_features(image: Image.Image):
        with torch.no_grad():
            tensor = preprocess(image).unsqueeze(0).to(device)
            feats = model.encode_image(tensor)
            return F.normalize(feats, dim=-1)

    sample_rows = []
    per_class = defaultdict(lambda: {"local": [], "global": []})
    per_difficulty = defaultdict(lambda: {"local": [], "global": []})
    local_all = []
    global_all = []
    missing = []

    for row in tqdm(rows, desc="CLIP zero-shot"):
        file_name = row["file_name"]
        stem = Path(file_name).stem
        image_path = generated_by_stem.get(stem)
        prompt_row = prompts_by_file.get(file_name, {})
        if image_path is None:
            missing.append(file_name)
            continue

        image = Image.open(image_path).convert("RGB")
        global_prompt = prompt_row.get("new_clip_prompt") or row["caption"][0]
        global_score = float((image_features(image) * text_features(global_prompt)).sum(dim=-1).item())
        global_all.append(global_score)

        labels = [label for label in row["caption"][1:] if label]
        boxes = row["bndboxes"][: len(labels)]
        local_scores = []
        for label, box in zip(labels, boxes):
            if not valid_box(box):
                continue
            crop = crop_box(image, box)
            local_text = display_class(label)
            score = float((image_features(crop) * text_features(local_text)).sum(dim=-1).item())
            local_scores.append(score)
            per_class[label]["local"].append(score)
            per_class[label]["global"].append(global_score)

        if local_scores:
            local_score = mean_or_nan(local_scores)
            local_all.append(local_score)
        else:
            local_score = float("nan")

        difficulty = row.get("zero_shot_difficulty", "unknown")
        per_difficulty[difficulty]["local"].append(local_score)
        per_difficulty[difficulty]["global"].append(global_score)
        sample_rows.append(
            {
                "file_name": image_path.name,
                "main_class": row.get("zero_shot_main_class", ""),
                "difficulty": difficulty,
                "num_objects": len(labels),
                "clip_local": local_score,
                "clip_global": global_score,
                "global_prompt": global_prompt,
            }
        )

    summary = {
        "num_metadata": len(rows),
        "num_evaluated": len(sample_rows),
        "num_missing_generated": len(missing),
        "clip_local": mean_or_nan(local_all),
        "clip_global": mean_or_nan(global_all),
        "fid": None,
        "fid_note": "Not computed because this zero-shot set has no paired/reference real images.",
        "yoloscore": None,
        "yoloscore_note": "Not computed as closed-set mAP because the DIOR YOLOScore detector does not cover these open-set categories.",
        "per_class": {
            key: {
                "n": len(value["local"]),
                "clip_local": mean_or_nan(value["local"]),
                "clip_global": mean_or_nan(value["global"]),
            }
            for key, value in sorted(per_class.items())
        },
        "per_difficulty": {
            key: {
                "n": len([v for v in value["local"] if not np.isnan(v)]),
                "clip_local": mean_or_nan([v for v in value["local"] if not np.isnan(v)]),
                "clip_global": mean_or_nan(value["global"]),
            }
            for key, value in sorted(per_difficulty.items())
        },
        "missing_generated": missing[:20],
    }

    (args.output_dir / "zeroshot_clip_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with (args.output_dir / "zeroshot_clip_samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sample_rows[0].keys()) if sample_rows else [])
        if sample_rows:
            writer.writeheader()
            writer.writerows(sample_rows)

    with (args.output_dir / "zeroshot_clip_per_class.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class", "n", "clip_local", "clip_global"])
        writer.writeheader()
        for key, value in summary["per_class"].items():
            writer.writerow({"class": key, **value})

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
