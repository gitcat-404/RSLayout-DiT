#!/usr/bin/env python
"""Build a 300-sample zero-shot layout set for DIOR-RSVG generalization tests."""

import argparse
import json
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rslayout_utils.layout_condition import parse_layout_objects
from rslayout_utils.prompting import build_satellite_prompts, legacy_clip_prompt


UNSEEN_CLASSES = [
    "solar-panel",
    "parking-lot",
    "roundabout",
    "railway-track",
    "container-yard",
    "pier",
    "swimming-pool",
    "helipad",
    "oil-refinery",
    "greenhouse",
]

RENDER_ALIASES = {
    "solar-panel": "storagetank",
    "parking-lot": "vehicle",
    "roundabout": "roundabout",
    "railway-track": "trainstation",
    "container-yard": "harbor",
    "pier": "harbor",
    "swimming-pool": "swimming-pool",
    "helipad": "airport",
    "oil-refinery": "chimney",
    "greenhouse": "golffield",
}

DISPLAY_NAMES = {
    "solar-panel": "solar panel",
    "parking-lot": "parking lot",
    "roundabout": "roundabout",
    "railway-track": "railway track",
    "container-yard": "container yard",
    "pier": "pier",
    "swimming-pool": "swimming pool",
    "helipad": "helipad",
    "oil-refinery": "oil refinery",
    "greenhouse": "greenhouse",
}

SCENES = {
    "solar-panel": "solar energy farm",
    "parking-lot": "urban parking area",
    "roundabout": "urban road junction",
    "railway-track": "railway corridor",
    "container-yard": "port logistics yard",
    "pier": "coastal harbor area",
    "swimming-pool": "urban residential area",
    "helipad": "airport emergency area",
    "oil-refinery": "industrial refinery area",
    "greenhouse": "agricultural greenhouse area",
}

SHAPES = {
    "solar-panel": (0.10, 0.040),
    "parking-lot": (0.24, 0.16),
    "roundabout": (0.16, 0.16),
    "railway-track": (0.52, 0.040),
    "container-yard": (0.28, 0.18),
    "pier": (0.34, 0.060),
    "swimming-pool": (0.11, 0.060),
    "helipad": (0.13, 0.13),
    "oil-refinery": (0.22, 0.18),
    "greenhouse": (0.20, 0.070),
}

NUMBER_WORDS = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
}


def parse_args():
    parser = argparse.ArgumentParser("Create a zero-shot DIOR-RSVG layout metadata set.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="path_to_data/DIOR_RSVG_zeroshot_300/test",
        help="Output split directory containing metadata.jsonl and metadata_rslayout_prompts.jsonl.",
    )
    parser.add_argument("--count_per_class", type=int, default=30)
    parser.add_argument("--max_objects", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument("--write_previews", action="store_true", help="Also render layout preview PNGs.")
    return parser.parse_args()


def display_name(label: str) -> str:
    return DISPLAY_NAMES.get(label, label.replace("-", " "))


def pluralize(name: str, count: int) -> str:
    if count == 1:
        return name
    if name.endswith("y"):
        return name[:-1] + "ies"
    if name.endswith(("s", "x", "ch", "sh")):
        return name + "es"
    return name + "s"


def rotated_box(cx: float, cy: float, w: float, h: float, angle: float) -> List[float]:
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    corners = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    pts = []
    for x, y in corners:
        px = cx + x * cos_a - y * sin_a
        py = cy + x * sin_a + y * cos_a
        pts.extend([px, py])
    return pts


def bbox_from_obbox(obbox: Sequence[float]) -> List[float]:
    xs = [float(v) for v in obbox[0::2]]
    ys = [float(v) for v in obbox[1::2]]
    return [min(xs), min(ys), max(xs), max(ys)]


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(ix1 - ix0, 0.0), max(iy1 - iy0, 0.0)
    inter = iw * ih
    area_a = max(ax1 - ax0, 0.0) * max(ay1 - ay0, 0.0)
    area_b = max(bx1 - bx0, 0.0) * max(by1 - by0, 0.0)
    return inter / max(area_a + area_b - inter, 1e-8)


def sample_object(label: str, rng: random.Random, existing_bboxes: Sequence[Sequence[float]]):
    base_w, base_h = SHAPES[label]
    for _ in range(200):
        scale = rng.uniform(0.75, 1.30)
        w = min(base_w * scale, 0.85)
        h = min(base_h * scale, 0.65)
        angle = rng.uniform(-math.pi / 2, math.pi / 2)
        margin = 0.08
        if label in {"pier", "railway-track"}:
            angle = rng.choice([0.0, math.pi / 2, rng.uniform(-0.25, 0.25)])
        cx = rng.uniform(margin + w / 2, 1.0 - margin - w / 2)
        cy = rng.uniform(margin + h / 2, 1.0 - margin - h / 2)
        if label == "pier":
            cy = rng.choice([rng.uniform(0.10, 0.25), rng.uniform(0.75, 0.90)])
        obbox = rotated_box(cx, cy, w, h, angle)
        if min(obbox) < 0.02 or max(obbox) > 0.98:
            continue
        bbox = bbox_from_obbox(obbox)
        if all(bbox_iou(bbox, prev) < 0.08 for prev in existing_bboxes):
            return bbox, obbox
    return bbox, obbox


def sample_labels(main_label: str, local_index: int, rng: random.Random) -> Tuple[str, List[str]]:
    if local_index < 10:
        return "single-object", [main_label]
    if local_index < 20:
        return "multi-instance", [main_label] * rng.randint(2, 5)
    other_pool = [label for label in UNSEEN_CLASSES if label != main_label]
    labels = [main_label] + rng.sample(other_pool, k=rng.randint(1, 3))
    return "unseen-composition", labels


def compact_prompt(labels: Sequence[str]) -> str:
    counts = Counter(display_name(label) for label in labels)
    parts = [
        f"{NUMBER_WORDS.get(count, str(count))} {pluralize(name, count)}"
        for name, count in counts.items()
    ]
    if len(parts) == 1:
        summary = parts[0]
    elif len(parts) == 2:
        summary = f"{parts[0]} and {parts[1]}"
    else:
        summary = ", ".join(parts[:-1]) + f", and {parts[-1]}"
    return f"A satellite image of a remote sensing scene, showing {summary}."


def build_record(sample_id: int, main_label: str, local_index: int, rng: random.Random, max_objects: int):
    difficulty, labels = sample_labels(main_label, local_index, rng)
    labels = labels[:max_objects]
    bboxes = []
    obboxes = []
    for label in labels:
        bbox, obbox = sample_object(label, rng, bboxes)
        bboxes.append([round(v, 6) for v in bbox])
        obboxes.append([round(v, 6) for v in obbox])

    while len(bboxes) < max_objects:
        bboxes.append([0, 0, 0, 0])
        obboxes.append([0, 0, 0, 0, 0, 0, 0, 0])

    caption = [f"This is an aerial image of {SCENES[main_label]}. "] + labels
    while len(caption) < max_objects + 1:
        caption.append("")

    file_name = f"zeroshot_{sample_id:06d}_{main_label}_{local_index:02d}.png"
    return {
        "file_name": file_name,
        "caption": caption,
        "bndboxes": bboxes,
        "obboxes": obboxes,
        "zero_shot_main_class": main_label,
        "zero_shot_difficulty": difficulty,
        "render_aliases": {label: RENDER_ALIASES[label] for label in sorted(set(labels))},
    }


def write_prompt_record(sample: Dict) -> Dict:
    objects = parse_layout_objects(sample["caption"], sample["bndboxes"], sample["obboxes"])
    old_t5 = sample["caption"][0]
    old_clip = legacy_clip_prompt(old_t5, objects)
    new_clip, new_t5 = build_satellite_prompts(sample["caption"], objects)
    # Keep the CLIP prompt concise and explicitly open-category.
    labels = [obj.label for obj in objects if obj.valid]
    new_clip = compact_prompt(labels)
    return {
        "file_name": sample["file_name"],
        "objects": labels,
        "old_clip_prompt": old_clip,
        "old_t5_prompt": old_t5,
        "new_clip_prompt": new_clip,
        "new_t5_prompt": new_t5,
    }


def maybe_write_previews(output_dir: Path, samples: Sequence[Dict]):
    from rslayout_dit.layout_render import render_layout_with_labels

    preview_dir = output_dir / "layout_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        objects = parse_layout_objects(sample["caption"], sample["bndboxes"], sample["obboxes"])
        image = render_layout_with_labels(objects, resolution=512)
        image.save(preview_dir / sample["file_name"].replace(".jpg", ".png"))


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    sample_id = 0
    for label in UNSEEN_CLASSES:
        for local_index in range(args.count_per_class):
            samples.append(build_record(sample_id, label, local_index, rng, args.max_objects))
            sample_id += 1

    metadata_path = output_dir / "metadata.jsonl"
    prompt_path = output_dir / "metadata_rslayout_prompts.jsonl"
    with metadata_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    with prompt_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(write_prompt_record(sample), ensure_ascii=False) + "\n")

    design = {
        "name": "DIOR-RSVG zero-shot layout generalization set",
        "num_samples": len(samples),
        "classes": UNSEEN_CLASSES,
        "count_per_class": args.count_per_class,
        "difficulty_protocol": {
            "0-9": "single-object",
            "10-19": "multi-instance of the same unseen class",
            "20-29": "composition of multiple unseen classes",
        },
        "render_aliases": RENDER_ALIASES,
        "notes": "Labels remain unseen in text prompts. Unsupported labels reuse nearest supported layout colors through CLASS_ALIASES.",
    }
    with (output_dir / "zero_shot_design.json").open("w", encoding="utf-8") as f:
        json.dump(design, f, ensure_ascii=False, indent=2)

    if args.write_previews:
        maybe_write_previews(output_dir, samples)

    print(f"Wrote {len(samples)} samples")
    print(f"metadata: {metadata_path}")
    print(f"prompts:  {prompt_path}")
    print(f"design:   {output_dir / 'zero_shot_design.json'}")


if __name__ == "__main__":
    main()
