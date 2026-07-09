#!/usr/bin/env python
import argparse
import json
import os
import re
import shutil
from pathlib import Path

from PIL import Image


YOLO_CLASSES = {
    "Expressway-Service-area": 0,
    "Expressway-toll-station": 1,
    "airplane": 2,
    "airport": 3,
    "baseballfield": 4,
    "basketballcourt": 5,
    "bridge": 6,
    "chimney": 7,
    "dam": 8,
    "golffield": 9,
    "groundtrackfield": 10,
    "harbor": 11,
    "overpass": 12,
    "ship": 13,
    "stadium": 14,
    "storagetank": 15,
    "tenniscourt": 16,
    "trainstation": 17,
    "vehicle": 18,
    "windmill": 19,
}


GENERATED_RE = re.compile(r"^(?P<index>\d+)_(?P<image_id>.+)_generated\.png$")


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def link_or_copy(src, dst, copy=False):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def make_yolo_label(row):
    lines = []
    labels = row.get("caption", [])[1:]
    boxes = row.get("bndboxes", [])
    for label, box in zip(labels, boxes):
        if not label:
            continue
        if label not in YOLO_CLASSES:
            raise KeyError(f"Unknown class {label!r} in {row['file_name']}")
        x1, y1, x2, y2 = box
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        if w <= 0 or h <= 0:
            continue
        xc = x1 + w / 2
        yc = y1 + h / 2
        lines.append(f"{YOLO_CLASSES[label]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--real-image-dir", type=Path, required=True)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--copy", action="store_true", help="Copy images instead of symlinking them.")
    args = parser.parse_args()

    rows = load_jsonl(args.metadata)
    rows_by_file = {row["file_name"]: row for row in rows}

    generated_by_file = {}
    for path in args.generated_dir.glob("*_generated.png"):
        match = GENERATED_RE.match(path.name)
        if not match:
            continue
        file_name = f"{match.group('image_id')}.jpg"
        generated_by_file[file_name] = path

    out = args.output_dir
    real_dir = out / "real"
    generated_dir = out / "generated"
    yolo_img_dir = out / "yolo" / "images" / "val"
    yolo_label_dir = out / "yolo" / "labels" / "val"
    for directory in [real_dir, generated_dir, yolo_img_dir, yolo_label_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    clip_rows = []
    missing_generated = []
    missing_real = []
    for row in rows:
        file_name = row["file_name"]
        generated_src = generated_by_file.get(file_name)
        real_src = args.real_image_dir / file_name
        if generated_src is None:
            missing_generated.append(file_name)
            continue
        if not real_src.exists():
            missing_real.append(file_name)
            continue

        stem = Path(file_name).stem
        generated_name = f"{stem}.png"
        real_name = file_name

        generated_dst = generated_dir / generated_name
        real_dst = real_dir / real_name
        yolo_image_dst = yolo_img_dir / generated_name
        yolo_label_dst = yolo_label_dir / f"{stem}.txt"

        link_or_copy(generated_src, generated_dst, copy=args.copy)
        link_or_copy(real_src, real_dst, copy=args.copy)
        link_or_copy(generated_src, yolo_image_dst, copy=args.copy)
        yolo_label_dst.write_text(make_yolo_label(row) + "\n", encoding="utf-8")

        clip_row = dict(row)
        clip_row["file_name"] = generated_name
        clip_rows.append(clip_row)

    with (out / "metadata_clip.jsonl").open("w", encoding="utf-8") as handle:
        for row in clip_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    data_yaml = out / "yolo" / "config.yaml"
    names = "\n".join(f"  {idx}: {name}" for name, idx in sorted(YOLO_CLASSES.items(), key=lambda item: item[1]))
    data_yaml.write_text(
        f"path: {str((out / 'yolo').resolve())}\n"
        "train: images/val\n"
        "val: images/val\n"
        "names:\n"
        f"{names}\n",
        encoding="utf-8",
    )

    summary = {
        "metadata_rows": len(rows),
        "prepared_rows": len(clip_rows),
        "missing_generated": len(missing_generated),
        "missing_real": len(missing_real),
        "output_dir": str(out.resolve()),
    }
    (out / "prepare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
