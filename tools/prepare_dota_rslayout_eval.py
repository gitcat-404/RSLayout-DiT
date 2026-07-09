#!/usr/bin/env python
import argparse
import json
import os
import re
import shutil
from pathlib import Path


DOTA_CLASSES = [
    "plane",
    "ship",
    "storage-tank",
    "baseball-diamond",
    "tennis-court",
    "basketball-court",
    "ground-track-field",
    "harbor",
    "bridge",
    "large-vehicle",
    "small-vehicle",
    "helicopter",
    "roundabout",
    "soccer-ball-field",
    "swimming-pool",
]

CLASS_TO_ID = {name: idx for idx, name in enumerate(DOTA_CLASSES)}
DIOR_CLASSES = [
    "Expressway-Service-area",
    "Expressway-toll-station",
    "airplane",
    "airport",
    "baseballfield",
    "basketballcourt",
    "bridge",
    "chimney",
    "dam",
    "golffield",
    "groundtrackfield",
    "harbor",
    "overpass",
    "ship",
    "stadium",
    "storagetank",
    "tenniscourt",
    "trainstation",
    "vehicle",
    "windmill",
]
DIOR_CLASS_TO_ID = {name: idx for idx, name in enumerate(DIOR_CLASSES)}
DOTA_TO_DIOR_PROXY = {
    "plane": "airplane",
    "ship": "ship",
    "storage-tank": "storagetank",
    "baseball-diamond": "baseballfield",
    "tennis-court": "tenniscourt",
    "basketball-court": "basketballcourt",
    "ground-track-field": "groundtrackfield",
    "harbor": "harbor",
    "bridge": "bridge",
    "large-vehicle": "vehicle",
    "small-vehicle": "vehicle",
}
GENERATED_RE = re.compile(r"^(?P<index>\d+)_(?P<image_id>.+)_generated\.png$")
LAYOUTDIFFUSION_RE = re.compile(r"^(?P<image_id>.+)_(?P<sample_idx>\d+)\.png$")


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def link_or_copy(src: Path, dst: Path, copy: bool = False):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def class_id(label: str) -> int:
    if label not in CLASS_TO_ID:
        raise KeyError(f"Unknown DOTA class {label!r}")
    return CLASS_TO_ID[label]


def make_yolo_hbb_label(row):
    lines = []
    for label, box in zip(row.get("caption", [])[1:], row.get("bndboxes", [])):
        if not label:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        if w <= 0 or h <= 0:
            continue
        xc = x1 + w / 2.0
        yc = y1 + h / 2.0
        lines.append(f"{class_id(label)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return "\n".join(lines)


def make_yolo_obb_label(row):
    lines = []
    for label, obbox in zip(row.get("caption", [])[1:], row.get("obboxes", [])):
        if not label:
            continue
        coords = [min(1.0, max(0.0, float(v))) for v in obbox]
        if len(coords) != 8 or not any(abs(v) > 1e-8 for v in coords):
            continue
        lines.append(f"{class_id(label)} " + " ".join(f"{v:.6f}" for v in coords))
    return "\n".join(lines)


def make_yolo_hbb_proxy_label(row):
    lines = []
    skipped = 0
    for label, box in zip(row.get("caption", [])[1:], row.get("bndboxes", [])):
        proxy_label = DOTA_TO_DIOR_PROXY.get(label)
        if not proxy_label:
            if label:
                skipped += 1
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        if w <= 0 or h <= 0:
            continue
        xc = x1 + w / 2.0
        yc = y1 + h / 2.0
        lines.append(f"{DIOR_CLASS_TO_ID[proxy_label]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return "\n".join(lines), skipped


def make_yolo_obb_proxy_label(row):
    lines = []
    skipped = 0
    for label, obbox in zip(row.get("caption", [])[1:], row.get("obboxes", [])):
        proxy_label = DOTA_TO_DIOR_PROXY.get(label)
        if not proxy_label:
            if label:
                skipped += 1
            continue
        coords = [min(1.0, max(0.0, float(v))) for v in obbox]
        if len(coords) != 8 or not any(abs(v) > 1e-8 for v in coords):
            continue
        lines.append(f"{DIOR_CLASS_TO_ID[proxy_label]} " + " ".join(f"{v:.6f}" for v in coords))
    return "\n".join(lines), skipped


def write_yaml(path: Path, root: Path, classes):
    names = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(classes))
    path.write_text(
        f"path: {root.resolve()}\n"
        "train: images/val\n"
        "val: images/val\n"
        "names:\n"
        f"{names}\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--real-image-dir", type=Path, required=True)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--copy", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(args.metadata)

    generated_by_file = {}
    for path in args.generated_dir.glob("*_generated.png"):
        match = GENERATED_RE.match(path.name)
        if not match:
            continue
        image_id = match.group("image_id")
        for suffix in (".png", ".jpg", ".jpeg"):
            generated_by_file[f"{image_id}{suffix}"] = path
    for path in list(args.generated_dir.glob("*.png")) + list(args.generated_dir.glob("*.jpg")) + list(args.generated_dir.glob("*.jpeg")):
        if GENERATED_RE.match(path.name):
            continue
        generated_by_file.setdefault(path.name, path)
        layoutdiffusion_match = LAYOUTDIFFUSION_RE.match(path.name)
        if layoutdiffusion_match:
            image_id = layoutdiffusion_match.group("image_id")
            generated_by_file.setdefault(f"{image_id}.png", path)
            continue

    out = args.output_dir
    real_dir = out / "real"
    generated_dir = out / "generated"
    yolo_hbb = out / "yolo_hbb"
    yolo_obb = out / "yolo_obb"
    yolo_hbb_dior_proxy = out / "yolo_hbb_dior_proxy"
    yolo_obb_dior_proxy = out / "yolo_obb_dior_proxy"

    for directory in [
        real_dir,
        generated_dir,
        yolo_hbb / "images" / "val",
        yolo_hbb / "labels" / "val",
        yolo_obb / "images" / "val",
        yolo_obb / "labels" / "val",
        yolo_hbb_dior_proxy / "images" / "val",
        yolo_hbb_dior_proxy / "labels" / "val",
        yolo_obb_dior_proxy / "images" / "val",
        yolo_obb_dior_proxy / "labels" / "val",
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    clip_rows = []
    missing_generated = []
    missing_real = []
    skipped_proxy_objects = 0
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

        link_or_copy(generated_src, generated_dir / generated_name, copy=args.copy)
        link_or_copy(real_src, real_dir / file_name, copy=args.copy)

        for yolo_root in (yolo_hbb, yolo_obb, yolo_hbb_dior_proxy, yolo_obb_dior_proxy):
            link_or_copy(generated_src, yolo_root / "images" / "val" / generated_name, copy=args.copy)

        (yolo_hbb / "labels" / "val" / f"{stem}.txt").write_text(
            make_yolo_hbb_label(row) + "\n",
            encoding="utf-8",
        )
        (yolo_obb / "labels" / "val" / f"{stem}.txt").write_text(
            make_yolo_obb_label(row) + "\n",
            encoding="utf-8",
        )
        hbb_proxy, skipped_hbb = make_yolo_hbb_proxy_label(row)
        obb_proxy, skipped_obb = make_yolo_obb_proxy_label(row)
        skipped_proxy_objects += max(skipped_hbb, skipped_obb)
        (yolo_hbb_dior_proxy / "labels" / "val" / f"{stem}.txt").write_text(
            hbb_proxy + "\n",
            encoding="utf-8",
        )
        (yolo_obb_dior_proxy / "labels" / "val" / f"{stem}.txt").write_text(
            obb_proxy + "\n",
            encoding="utf-8",
        )

        clip_row = dict(row)
        clip_row["file_name"] = generated_name
        clip_rows.append(clip_row)

    with (out / "metadata_clip.jsonl").open("w", encoding="utf-8") as handle:
        for row in clip_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_yaml(yolo_hbb / "config.yaml", yolo_hbb, DOTA_CLASSES)
    write_yaml(yolo_obb / "config.yaml", yolo_obb, DOTA_CLASSES)
    write_yaml(yolo_hbb_dior_proxy / "config.yaml", yolo_hbb_dior_proxy, DIOR_CLASSES)
    write_yaml(yolo_obb_dior_proxy / "config.yaml", yolo_obb_dior_proxy, DIOR_CLASSES)

    summary = {
        "metadata_rows": len(rows),
        "prepared_rows": len(clip_rows),
        "missing_generated": len(missing_generated),
        "missing_real": len(missing_real),
        "skipped_proxy_objects": skipped_proxy_objects,
        "output_dir": str(out.resolve()),
        "dota_classes": DOTA_CLASSES,
        "dior_proxy_classes": DIOR_CLASSES,
        "dior_proxy_note": "YOLO checkpoint is DIOR-20; proxy labels map overlapping DOTA classes to DIOR classes and skip unmapped classes.",
    }
    (out / "prepare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
