#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO


def read_names(config_path: Path):
    try:
        import yaml

        data = yaml.safe_load(config_path.read_text())
        names = data.get("names", {})
        return {int(k): v for k, v in names.items()}
    except Exception:
        return {}


def xywh_to_xyxy(box):
    x, y, w, h = box
    return np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2], dtype=np.float32)


def box_iou_one_to_many(box, boxes):
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.float32)
    boxes = np.asarray(boxes, dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = max(0.0, (box[2] - box[0])) * max(0.0, (box[3] - box[1]))
    area2 = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - inter + 1e-9
    return inter / union


def ap_from_pr(tp, fp, n_gt):
    if n_gt == 0:
        return None
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    recall = tp / (n_gt + 1e-9)
    precision = tp / np.maximum(tp + fp, 1e-9)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def size_group(width_px, height_px):
    area = width_px * height_px
    if area < 32 * 32:
        return "small"
    if area < 96 * 96:
        return "medium"
    return "large"


def load_ground_truth(labels_dir: Path, image_size: int):
    gts = defaultdict(lambda: defaultdict(list))
    image_to_classes = defaultdict(list)
    for label_path in sorted(labels_dir.glob("*.txt")):
        stem = label_path.stem
        for line in label_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            xywh = np.array([float(x) for x in parts[1:5]], dtype=np.float32)
            group = size_group(xywh[2] * image_size, xywh[3] * image_size)
            gts[group][cls].append(
                {
                    "image": stem,
                    "box": xywh_to_xyxy(xywh),
                    "matched": False,
                }
            )
            image_to_classes[stem].append((cls, group, xywh_to_xyxy(xywh)))
    return gts, image_to_classes


def run_predictions(weights: Path, images_dir: Path, imgsz: int, batch: int, conf: float, device: str):
    model = YOLO(str(weights))
    images = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    predictions = defaultdict(list)
    for result in model.predict(
        source=[str(p) for p in images],
        imgsz=imgsz,
        batch=batch,
        conf=conf,
        iou=0.7,
        device=device,
        verbose=False,
        stream=True,
    ):
        stem = Path(result.path).stem
        if result.boxes is None:
            continue
        h, w = result.orig_shape
        xyxy = result.boxes.xyxy.cpu().numpy()
        cls = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()
        for box, c, score in zip(xyxy, cls, confs):
            norm = np.array([box[0] / w, box[1] / h, box[2] / w, box[3] / h], dtype=np.float32)
            predictions[c].append({"image": stem, "box": norm, "conf": float(score)})
    return predictions


def eval_group(gts_all, image_to_classes, predictions, group, iou_threshold):
    aps = []
    for cls, gt_list in gts_all[group].items():
        n_gt = len(gt_list)
        matched = {(gt["image"], idx): False for idx, gt in enumerate(gt_list)}
        gt_by_image = defaultdict(list)
        for idx, gt in enumerate(gt_list):
            gt_by_image[gt["image"]].append((idx, gt["box"]))

        dets = sorted(predictions.get(cls, []), key=lambda x: x["conf"], reverse=True)
        tp, fp = [], []
        for det in dets:
            same_image_group_gt = gt_by_image.get(det["image"], [])
            other_group_boxes = [
                box
                for c, g, box in image_to_classes.get(det["image"], [])
                if c == cls and g != group
            ]
            group_boxes = [box for _, box in same_image_group_gt]
            group_ious = box_iou_one_to_many(det["box"], group_boxes)
            other_ious = box_iou_one_to_many(det["box"], other_group_boxes)

            best_other = float(other_ious.max()) if len(other_ious) else 0.0
            best_group = float(group_ious.max()) if len(group_ious) else 0.0
            if best_other >= iou_threshold and best_other > best_group:
                continue

            if best_group >= iou_threshold:
                best_pos = int(group_ious.argmax())
                gt_idx = same_image_group_gt[best_pos][0]
                key = (det["image"], gt_idx)
                if not matched[key]:
                    matched[key] = True
                    tp.append(1.0)
                    fp.append(0.0)
                else:
                    tp.append(0.0)
                    fp.append(1.0)
            else:
                tp.append(0.0)
                fp.append(1.0)
        ap = ap_from_pr(np.array(tp), np.array(fp), n_gt)
        if ap is not None:
            aps.append(ap)
    return float(np.mean(aps)) if aps else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--yolo-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--label-image-size", type=int, default=512)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = args.yolo_dir / "images" / "val"
    labels_dir = args.yolo_dir / "labels" / "val"
    names = read_names(args.yolo_dir / "config.yaml")

    gts_all, image_to_classes = load_ground_truth(labels_dir, args.label_image_size)
    predictions = run_predictions(args.weights, images_dir, args.imgsz, args.batch, args.conf, args.device)

    thresholds = np.arange(0.5, 0.96, 0.05)
    rows = []
    for group in ["small", "medium", "large"]:
        map50 = eval_group(gts_all, image_to_classes, predictions, group, 0.5)
        maps = [eval_group(gts_all, image_to_classes, predictions, group, float(t)) for t in thresholds]
        n_instances = sum(len(v) for v in gts_all[group].values())
        rows.append(
            {
                "size": group,
                "instances": n_instances,
                "map50": map50,
                "map5095": float(np.mean(maps)),
            }
        )

    with (args.out_dir / "sizewise_yoloscore.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (args.out_dir / "sizewise_yoloscore.json").write_text(json.dumps({"names": names, "rows": rows}, indent=2))

    x = np.arange(len(rows))
    width = 0.38
    plt.figure(figsize=(5.8, 3.8))
    plt.bar(x - width / 2, [r["map50"] for r in rows], width, label="mAP50")
    plt.bar(x + width / 2, [r["map5095"] for r in rows], width, label="mAP50-95")
    plt.xticks(x, [r["size"] for r in rows])
    plt.ylabel("YOLOScore")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(args.out_dir / "sizewise_yoloscore.png", dpi=300)
    plt.close()

    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
