#!/usr/bin/env python3
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

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
    return inter / (area1 + area2 - inter + 1e-9)


def ap_from_pr(tp, fp, n_gt):
    if n_gt == 0:
        return None
    if len(tp) == 0:
        return 0.0
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


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def training_class_counts(metadata_path: Path, names_by_id):
    name_to_id = {name: idx for idx, name in names_by_id.items()}
    counts = Counter()
    for row in load_jsonl(metadata_path):
        for label in row.get("caption", [])[1:]:
            if label in name_to_id:
                counts[name_to_id[label]] += 1
    return counts


def split_frequency_groups(class_counts):
    classes = [cls for cls, count in class_counts.items() if count > 0]
    classes = sorted(classes, key=lambda cls: (class_counts[cls], cls))
    n = len(classes)
    if n == 0:
        return {"rare": set(), "common": set(), "frequent": set()}
    rare_n = max(1, int(round(n * 0.3)))
    frequent_n = max(1, int(round(n * 0.3)))
    rare = set(classes[:rare_n])
    frequent = set(classes[-frequent_n:])
    common = set(classes[rare_n : n - frequent_n])
    return {"rare": rare, "common": common, "frequent": frequent}


def load_ground_truth(labels_dir: Path, label_image_size: int):
    gts_by_class = defaultdict(list)
    image_gt_index = defaultdict(list)
    for label_path in sorted(labels_dir.glob("*.txt")):
        stem = label_path.stem
        for line in label_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            xywh = np.array([float(x) for x in parts[1:5]], dtype=np.float32)
            box = xywh_to_xyxy(xywh)
            group = size_group(xywh[2] * label_image_size, xywh[3] * label_image_size)
            record = {
                "image": stem,
                "class": cls,
                "box": box,
                "size": group,
                "area": float(xywh[2] * xywh[3]),
            }
            gts_by_class[cls].append(record)
            image_gt_index[stem].append(record)
    return gts_by_class, image_gt_index


def run_predictions(weights: Path, images_dir: Path, imgsz: int, batch: int, conf: float, device: str):
    model = YOLO(str(weights))
    preds_by_class = defaultdict(list)
    for result in model.predict(
        source=str(images_dir),
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
            preds_by_class[c].append({"image": stem, "class": int(c), "box": norm, "conf": float(score)})
    return preds_by_class


def make_gt_filter(metric_type, value):
    if metric_type == "all":
        return lambda gt: True
    if metric_type == "size":
        return lambda gt: gt["size"] == value
    if metric_type == "freq":
        freq_classes = value
        return lambda gt: gt["class"] in freq_classes
    raise ValueError(metric_type)


def evaluate_subset(gts_by_class, image_gt_index, preds_by_class, gt_filter, iou_threshold):
    aps = []
    for cls, all_gt in gts_by_class.items():
        gt_list = [gt for gt in all_gt if gt_filter(gt)]
        if not gt_list:
            continue
        gt_by_image = defaultdict(list)
        for idx, gt in enumerate(gt_list):
            gt_by_image[gt["image"]].append((idx, gt["box"]))
        matched = {(gt["image"], idx): False for idx, gt in enumerate(gt_list)}
        tp, fp = [], []
        for det in sorted(preds_by_class.get(cls, []), key=lambda x: x["conf"], reverse=True):
            subset_gts = gt_by_image.get(det["image"], [])
            subset_boxes = [box for _, box in subset_gts]
            subset_ious = box_iou_one_to_many(det["box"], subset_boxes)
            best_subset = float(subset_ious.max()) if len(subset_ious) else 0.0

            other_boxes = [
                gt["box"]
                for gt in image_gt_index.get(det["image"], [])
                if gt["class"] == cls and not gt_filter(gt)
            ]
            other_ious = box_iou_one_to_many(det["box"], other_boxes)
            best_other = float(other_ious.max()) if len(other_ious) else 0.0
            if best_other >= iou_threshold and best_other > best_subset:
                continue

            if best_subset >= iou_threshold:
                best_pos = int(subset_ious.argmax())
                gt_idx = subset_gts[best_pos][0]
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
        ap = ap_from_pr(np.array(tp), np.array(fp), len(gt_list))
        if ap is not None:
            aps.append(ap)
    return float(np.mean(aps)) if aps else 0.0


def matched_edge_error(gts_by_class, preds_by_class, iou_threshold=0.5):
    errors = []
    center_errors = []
    for cls, all_gt in gts_by_class.items():
        gt_by_image = defaultdict(list)
        for idx, gt in enumerate(all_gt):
            gt_by_image[gt["image"]].append((idx, gt["box"]))
        matched = set()
        for det in sorted(preds_by_class.get(cls, []), key=lambda x: x["conf"], reverse=True):
            candidates = gt_by_image.get(det["image"], [])
            boxes = [box for _, box in candidates]
            ious = box_iou_one_to_many(det["box"], boxes)
            if len(ious) == 0 or float(ious.max()) < iou_threshold:
                continue
            best_pos = int(ious.argmax())
            gt_idx, gt_box = candidates[best_pos]
            key = (det["image"], cls, gt_idx)
            if key in matched:
                continue
            matched.add(key)
            errors.append(float(np.mean(np.abs(det["box"] - gt_box))))
            det_center = np.array([(det["box"][0] + det["box"][2]) / 2, (det["box"][1] + det["box"][3]) / 2])
            gt_center = np.array([(gt_box[0] + gt_box[2]) / 2, (gt_box[1] + gt_box[3]) / 2])
            center_errors.append(float(np.linalg.norm(det_center - gt_center)))
    return {
        "matched_instances": len(errors),
        "mean_edge_error_norm": float(np.mean(errors)) if errors else None,
        "mean_center_error_norm": float(np.mean(center_errors)) if center_errors else None,
    }


def summarize_instances(gts_by_class, gt_filter):
    return sum(1 for gt_list in gts_by_class.values() for gt in gt_list if gt_filter(gt))


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--yolo-dir", type=Path, required=True)
    parser.add_argument("--train-metadata", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--label-image-size", type=int, default=512)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    names = read_names(args.yolo_dir / "config.yaml")
    gts_by_class, image_gt_index = load_ground_truth(args.yolo_dir / "labels" / "val", args.label_image_size)
    preds_by_class = run_predictions(
        args.weights,
        args.yolo_dir / "images" / "val",
        args.imgsz,
        args.batch,
        args.conf,
        args.device,
    )

    thresholds = np.arange(0.5, 0.96, 0.05)
    size_rows = []
    for group in ["small", "medium", "large"]:
        filt = make_gt_filter("size", group)
        maps = [evaluate_subset(gts_by_class, image_gt_index, preds_by_class, filt, float(t)) for t in thresholds]
        size_rows.append(
            {
                "group": group,
                "instances": summarize_instances(gts_by_class, filt),
                "map50": maps[0],
                "map75": maps[5],
                "map90": maps[8],
                "map5095": float(np.mean(maps)),
            }
        )

    class_counts = training_class_counts(args.train_metadata, names)
    freq_groups = split_frequency_groups(class_counts)
    freq_rows = []
    for group in ["rare", "common", "frequent"]:
        filt = make_gt_filter("freq", freq_groups[group])
        maps = [evaluate_subset(gts_by_class, image_gt_index, preds_by_class, filt, float(t)) for t in thresholds]
        freq_rows.append(
            {
                "group": group,
                "classes": ",".join(names.get(cls, str(cls)) for cls in sorted(freq_groups[group])),
                "instances": summarize_instances(gts_by_class, filt),
                "map50": maps[0],
                "map75": maps[5],
                "map90": maps[8],
                "map5095": float(np.mean(maps)),
            }
        )

    all_filter = make_gt_filter("all", None)
    all_maps = [evaluate_subset(gts_by_class, image_gt_index, preds_by_class, all_filter, float(t)) for t in thresholds]
    boundary = {
        "all_map50": all_maps[0],
        "all_map75": all_maps[5],
        "all_map90": all_maps[8],
        "all_map5095": float(np.mean(all_maps)),
        **matched_edge_error(gts_by_class, preds_by_class, iou_threshold=0.5),
    }

    per_class_counts = {
        names.get(cls, str(cls)): {
            "train_instances": int(class_counts.get(cls, 0)),
            "eval_instances": int(len(gts_by_class.get(cls, []))),
        }
        for cls in sorted(names)
    }

    output = {
        "size": size_rows,
        "frequency": freq_rows,
        "boundary_proxy": boundary,
        "class_counts": per_class_counts,
        "notes": {
            "size_bins": "small area < 32^2 px, medium area < 96^2 px, otherwise large, using label_image_size.",
            "frequency_bins": "classes are sorted by training instance count; bottom 30% rare, middle 40% common, top 30% frequent.",
            "edge_error": "mean absolute normalized x1/y1/x2/y2 error over matched detections with IoU >= 0.5.",
        },
    }
    (args.out_dir / "finegrained_yoloscore.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    write_csv(args.out_dir / "size_yoloscore.csv", size_rows, ["group", "instances", "map50", "map75", "map90", "map5095"])
    write_csv(args.out_dir / "frequency_yoloscore.csv", freq_rows, ["group", "classes", "instances", "map50", "map75", "map90", "map5095"])
    (args.out_dir / "boundary_proxy.json").write_text(json.dumps(boundary, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
