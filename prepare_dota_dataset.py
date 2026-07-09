import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, CLIPModel


Image.MAX_IMAGE_PIXELS = None

CLASSES = [
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


def parse_dota_txt(path):
    anns = []
    if not path.exists():
        return anns
    with path.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            try:
                poly = np.array([float(v) for v in parts[:8]], dtype=np.float32)
            except ValueError:
                continue
            label = parts[8]
            if label not in CLASSES:
                continue
            diff = int(parts[9]) if len(parts) > 9 and parts[9].isdigit() else 0
            anns.append((poly, label, diff))
    return anns


def poly_area(poly):
    pts = poly.reshape(4, 2)
    x = pts[:, 0]
    y = pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def hbb(poly):
    pts = poly.reshape(4, 2)
    xy_min = pts.min(axis=0)
    xy_max = pts.max(axis=0)
    return np.array([xy_min[0], xy_min[1], xy_max[0], xy_max[1]], dtype=np.float32)


def make_caption(labels):
    names = [name for name in labels if name]
    if not names:
        return "This is an aerial image."
    counts = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    parts = []
    for name, count in sorted(counts.items()):
        if count == 1:
            parts.append(f"one {name}")
        else:
            parts.append(f"{count} {name}s")
    return "This is an aerial image with " + ", ".join(parts) + "."


def crop_positions(length, crop_size, stride):
    if length <= crop_size:
        return [0]
    positions = list(range(0, length - crop_size + 1, stride))
    last = length - crop_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def split_mode(raw_root, out_root, mode, crop_size=512, gap=100, max_objects=6):
    image_dir = raw_root / mode / "images"
    ann_dir = raw_root / mode / "annotations" / "version1.0"
    out_img_dir = out_root / mode / "images"
    out_ann_dir = out_root / mode / "labelTxt"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_ann_dir.mkdir(parents=True, exist_ok=True)

    stride = crop_size - gap
    metadata = []
    image_paths = sorted(image_dir.glob("*.png"))
    for image_path in tqdm(image_paths, desc=f"split {mode}"):
        anns = parse_dota_txt(ann_dir / f"{image_path.stem}.txt")
        if not anns:
            continue
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            for y0 in crop_positions(height, crop_size, stride):
                for x0 in crop_positions(width, crop_size, stride):
                    kept = []
                    for poly, label, diff in anns:
                        shifted = poly.copy()
                        shifted[0::2] -= x0
                        shifted[1::2] -= y0
                        box = hbb(shifted)
                        if box[0] < 0 or box[1] < 0 or box[2] > crop_size or box[3] > crop_size:
                            continue
                        if box[2] - box[0] < 2 or box[3] - box[1] < 2:
                            continue
                        kept.append((shifted, label, diff, poly_area(shifted)))
                    if not kept:
                        continue
                    kept = sorted(kept, key=lambda item: item[3], reverse=True)[:max_objects]

                    tile_name = f"{image_path.stem}__{x0}___{y0}.png"
                    tile = Image.new("RGB", (crop_size, crop_size))
                    tile.paste(image.crop((x0, y0, min(x0 + crop_size, width), min(y0 + crop_size, height))), (0, 0))
                    tile.save(out_img_dir / tile_name)

                    labels = [item[1] for item in kept]
                    obboxes = [(item[0] / crop_size).clip(0, 1).round(6).tolist() for item in kept]
                    bndboxes = [(hbb(item[0]) / crop_size).clip(0, 1).round(6).tolist() for item in kept]

                    with (out_ann_dir / f"{Path(tile_name).stem}.txt").open("w") as f:
                        for poly, label, diff, _ in kept:
                            coords = " ".join(str(int(round(v))) for v in poly.tolist())
                            f.write(f"{coords} {label} {diff}\n")

                    while len(labels) < max_objects:
                        labels.append("")
                        obboxes.append([0, 0, 0, 0, 0, 0, 0, 0])
                        bndboxes.append([0, 0, 0, 0])
                    metadata.append(
                        {
                            "file_name": tile_name,
                            "caption": [make_caption(labels)] + labels,
                            "bndboxes": bndboxes,
                            "obboxes": obboxes,
                        }
                    )

    with (out_root / mode / "metadata.jsonl").open("w") as f:
        for item in metadata:
            f.write(json.dumps(item) + "\n")
    return len(metadata)


def build_foreground(out_root, max_per_class=500):
    fg_root = out_root / "results" / "foreground"
    for cls in CLASSES:
        (fg_root / cls).mkdir(parents=True, exist_ok=True)

    counts = {cls: len(list((fg_root / cls).glob("*.png"))) for cls in CLASSES}
    samples = []
    with (out_root / "train" / "metadata.jsonl").open() as f:
        for line in f:
            samples.append(json.loads(line))
    random.seed(1234)
    random.shuffle(samples)

    for sample in tqdm(samples, desc="foreground"):
        image_path = out_root / "train" / "images" / sample["file_name"]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            for idx, label in enumerate(sample["caption"][1:]):
                if not label or counts[label] >= max_per_class:
                    continue
                x0, y0, x1, y1 = sample["bndboxes"][idx]
                box = [int(round(x0 * image.width)), int(round(y0 * image.height)), int(round(x1 * image.width)), int(round(y1 * image.height))]
                if box[2] <= box[0] or box[3] <= box[1]:
                    continue
                crop = image.crop(box)
                if crop.width < 2 or crop.height < 2:
                    continue
                out_name = f"{Path(sample['file_name']).stem}_{idx}.png"
                crop.save(fg_root / label / out_name)
                counts[label] += 1
        if all(count >= max_per_class for count in counts.values()):
            break

    print("foreground_counts=" + json.dumps(counts, sort_keys=True))


class CLIPEncoder:
    def __init__(self, model_name, device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()

    @torch.no_grad()
    def __call__(self, caption, image):
        text_inputs = self.tokenizer(caption, padding=True, truncation=True, return_tensors="pt").to(self.device)
        image_inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        txt_emb = F.normalize(self.model.get_text_features(**text_inputs), dim=-1).cpu()
        img_emb = F.normalize(self.model.get_image_features(**image_inputs), dim=-1).cpu()
        return txt_emb, img_emb


def build_embeddings(out_root, max_samples, device):
    emb_path = out_root / "dota_emb.pt"
    if emb_path.exists():
        print(f"embedding_exists={emb_path}")
        return
    samples = []
    with (out_root / "train" / "metadata.jsonl").open() as f:
        for line in f:
            samples.append(json.loads(line))
    random.seed(1234)
    if len(samples) > max_samples:
        samples = random.sample(samples, max_samples)

    encoder = CLIPEncoder("openai/clip-vit-large-patch14", device)
    emb = {}
    for sample in tqdm(samples, desc="clip emb"):
        image = Image.open(out_root / "train" / "images" / sample["file_name"]).convert("RGB")
        txt_emb, img_emb = encoder(sample["caption"][0], image)
        emb[sample["file_name"]] = {"txt_emb": txt_emb, "img_emb": img_emb}
    torch.save(emb, emb_path)
    print(f"saved_embedding={emb_path} count={len(emb)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", default="/path/to/DOTA/extracted")
    parser.add_argument("--out_root", default="path_to_data/DOTA")
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--gap", type=int, default=100)
    parser.add_argument("--max_objects", type=int, default=6)
    parser.add_argument("--max_foreground_per_class", type=int, default=500)
    parser.add_argument("--max_emb_samples", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for mode in ("train", "val"):
        meta_path = out_root / mode / "metadata.jsonl"
        if meta_path.exists():
            count = sum(1 for _ in meta_path.open())
            print(f"{mode}_metadata_exists={count}")
        else:
            count = split_mode(raw_root, out_root, mode, args.crop_size, args.gap, args.max_objects)
            print(f"{mode}_metadata_created={count}")

    build_foreground(out_root, args.max_foreground_per_class)
    build_embeddings(out_root, args.max_emb_samples, args.device)


if __name__ == "__main__":
    main()
