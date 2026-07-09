import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


DIOR_CLASSES = [
    "vehicle",
    "baseballfield",
    "groundtrackfield",
    "windmill",
    "bridge",
    "overpass",
    "ship",
    "airplane",
    "tenniscourt",
    "airport",
    "Expressway-Service-area",
    "basketballcourt",
    "stadium",
    "storagetank",
    "chimney",
    "dam",
    "Expressway-toll-station",
    "golffield",
    "trainstation",
    "harbor",
]

DOTA_ADDITIONAL_CLASSES = [
    "large-vehicle",
    "small-vehicle",
    "helicopter",
    "roundabout",
    "soccer-ball-field",
    "swimming-pool",
]

CLASS_ALIASES = {
    "plane": "airplane",
    "storage-tank": "storagetank",
    "baseball-diamond": "baseballfield",
    "tennis-court": "tenniscourt",
    "basketball-court": "basketballcourt",
    "ground-track-field": "groundtrackfield",
    # Zero-shot DIOR-RSVG generalization probes. These labels stay unchanged
    # in prompts, but reuse the closest seen layout color for conditioning.
    "solar-panel": "storagetank",
    "parking-lot": "vehicle",
    "railway-track": "trainstation",
    "container-yard": "harbor",
    "pier": "harbor",
    "helipad": "airport",
    "oil-refinery": "chimney",
    "greenhouse": "golffield",
}

LAYOUT_CLASSES = DIOR_CLASSES + DOTA_ADDITIONAL_CLASSES

CLASS_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(LAYOUT_CLASSES)}
for alias, canonical in CLASS_ALIASES.items():
    CLASS_TO_ID[alias] = CLASS_TO_ID[canonical]

PALETTE = np.array(
    [
        [230, 25, 75],
        [60, 180, 75],
        [255, 225, 25],
        [0, 130, 200],
        [245, 130, 48],
        [145, 30, 180],
        [70, 240, 240],
        [240, 50, 230],
        [210, 245, 60],
        [250, 190, 190],
        [0, 128, 128],
        [230, 190, 255],
        [170, 110, 40],
        [255, 250, 200],
        [128, 0, 0],
        [170, 255, 195],
        [128, 128, 0],
        [255, 215, 180],
        [0, 0, 128],
        [128, 128, 128],
        [255, 105, 97],
        [97, 168, 255],
        [255, 179, 71],
        [120, 220, 170],
        [190, 120, 255],
        [255, 240, 120],
    ],
    dtype=np.uint8,
)


@dataclass
class LayoutObject:
    label: str
    bbox: List[float]
    obbox: List[float]

    @property
    def valid(self) -> bool:
        return bool(self.label) and any(abs(float(v)) > 1e-8 for v in self.obbox)

    @property
    def class_id(self) -> Optional[int]:
        return CLASS_TO_ID.get(self.label)


def parse_layout_objects(caption: Sequence[str], bboxes: Sequence[Sequence[float]], obboxes: Sequence[Sequence[float]]):
    objects: List[LayoutObject] = []
    for label, bbox, obbox in zip(caption[1:], bboxes, obboxes):
        obj = LayoutObject(label=str(label), bbox=[float(v) for v in bbox], obbox=[float(v) for v in obbox])
        if obj.valid and obj.class_id is not None:
            objects.append(obj)
    return objects


def prompt_from_caption(caption: Sequence[str], mode: str = "scene") -> str:
    if mode == "scene":
        return str(caption[0])
    if mode == "scene_with_objects":
        objects = [name for name in caption[1:] if name]
        if not objects:
            return str(caption[0])
        return f"{caption[0]} Objects: {', '.join(objects)}."
    raise ValueError(f"Unknown prompt mode: {mode}")


def _scaled_polygon(obbox: Sequence[float], width: int, height: int) -> List[Tuple[int, int]]:
    pts = []
    for x, y in zip(obbox[0::2], obbox[1::2]):
        px = int(round(float(x) * (width - 1)))
        py = int(round(float(y) * (height - 1)))
        pts.append((max(0, min(width - 1, px)), max(0, min(height - 1, py))))
    return pts


def _expanded_oriented_polygon(
    obbox: Sequence[float],
    width: int,
    height: int,
    min_size: float,
) -> List[Tuple[int, int]]:
    """Expand small OBBs in image space while preserving center and orientation."""
    pts = np.asarray(
        [
            [float(x) * (width - 1), float(y) * (height - 1)]
            for x, y in zip(obbox[0::2], obbox[1::2])
        ],
        dtype=np.float32,
    )
    if pts.shape != (4, 2):
        return _scaled_polygon(obbox, width, height)

    center = pts.mean(axis=0)
    side_a = pts[1] - pts[0]
    side_b = pts[2] - pts[1]
    len_a = float(np.linalg.norm(side_a))
    len_b = float(np.linalg.norm(side_b))
    if len_a < 1e-6 or len_b < 1e-6:
        return _scaled_polygon(obbox, width, height)

    unit_a = side_a / len_a
    unit_b = side_b / len_b
    len_a = max(len_a, float(min_size))
    len_b = max(len_b, float(min_size))
    half_a = unit_a * (len_a * 0.5)
    half_b = unit_b * (len_b * 0.5)
    expanded = np.stack(
        [
            center - half_a - half_b,
            center + half_a - half_b,
            center + half_a + half_b,
            center - half_a + half_b,
        ],
        axis=0,
    )
    expanded[:, 0] = np.clip(expanded[:, 0], 0, width - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, height - 1)
    return [(int(round(x)), int(round(y))) for x, y in expanded]


def _angle_from_obbox(obbox: Sequence[float]) -> float:
    x0, y0, x1, y1 = [float(v) for v in obbox[:4]]
    return math.atan2(y1 - y0, x1 - x0)


def _draw_mask(size: Tuple[int, int], polygon: Sequence[Tuple[int, int]], value: float = 1.0) -> np.ndarray:
    img = Image.new("F", size, 0.0)
    ImageDraw.Draw(img).polygon(list(polygon), fill=float(value))
    return np.asarray(img, dtype=np.float32)


def _draw_boundary(size: Tuple[int, int], polygon: Sequence[Tuple[int, int]], width: int = 2) -> np.ndarray:
    img = Image.new("F", size, 0.0)
    draw = ImageDraw.Draw(img)
    draw.line(list(polygon) + [polygon[0]], fill=1.0, width=width)
    return np.asarray(img, dtype=np.float32)


def _gaussian_heatmap(width: int, height: int, cx: float, cy: float, sigma: float) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    heat = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / max(2.0 * sigma * sigma, 1e-6))
    return heat.astype(np.float32)


def build_layout_tensor(
    objects: Sequence[LayoutObject],
    resolution: int = 512,
    mode: str = "layout_control",
) -> torch.Tensor:
    """Build a control tensor in CHW format.

    simple_control returns 3 RGB channels. layout_control/layout_control_v2 returns:
    object mask + class one-hot masks + sin(angle) + cos(angle) + center heatmap + boundary.

    layout_control_v2 keeps the same 25-channel contract but strengthens small-object
    signals for FLUX's 32x32 packed latent grid:
      - small OBBs are expanded to at least 2 latent tokens in each direction;
      - center heatmaps are wider and include a weak filled target prior;
      - edges are thicker, so tiny oriented boxes survive the condition encoder.
    """
    width = height = int(resolution)
    if mode == "simple_control":
        canvas = np.zeros((height, width, 3), dtype=np.float32)
        for obj in objects:
            cid = obj.class_id
            if cid is None:
                continue
            polygon = _scaled_polygon(obj.obbox, width, height)
            mask = _draw_mask((width, height), polygon, 1.0)[..., None]
            color = PALETTE[cid].astype(np.float32) / 255.0
            angle = _angle_from_obbox(obj.obbox)
            direction_gain = 0.65 + 0.35 * np.array(
                [(math.sin(angle) + 1.0) * 0.5, (math.cos(angle) + 1.0) * 0.5, 1.0],
                dtype=np.float32,
            )
            canvas = np.maximum(canvas, mask * color * direction_gain)
        return torch.from_numpy(canvas).permute(2, 0, 1).contiguous()

    if mode not in {"layout_control", "layout_control_v2"}:
        raise ValueError(f"Unknown layout mode: {mode}")

    use_v2 = mode == "layout_control_v2"
    latent_stride = max(1.0, float(resolution) / 32.0)
    min_control_size = 2.0 * latent_stride
    edge_width = max(2, int(round(latent_stride / 8.0)))
    if use_v2:
        edge_width = max(4, int(round(latent_stride / 4.0)))

    channels = 1 + len(LAYOUT_CLASSES) + 4
    cond = np.zeros((channels, height, width), dtype=np.float32)
    mask_ch = 0
    class_offset = 1
    sin_ch = class_offset + len(LAYOUT_CLASSES)
    cos_ch = sin_ch + 1
    heat_ch = cos_ch + 1
    edge_ch = heat_ch + 1

    for obj in objects:
        cid = obj.class_id
        if cid is None:
            continue
        polygon = _scaled_polygon(obj.obbox, width, height)
        control_polygon = (
            _expanded_oriented_polygon(obj.obbox, width, height, min_control_size) if use_v2 else polygon
        )
        poly_mask = _draw_mask((width, height), control_polygon, 1.0)
        cond[mask_ch] = np.maximum(cond[mask_ch], poly_mask)
        cond[class_offset + cid] = np.maximum(cond[class_offset + cid], poly_mask)

        angle = _angle_from_obbox(obj.obbox)
        cond[sin_ch] = np.where(poly_mask > 0, math.sin(angle), cond[sin_ch])
        cond[cos_ch] = np.where(poly_mask > 0, math.cos(angle), cond[cos_ch])

        xs = [p[0] for p in control_polygon]
        ys = [p[1] for p in control_polygon]
        cx = float(sum(xs)) / max(len(xs), 1)
        cy = float(sum(ys)) / max(len(ys), 1)
        box_span = max(max(xs) - min(xs), max(ys) - min(ys), 4)
        if use_v2:
            sigma = max(box_span / 3.0, latent_stride)
            heat = np.maximum(_gaussian_heatmap(width, height, cx, cy, sigma), 0.35 * poly_mask)
            cond[heat_ch] = np.maximum(cond[heat_ch], np.clip(heat, 0.0, 1.0))
            cond[edge_ch] = np.maximum(
                cond[edge_ch],
                _draw_boundary((width, height), control_polygon, width=edge_width),
            )
            cond[edge_ch] = np.maximum(
                cond[edge_ch],
                _draw_boundary((width, height), polygon, width=max(2, edge_width // 2)),
            )
        else:
            sigma = box_span / 6.0
            cond[heat_ch] = np.maximum(cond[heat_ch], _gaussian_heatmap(width, height, cx, cy, sigma))
            cond[edge_ch] = np.maximum(cond[edge_ch], _draw_boundary((width, height), polygon, width=edge_width))

    return torch.from_numpy(cond).contiguous()


def layout_to_pil(
    objects: Sequence[LayoutObject],
    resolution: int = 512,
    draw_labels: bool = True,
    background: Tuple[int, int, int] = (18, 24, 32),
) -> Image.Image:
    img = Image.new("RGB", (resolution, resolution), background)
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()

    for obj in objects:
        cid = obj.class_id
        if cid is None:
            continue
        polygon = _scaled_polygon(obj.obbox, resolution, resolution)
        color = tuple(int(v) for v in PALETTE[cid])
        draw.polygon(polygon, fill=color + (90,), outline=color + (255,))
        draw.line(polygon + [polygon[0]], fill=color + (255,), width=3)
        if draw_labels:
            x = min(p[0] for p in polygon)
            y = min(p[1] for p in polygon)
            draw.rectangle((x, y, x + 7 * len(obj.label) + 6, y + 13), fill=(0, 0, 0, 160))
            draw.text((x + 3, y + 2), obj.label, fill=(255, 255, 255, 255), font=font)

    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def overlay_layout(image: Image.Image, objects: Sequence[LayoutObject], alpha: float = 0.45) -> Image.Image:
    layout = layout_to_pil(objects, resolution=image.size[0], draw_labels=True)
    return Image.blend(image.convert("RGB"), layout.resize(image.size), alpha=alpha)


def channel_count(mode: str) -> int:
    if mode == "simple_control":
        return 3
    if mode in {"layout_control", "layout_control_v2"}:
        return 1 + len(LAYOUT_CLASSES) + 4
    raise ValueError(f"Unknown layout mode: {mode}")


def objects_from_spec(spec: Iterable[dict]) -> List[LayoutObject]:
    objects = []
    for item in spec:
        label = str(item["label"])
        bbox = item.get("bbox")
        obbox = item.get("obbox")
        if obbox is None and bbox is not None:
            x0, y0, x1, y1 = [float(v) for v in bbox]
            obbox = [x0, y0, x1, y0, x1, y1, x0, y1]
        if bbox is None and obbox is not None:
            xs = [float(v) for v in obbox[0::2]]
            ys = [float(v) for v in obbox[1::2]]
            bbox = [min(xs), min(ys), max(xs), max(ys)]
        if bbox is None or obbox is None:
            raise ValueError("Each object spec needs either bbox or obbox.")
        objects.append(LayoutObject(label=label, bbox=list(bbox), obbox=list(obbox)))
    return objects
