import os
from typing import Optional

from PIL import Image, ImageDraw

from .layout_condition import layout_to_pil, overlay_layout


def make_comparison(
    generated: Image.Image,
    objects,
    gt: Optional[Image.Image] = None,
    resolution: int = 512,
) -> Image.Image:
    generated = generated.convert("RGB").resize((resolution, resolution))
    layout = layout_to_pil(objects, resolution=resolution)
    overlay = overlay_layout(generated, objects)
    panels = [layout, generated, overlay]
    labels = ["layout", "generated", "overlay"]

    if gt is not None:
        gt = gt.convert("RGB").resize((resolution, resolution))
        panels.insert(0, gt)
        labels.insert(0, "ground truth")

    header_h = 24
    canvas = Image.new("RGB", (resolution * len(panels), resolution + header_h), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    for idx, (panel, label) in enumerate(zip(panels, labels)):
        x = idx * resolution
        canvas.paste(panel, (x, header_h))
        draw.text((x + 8, 6), label, fill=(20, 20, 20))
    return canvas


def save_outputs(output_dir: str, stem: str, generated: Image.Image, objects, gt: Optional[Image.Image] = None, resolution: int = 512):
    os.makedirs(output_dir, exist_ok=True)
    generated.save(os.path.join(output_dir, f"{stem}_generated.png"))
    layout_to_pil(objects, resolution=resolution).save(os.path.join(output_dir, f"{stem}_layout.png"))
    overlay_layout(generated.resize((resolution, resolution)), objects).save(os.path.join(output_dir, f"{stem}_overlay.png"))
    make_comparison(generated, objects, gt=gt, resolution=resolution).save(os.path.join(output_dir, f"{stem}_comparison.png"))
