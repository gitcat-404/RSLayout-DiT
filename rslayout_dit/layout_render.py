"""
Layout rendering functions for RSLayout-DiT
Converts layout objects to RGB images for VAE encoding
"""

import math
from enum import Enum
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from rslayout_utils.layout_condition import LayoutObject, PALETTE, CLASS_TO_ID


class LayoutRenderStyle(Enum):
    """Different rendering styles for layout visualization"""
    COLORED_POLYGONS = "colored_polygons"  # Colored polygons with direction arrows (recommended)
    SEMANTIC_MAP = "semantic_map"  # Semantic segmentation style
    EDGE_MAP = "edge_map"  # Only boundaries
    HEATMAP = "heatmap"  # Gaussian heatmaps


def _scaled_polygon(obbox: List[float], width: int, height: int) -> List[Tuple[int, int]]:
    """Scale normalized obbox coordinates to pixel coordinates"""
    pts = []
    for x, y in zip(obbox[0::2], obbox[1::2]):
        px = int(round(float(x) * (width - 1)))
        py = int(round(float(y) * (height - 1)))
        pts.append((max(0, min(width - 1, px)), max(0, min(height - 1, py))))
    return pts


def _angle_from_obbox(obbox: List[float]) -> float:
    """Calculate angle from oriented bounding box"""
    x0, y0, x1, y1 = [float(v) for v in obbox[:4]]
    return math.atan2(y1 - y0, x1 - x0)


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    center: Tuple[float, float],
    angle: float,
    length: float,
    color: Tuple[int, int, int],
    width: int = 3
):
    """Draw a direction arrow"""
    cx, cy = center
    # Arrow end point
    end_x = cx + length * math.cos(angle)
    end_y = cy + length * math.sin(angle)

    # Main line
    draw.line([(cx, cy), (end_x, end_y)], fill=color, width=width)

    # Arrow head
    arrow_angle = 30 * math.pi / 180  # 30 degrees
    arrow_len = length * 0.3

    left_x = end_x - arrow_len * math.cos(angle - arrow_angle)
    left_y = end_y - arrow_len * math.sin(angle - arrow_angle)
    right_x = end_x - arrow_len * math.cos(angle + arrow_angle)
    right_y = end_y - arrow_len * math.sin(angle + arrow_angle)

    draw.line([(end_x, end_y), (left_x, left_y)], fill=color, width=width)
    draw.line([(end_x, end_y), (right_x, right_y)], fill=color, width=width)


def render_layout_to_rgb(
    objects: List[LayoutObject],
    resolution: int = 512,
    style: LayoutRenderStyle = LayoutRenderStyle.COLORED_POLYGONS,
    background_color: Tuple[int, int, int] = (0, 0, 0),
    draw_arrows: bool = True,
    arrow_length: float = 25.0,
) -> Image.Image:
    """
    Render layout objects to an RGB image for VAE encoding

    Args:
        objects: List of layout objects
        resolution: Output image resolution
        style: Rendering style
        background_color: Background RGB color
        draw_arrows: Whether to draw direction arrows
        arrow_length: Length of direction arrows in pixels

    Returns:
        PIL Image in RGB mode
    """
    canvas = Image.new("RGB", (resolution, resolution), background_color)
    draw = ImageDraw.Draw(canvas)

    if style == LayoutRenderStyle.COLORED_POLYGONS:
        # Create overlay for semi-transparent polygons
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)

        for obj in objects:
            if obj.class_id is None:
                continue

            # Scale polygon to target resolution
            polygon = _scaled_polygon(obj.obbox, resolution, resolution)

            # Get class color
            color = tuple(int(v) for v in PALETTE[obj.class_id])

            # Draw filled polygon with transparency
            overlay_draw.polygon(polygon, fill=color + (180,))

            # Draw boundary (solid line)
            overlay_draw.line(polygon + [polygon[0]], fill=color + (255,), width=3)

        # Composite overlay onto canvas
        canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")

        # Draw direction arrows on top
        if draw_arrows:
            for obj in objects:
                if obj.class_id is None:
                    continue

                polygon = _scaled_polygon(obj.obbox, resolution, resolution)
                color = tuple(int(v) for v in PALETTE[obj.class_id])

                # Calculate center
                center_x = sum(p[0] for p in polygon) / len(polygon)
                center_y = sum(p[1] for p in polygon) / len(polygon)

                # Get angle
                angle = _angle_from_obbox(obj.obbox)

                # Draw arrow
                _draw_arrow(draw, (center_x, center_y), angle, arrow_length, (255, 255, 255), width=3)

    elif style == LayoutRenderStyle.SEMANTIC_MAP:
        # Semantic segmentation style (solid colors, no transparency)
        for obj in objects:
            if obj.class_id is None:
                continue

            polygon = _scaled_polygon(obj.obbox, resolution, resolution)
            color = tuple(int(v) for v in PALETTE[obj.class_id])
            draw.polygon(polygon, fill=color)

    elif style == LayoutRenderStyle.EDGE_MAP:
        # Only draw boundaries (white lines on black background)
        for obj in objects:
            if obj.class_id is None:
                continue

            polygon = _scaled_polygon(obj.obbox, resolution, resolution)
            draw.line(polygon + [polygon[0]], fill=(255, 255, 255), width=3)

            if draw_arrows:
                center_x = sum(p[0] for p in polygon) / len(polygon)
                center_y = sum(p[1] for p in polygon) / len(polygon)
                angle = _angle_from_obbox(obj.obbox)
                _draw_arrow(draw, (center_x, center_y), angle, arrow_length, (255, 255, 255), width=2)

    elif style == LayoutRenderStyle.HEATMAP:
        # Gaussian heatmap style
        heatmap = np.zeros((resolution, resolution, 3), dtype=np.float32)

        for obj in objects:
            if obj.class_id is None:
                continue

            polygon = _scaled_polygon(obj.obbox, resolution, resolution)
            color = PALETTE[obj.class_id].astype(np.float32) / 255.0

            # Calculate center and sigma
            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)
            sigma = max(max(xs) - min(xs), max(ys) - min(ys), 4) / 4.0

            # Generate Gaussian heatmap
            yy, xx = np.mgrid[0:resolution, 0:resolution]
            gaussian = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma * sigma))

            # Add to heatmap with class color
            for c in range(3):
                heatmap[:, :, c] = np.maximum(heatmap[:, :, c], gaussian * color[c])

        # Convert to PIL Image
        heatmap = (heatmap * 255).astype(np.uint8)
        canvas = Image.fromarray(heatmap, mode="RGB")

    return canvas


def render_layout_with_labels(
    objects: List[LayoutObject],
    resolution: int = 512,
    draw_labels: bool = True,
    background_color: Tuple[int, int, int] = (18, 24, 32),
) -> Image.Image:
    """
    Render layout with class labels (for visualization, not for training)

    Args:
        objects: List of layout objects
        resolution: Output image resolution
        draw_labels: Whether to draw text labels
        background_color: Background RGB color

    Returns:
        PIL Image with labels
    """
    img = Image.new("RGB", (resolution, resolution), background_color)
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font = ImageFont.load_default()

    for obj in objects:
        if obj.class_id is None:
            continue

        polygon = _scaled_polygon(obj.obbox, resolution, resolution)
        color = tuple(int(v) for v in PALETTE[obj.class_id])

        # Draw polygon
        draw.polygon(polygon, fill=color + (90,), outline=color + (255,))
        draw.line(polygon + [polygon[0]], fill=color + (255,), width=3)

        # Draw label
        if draw_labels:
            x = min(p[0] for p in polygon)
            y = min(p[1] for p in polygon)
            label_text = obj.label

            # Background for text
            bbox = draw.textbbox((x, y), label_text, font=font)
            draw.rectangle(bbox, fill=(0, 0, 0, 200))
            draw.text((x + 2, y + 2), label_text, fill=(255, 255, 255, 255), font=font)

    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def batch_render_layouts(
    objects_list: List[List[LayoutObject]],
    resolution: int = 512,
    style: LayoutRenderStyle = LayoutRenderStyle.COLORED_POLYGONS,
) -> torch.Tensor:
    """
    Batch render multiple layouts to tensors

    Args:
        objects_list: List of object lists (one per sample)
        resolution: Output resolution
        style: Rendering style

    Returns:
        Tensor of shape [B, 3, H, W] in range [0, 1]
    """
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to [0, 1] and CHW format
    ])

    rendered = []
    for objects in objects_list:
        img = render_layout_to_rgb(objects, resolution, style)
        tensor = transform(img)
        rendered.append(tensor)

    return torch.stack(rendered)
