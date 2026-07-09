"""
Dataset for RSLayout-DiT training
Loads DIOR data and renders layout conditions as RGB images
"""

import json
import os
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from rslayout_utils.layout_condition import parse_layout_objects
from rslayout_utils.prompting import build_satellite_prompts, legacy_clip_prompt
from .layout_render import render_layout_to_rgb, LayoutRenderStyle


class RSLayoutDataset(Dataset):
    """
    DIOR dataset for RSLayout-DiT training
    Returns original images and rendered layout condition images
    """

    def __init__(
        self,
        data_dir: str,
        resolution: int = 512,
        render_style: LayoutRenderStyle = LayoutRenderStyle.COLORED_POLYGONS,
        prompt_mode: str = "satellite",  # "satellite" or "scene"
        max_samples: Optional[int] = None,
        draw_arrows: bool = True,
    ):
        self.data_dir = data_dir
        self.resolution = int(resolution)
        self.render_style = render_style
        self.prompt_mode = prompt_mode
        self.draw_arrows = draw_arrows

        # Load metadata
        metadata_path = os.path.join(data_dir, "metadata.jsonl")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"metadata.jsonl not found under {data_dir}")

        self.samples: List[Dict[str, Any]] = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

        if max_samples is not None:
            self.samples = self.samples[:int(max_samples)]

        # Image transforms (normalize to [-1, 1] for VAE)
        self.image_transform = transforms.Compose([
            transforms.Resize((self.resolution, self.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # [0, 1] -> [-1, 1]
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_image_path(self, file_name: str) -> str:
        """Support both flat DIOR metadata and DOTA's images/ subdirectory layout."""
        candidates = [
            os.path.join(self.data_dir, file_name),
            os.path.join(self.data_dir, "images", file_name),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(
            f"Image {file_name!r} not found. Tried: " + ", ".join(candidates)
        )

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]

        # Load original image
        image_path = self._resolve_image_path(sample["file_name"])
        image = Image.open(image_path).convert("RGB")

        # Parse layout objects
        caption = sample["caption"]
        objects = parse_layout_objects(caption, sample["bndboxes"], sample["obboxes"])

        # Render layout condition as RGB image
        layout_image = render_layout_to_rgb(
            objects,
            resolution=self.resolution,
            style=self.render_style,
            draw_arrows=self.draw_arrows,
        )

        # Build prompts
        if self.prompt_mode == "satellite":
            clip_prompt, t5_prompt = build_satellite_prompts(caption, objects)
        else:  # "scene"
            from rslayout_utils.layout_condition import prompt_from_caption
            scene_prompt = prompt_from_caption(caption, "scene")
            clip_prompt = legacy_clip_prompt(scene_prompt, objects)
            t5_prompt = scene_prompt

        return {
            "pixel_values": self.image_transform(image),  # [3, H, W], [-1, 1]
            "cond_pixel_values": self.image_transform(layout_image),  # [3, H, W], [-1, 1]
            "prompt": t5_prompt,
            "clip_prompt": clip_prompt,
            "file_name": sample["file_name"],
            "objects": objects,
            "raw": sample,
        }


def collate_rslayout_batch(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for RSLayout-DiT dataloader

    Args:
        examples: List of dataset samples

    Returns:
        Batched dictionary
    """
    return {
        "pixel_values": torch.stack([ex["pixel_values"] for ex in examples]).float(),
        "cond_pixel_values": torch.stack([ex["cond_pixel_values"] for ex in examples]).float(),
        "prompts": [ex["prompt"] for ex in examples],
        "clip_prompts": [ex["clip_prompt"] for ex in examples],
        "file_names": [ex["file_name"] for ex in examples],
        "objects": [ex["objects"] for ex in examples],
        "raw": [ex["raw"] for ex in examples],
    }


class RSLayoutDatasetWithCache(RSLayoutDataset):
    """
    Dataset with optional latent caching for faster training
    """

    def __init__(
        self,
        data_dir: str,
        resolution: int = 512,
        render_style: LayoutRenderStyle = LayoutRenderStyle.COLORED_POLYGONS,
        prompt_mode: str = "satellite",
        max_samples: Optional[int] = None,
        draw_arrows: bool = True,
        cache_latents: bool = False,
        cache_dir: Optional[str] = None,
    ):
        super().__init__(
            data_dir=data_dir,
            resolution=resolution,
            render_style=render_style,
            prompt_mode=prompt_mode,
            max_samples=max_samples,
            draw_arrows=draw_arrows,
        )

        self.cache_latents = cache_latents
        self.cache_dir = cache_dir
        self.latent_cache = {}

        if cache_latents and cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self._load_cache()

    def _load_cache(self):
        """Load cached latents from disk"""
        cache_file = os.path.join(self.cache_dir, "latent_cache.pt")
        if os.path.exists(cache_file):
            print(f"Loading latent cache from {cache_file}")
            self.latent_cache = torch.load(cache_file)
            print(f"Loaded {len(self.latent_cache)} cached latents")

    def _save_cache(self):
        """Save cached latents to disk"""
        if self.cache_dir:
            cache_file = os.path.join(self.cache_dir, "latent_cache.pt")
            torch.save(self.latent_cache, cache_file)
            print(f"Saved {len(self.latent_cache)} latents to cache")

    def cache_sample_latents(self, index: int, image_latent: torch.Tensor, cond_latent: torch.Tensor):
        """Cache latents for a sample"""
        if self.cache_latents:
            file_name = self.samples[index]["file_name"]
            self.latent_cache[file_name] = {
                "image_latent": image_latent.cpu(),
                "cond_latent": cond_latent.cpu(),
            }

    def get_cached_latents(self, index: int) -> Optional[Dict[str, torch.Tensor]]:
        """Get cached latents for a sample"""
        if self.cache_latents:
            file_name = self.samples[index]["file_name"]
            return self.latent_cache.get(file_name)
        return None
