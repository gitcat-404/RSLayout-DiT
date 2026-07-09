import json
import os
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .layout_condition import build_layout_tensor, layout_to_pil, parse_layout_objects, prompt_from_caption
from .prompting import build_satellite_prompts, legacy_clip_prompt


class DiorLayoutDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        resolution: int = 512,
        control_mode: str = "layout_control",
        prompt_mode: str = "scene",
        max_samples: Optional[int] = None,
    ):
        self.data_dir = data_dir
        self.resolution = int(resolution)
        self.control_mode = control_mode
        self.prompt_mode = prompt_mode
        metadata_path = os.path.join(data_dir, "metadata.jsonl")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"metadata.jsonl not found under {data_dir}")

        self.samples: List[Dict[str, Any]] = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))
        if max_samples is not None:
            self.samples = self.samples[: int(max_samples)]

        self.image_transform = transforms.Compose(
            [
                transforms.Resize((self.resolution, self.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        image_path = os.path.join(self.data_dir, sample["file_name"])
        image = Image.open(image_path).convert("RGB")
        caption = sample["caption"]
        objects = parse_layout_objects(caption, sample["bndboxes"], sample["obboxes"])

        legacy_prompt = prompt_from_caption(caption, self.prompt_mode)
        clip_prompt, t5_prompt = build_satellite_prompts(caption, objects)
        legacy_clip = legacy_clip_prompt(legacy_prompt, objects)
        return {
            "pixel_values": self.image_transform(image),
            "control": build_layout_tensor(objects, self.resolution, self.control_mode),
            "prompt": t5_prompt,
            "clip_prompt": clip_prompt,
            "legacy_prompt": legacy_prompt,
            "legacy_clip_prompt": legacy_clip,
            "file_name": sample["file_name"],
            "objects": objects,
            "raw": sample,
        }


def collate_layout_batch(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "pixel_values": torch.stack([ex["pixel_values"] for ex in examples]).float(),
        "control": torch.stack([ex["control"] for ex in examples]).float(),
        "prompts": [ex["prompt"] for ex in examples],
        "clip_prompts": [ex["clip_prompt"] for ex in examples],
        "legacy_prompts": [ex["legacy_prompt"] for ex in examples],
        "legacy_clip_prompts": [ex["legacy_clip_prompt"] for ex in examples],
        "file_names": [ex["file_name"] for ex in examples],
        "objects": [ex["objects"] for ex in examples],
        "raw": [ex["raw"] for ex in examples],
    }


def save_layout_preview(path: str, objects, resolution: int = 512):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    layout_to_pil(objects, resolution=resolution).save(path)
