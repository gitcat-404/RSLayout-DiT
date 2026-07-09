#!/usr/bin/env python
import argparse
import json
import os
import random
import warnings
from typing import Dict, List

import torch
from diffusers import FluxPipeline
from PIL import Image

from rslayout_utils.layout_condition import build_layout_tensor, objects_from_spec, parse_layout_objects, prompt_from_caption
from rslayout_utils.models import build_control_pipeline, dtype_from_string, load_controlnet_metadata
from rslayout_utils.prompting import build_satellite_prompts, legacy_clip_prompt
from rslayout_utils.visualize import save_outputs

warnings.filterwarnings("ignore", message="`FluxPosEmbed` is deprecated.*", category=FutureWarning)


def parse_args():
    parser = argparse.ArgumentParser("Infer with RS-FLUX text-only or RS-FLUX layout control.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="/path/to/FLUX-RS")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["text_only", "simple_control", "layout_control", "layout_control_v2"],
        default="layout_control",
    )
    parser.add_argument("--controlnet_path", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="path_to_data/DIOR/val")
    parser.add_argument("--layout_json", type=str, default=None, help="Optional JSON file with prompt and objects.")
    parser.add_argument("--output_dir", type=str, default="outputs-rslayout-controlnet")
    parser.add_argument("--sample_index", type=int, default=None)
    parser.add_argument("--sample_file", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--prompt_mode", type=str, choices=["scene", "scene_with_objects"], default="scene")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=1.0)
    parser.add_argument("--control_guidance_start", type=float, default=0.0)
    parser.add_argument("--control_guidance_end", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--torch_dtype", type=str, choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--enable_model_cpu_offload", action="store_true")
    parser.add_argument("--enable_sequential_cpu_offload", action="store_true")
    return parser.parse_args()


def read_metadata(data_dir: str) -> List[Dict]:
    path = os.path.join(data_dir, "metadata.jsonl")
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def choose_samples(args) -> List[Dict]:
    if args.layout_json:
        with open(args.layout_json, "r", encoding="utf-8") as f:
            spec = json.load(f)
        return [{"custom": True, "prompt": spec["prompt"], "objects": objects_from_spec(spec["objects"]), "file_name": "custom"}]

    data = read_metadata(args.data_dir)
    if args.sample_file is not None:
        selected = [sample for sample in data if sample["file_name"] == args.sample_file]
        if not selected:
            raise ValueError(f"sample_file not found: {args.sample_file}")
        return selected
    if args.sample_index is not None:
        return [data[args.sample_index]]
    random.seed(args.seed)
    return random.sample(data, k=min(args.num_samples, len(data)))


def sample_to_prompt_objects(sample: Dict, prompt_mode: str, override_prompt: str = None):
    if sample.get("custom"):
        objects = sample["objects"]
        prompt = sample["prompt"]
        clip_prompt = legacy_clip_prompt(prompt, objects)
        gt = None
    else:
        caption = sample["caption"]
        objects = parse_layout_objects(caption, sample["bndboxes"], sample["obboxes"])
        clip_prompt, prompt = build_satellite_prompts(caption, objects)
        gt = None
    if override_prompt:
        prompt = override_prompt
        clip_prompt = legacy_clip_prompt(prompt, objects)
    return clip_prompt, prompt, objects, gt


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    dtype = dtype_from_string(args.torch_dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "text_only":
        pipe = FluxPipeline.from_pretrained(args.pretrained_model_name_or_path, torch_dtype=dtype)
        if args.enable_sequential_cpu_offload:
            pipe.enable_sequential_cpu_offload()
        elif args.enable_model_cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(device)
        control_method = None
    else:
        if args.controlnet_path is None:
            raise ValueError("--controlnet_path is required for control modes.")
        metadata = load_controlnet_metadata(args.controlnet_path)
        control_method = metadata["method"]
        if control_method != args.mode:
            raise ValueError(f"ControlNet was trained for {control_method}, but --mode is {args.mode}")
        pipe = build_control_pipeline(
            args.pretrained_model_name_or_path,
            args.controlnet_path,
            torch_dtype=dtype,
            device=None if (args.enable_model_cpu_offload or args.enable_sequential_cpu_offload) else device,
        )
        if args.enable_sequential_cpu_offload:
            pipe.enable_sequential_cpu_offload()
        elif args.enable_model_cpu_offload:
            pipe.enable_model_cpu_offload()

    generator_device = "cpu" if (args.enable_model_cpu_offload or args.enable_sequential_cpu_offload) else device
    generator = torch.Generator(device=generator_device).manual_seed(args.seed)
    samples = choose_samples(args)

    for index, sample in enumerate(samples):
        clip_prompt, prompt, objects, _ = sample_to_prompt_objects(sample, args.prompt_mode, args.prompt)
        gt = None
        if not sample.get("custom"):
            gt_path = os.path.join(args.data_dir, sample["file_name"])
            if os.path.exists(gt_path):
                gt = Image.open(gt_path).convert("RGB")

        kwargs = {
            "prompt": clip_prompt,
            "prompt_2": prompt,
            "height": args.resolution,
            "width": args.resolution,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "generator": generator,
        }
        if args.mode != "text_only":
            control = build_layout_tensor(objects, resolution=args.resolution, mode=control_method).unsqueeze(0)
            kwargs.update(
                {
                    "control_image": control,
                    "controlnet_conditioning_scale": args.controlnet_conditioning_scale,
                    "control_guidance_start": args.control_guidance_start,
                    "control_guidance_end": args.control_guidance_end,
                }
            )

        image = pipe(**kwargs).images[0]
        stem = os.path.splitext(sample.get("file_name", f"sample_{index:04d}"))[0]
        stem = f"{index:04d}_{stem}_{args.mode}"
        save_outputs(args.output_dir, stem, image, objects, gt=gt, resolution=args.resolution)
        with open(os.path.join(args.output_dir, f"{stem}_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(f"CLIP prompt: {clip_prompt}\n")
            f.write(f"T5 prompt: {prompt}\n")


if __name__ == "__main__":
    main()
