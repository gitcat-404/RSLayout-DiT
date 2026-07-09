#!/usr/bin/env python
"""Evaluation-friendly inference for RS-FLUX FluxControlNet layout checkpoints."""

import argparse
import json
import os
from pathlib import Path

import torch

from rslayout_utils.layout_condition import (
    build_layout_tensor,
    layout_to_pil,
    overlay_layout,
    parse_layout_objects,
)
from rslayout_utils.models import build_control_pipeline, dtype_from_string, load_controlnet_metadata
from rslayout_utils.prompting import build_satellite_prompts


def parse_args():
    parser = argparse.ArgumentParser("Infer RS-FLUX layout ControlNet with eval-compatible names.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/path/to/FLUX-RS",
    )
    parser.add_argument("--controlnet_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="path_to_data/DIOR/val")
    parser.add_argument("--prompt_metadata_name", type=str, default="metadata_rslayout_prompts.jsonl")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=1.0)
    parser.add_argument("--control_guidance_start", type=float, default=0.0)
    parser.add_argument("--control_guidance_end", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch_dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--save_layout", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_samples(args):
    data_dir = Path(args.data_dir)
    rows = load_jsonl(data_dir / "metadata.jsonl")
    for idx, row in enumerate(rows):
        row["_metadata_index"] = idx

    prompt_path = data_dir / args.prompt_metadata_name
    if prompt_path.exists():
        prompt_rows = load_jsonl(prompt_path)
        prompt_by_file = {row["file_name"]: row for row in prompt_rows}
        for row in rows:
            prompt_row = prompt_by_file.get(row["file_name"])
            if prompt_row:
                row["clip_prompt"] = prompt_row.get("new_clip_prompt")
                row["t5_prompt"] = prompt_row.get("new_t5_prompt")

    start = max(0, args.start_index)
    end = args.end_index if args.end_index is not None else len(rows)
    rows = rows[start:end]
    if args.num_samples is not None and args.num_samples > 0:
        rows = rows[: args.num_samples]
    return rows


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    dtype = dtype_from_string(args.torch_dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metadata = load_controlnet_metadata(args.controlnet_path)
    control_method = metadata["method"]

    print("Loading RS-FLUX ControlNet pipeline...")
    pipe = build_control_pipeline(
        args.pretrained_model_name_or_path,
        args.controlnet_path,
        torch_dtype=dtype,
        device=device,
    )
    pipe.transformer.eval()
    pipe.controlnet.eval()
    pipe.vae.eval()
    pipe.text_encoder.eval()
    pipe.text_encoder_2.eval()

    samples = load_samples(args)
    print(f"Generating {len(samples)} images with scale={args.controlnet_conditioning_scale}...")

    for local_idx, sample in enumerate(samples):
        output_index = int(sample.get("_metadata_index", local_idx))
        image_stem = os.path.splitext(sample.get("file_name", f"sample_{output_index:04d}"))[0]
        stem = f"{output_index:04d}_{image_stem}"
        generated_path = Path(args.output_dir) / f"{stem}_generated.png"
        if generated_path.exists() and not args.overwrite:
            print(f"[{local_idx + 1}/{len(samples)}] Skipping existing output: {generated_path}")
            continue

        caption = sample["caption"]
        objects = parse_layout_objects(caption, sample["bndboxes"], sample["obboxes"])
        built_clip_prompt, built_t5_prompt = build_satellite_prompts(caption, objects)
        clip_prompt = sample.get("clip_prompt") or built_clip_prompt
        t5_prompt = sample.get("t5_prompt") or built_t5_prompt

        print(f"\n[{local_idx + 1}/{len(samples)}] Processing {sample['file_name']}")
        print(f"CLIP prompt: {clip_prompt}")
        print(f"T5 prompt: {t5_prompt}")
        print(f"Objects: {len(objects)}")

        control = build_layout_tensor(objects, resolution=args.resolution, mode=control_method).unsqueeze(0)
        generator = torch.Generator(device=device).manual_seed(args.seed + output_index)

        with torch.inference_mode():
            image = pipe(
                prompt=clip_prompt,
                prompt_2=t5_prompt,
                height=args.resolution,
                width=args.resolution,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
                control_image=control,
                controlnet_conditioning_scale=args.controlnet_conditioning_scale,
                control_guidance_start=args.control_guidance_start,
                control_guidance_end=args.control_guidance_end,
                max_sequence_length=512,
            ).images[0]

        image.save(generated_path)
        if args.save_layout:
            layout_to_pil(objects, resolution=args.resolution).save(Path(args.output_dir) / f"{stem}_layout.png")
            overlay_layout(image.resize((args.resolution, args.resolution)), objects).save(
                Path(args.output_dir) / f"{stem}_overlay.png"
            )
        (Path(args.output_dir) / f"{stem}_prompt.txt").write_text(
            f"CLIP: {clip_prompt}\nT5: {t5_prompt}\n", encoding="utf-8"
        )
        print(f"Saved to {args.output_dir}/{stem}_*")

    print(f"\nDone! Generated {len(samples)} images in {args.output_dir}")


if __name__ == "__main__":
    main()
