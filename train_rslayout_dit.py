#!/usr/bin/env python
"""
Training script for RS-FLUX RSLayout-DiT
Based on RSLayout-DiT architecture with LoRA and causal attention
"""

import argparse
import json
import logging
import math
import os
import re
import signal
import warnings
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import FluxPipeline, FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from rslayout_dit.data import RSLayoutDataset, collate_rslayout_batch
from rslayout_dit.layout_render import LayoutRenderStyle
from rslayout_dit.lora_layers import MultiDoubleStreamBlockLoraProcessor, MultiSingleStreamBlockLoraProcessor

logger = get_logger(__name__, log_level="INFO")
warnings.filterwarnings("ignore", message="`FluxPosEmbed` is deprecated.*", category=FutureWarning)


def parse_args():
    parser = argparse.ArgumentParser("Train RS-FLUX with RSLayout-DiT LoRA")

    # Model paths
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                       default="/path/to/FLUX-RS",
                       help="Path to RS-FLUX model")
    parser.add_argument("--resume_lora_path", type=str, default=None,
                       help="Path to resume LoRA checkpoint")
    parser.add_argument("--no_auto_resume", action="store_true",
                       help="Disable automatically resuming from the latest LoRA checkpoint in output_dir")

    # Data
    parser.add_argument("--train_data_dir", type=str,
                       default="path_to_data/DIOR/train",
                       help="Training data directory")
    parser.add_argument("--validation_data_dir", type=str,
                       default="path_to_data/DIOR/val",
                       help="Validation data directory")
    parser.add_argument("--max_train_samples", type=int, default=None)

    # Layout rendering
    parser.add_argument("--render_style", type=str, default="colored_polygons",
                       choices=["colored_polygons", "semantic_map", "edge_map", "heatmap"])
    parser.add_argument("--draw_arrows", action="store_true", default=True)
    parser.add_argument("--prompt_mode", type=str, default="satellite",
                       choices=["satellite", "scene"])
    parser.add_argument("--t5_prompt_from_clip_prompt", action="store_true",
                       help="Feed the CLIP/simple prompt to both the CLIP and T5 text encoders.")

    # Training
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--cond_size", type=int, default=512,
                       help="Condition image size (should match resolution)")
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)

    # LoRA config
    parser.add_argument("--lora_rank", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_weight", type=float, default=1.0)
    parser.add_argument("--multi_scale_condition", action="store_true",
                       help="Use two layout condition branches: native layout and coarse low-pass layout.")
    parser.add_argument("--coarse_cond_size", type=int, default=256,
                       help="Downsample size for the coarse multi-scale layout condition before upsampling.")
    parser.add_argument("--init_extra_loras_from_first", action="store_true",
                       help="When resuming a one-condition LoRA into multi-scale mode, copy branch-0 weights to new branches.")
    parser.add_argument("--disable_causal_mask", action="store_true",
                       help="Ablation: use full bidirectional single-stream attention instead of condition-to-image causal masking.")

    # Checkpointing
    parser.add_argument("--output_dir", type=str, default="checkpoint-rslayout-dit")
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=3)
    parser.add_argument("--validation_steps", type=int, default=500)
    parser.add_argument("--num_validation_images", type=int, default=4)

    # System
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch_dtype", type=str, choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--guidance_scale", type=float, default=3.5)

    # Loss weighting
    parser.add_argument("--weighting_scheme", type=str, default="none",
                       choices=["none", "logit_normal", "mode", "cosmap"])
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)
    parser.add_argument("--mode_scale", type=float, default=1.29)
    parser.add_argument("--timestep_min", type=float, default=0.0,
                       help="Lower bound for sampled flow-matching timesteps after density sampling.")
    parser.add_argument("--timestep_max", type=float, default=1.0,
                       help="Upper bound for sampled flow-matching timesteps after density sampling.")
    parser.add_argument("--object_loss_weight", type=float, default=0.0,
                       help="Extra MSE weight inside annotated object boxes. 0 keeps the original loss.")
    parser.add_argument("--object_mask_dilation", type=int, default=2,
                       help="Object mask dilation in latent pixels for object weighted loss.")
    parser.add_argument("--small_object_loss_weight", type=float, default=0.0,
                       help="Extra object-box weight scaled by inverse sqrt normalized object area.")
    parser.add_argument("--small_object_reference_area", type=float, default=0.01,
                       help="Reference normalized area for small-object weighting.")
    parser.add_argument("--small_object_weight_max", type=float, default=4.0,
                       help="Maximum multiplicative small-object factor before applying small_object_loss_weight.")
    parser.add_argument("--class_balance_loss_weight", type=float, default=0.0,
                       help="Extra object-box weight scaled by inverse sqrt class frequency.")
    parser.add_argument("--boundary_loss_weight", type=float, default=0.0,
                       help="Extra MSE weight on object box boundaries.")
    parser.add_argument("--boundary_width", type=int, default=1,
                       help="Boundary width in latent pixels for boundary_loss_weight.")

    # Logging
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--logging_dir", type=str, default="logs")

    return parser.parse_args()


def cleanup_old_checkpoints(output_dir: str, limit: int):
    """Remove old checkpoints to save disk space"""
    if limit is None or limit <= 0 or not os.path.isdir(output_dir):
        return
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))
    while len(checkpoints) >= limit:
        old = checkpoints.pop(0)
        path = os.path.join(output_dir, old)
        logger.info(f"Removing old checkpoint {path}")
        import shutil
        shutil.rmtree(path, ignore_errors=True)


def setup_lora_processors(transformer, args, device, dtype):
    """Setup LoRA processors for all attention layers"""
    lora_attn_procs = {}
    n_loras = 2 if args.multi_scale_condition else 1
    ranks = [args.lora_rank] * n_loras
    network_alphas = [args.lora_alpha] * n_loras
    lora_weights = [args.lora_weight] * n_loras

    double_blocks_idx = list(range(19))  # FLUX has 19 double blocks
    single_blocks_idx = list(range(38))  # FLUX has 38 single blocks

    for name, attn_processor in transformer.attn_processors.items():
        match = re.search(r'\.(\d+)\.', name)
        if match:
            layer_index = int(match.group(1))
        else:
            layer_index = -1

        if name.startswith("transformer_blocks") and layer_index in double_blocks_idx:
            logger.info(f"Setting LoRA processor for {name}")
            lora_attn_procs[name] = MultiDoubleStreamBlockLoraProcessor(
                dim=3072,
                ranks=ranks,
                network_alphas=network_alphas,
                lora_weights=lora_weights,
                device=device,
                dtype=dtype,
                cond_width=args.cond_size,
                cond_height=args.cond_size,
                n_loras=n_loras,
            )
        elif name.startswith("single_transformer_blocks") and layer_index in single_blocks_idx:
            logger.info(f"Setting LoRA processor for {name}")
            lora_attn_procs[name] = MultiSingleStreamBlockLoraProcessor(
                dim=3072,
                ranks=ranks,
                network_alphas=network_alphas,
                lora_weights=lora_weights,
                device=device,
                dtype=dtype,
                cond_width=args.cond_size,
                cond_height=args.cond_size,
                n_loras=n_loras,
                use_causal_mask=not args.disable_causal_mask,
            )
        else:
            # Keep original processor for other layers
            lora_attn_procs[name] = attn_processor

    transformer.set_attn_processor(lora_attn_procs)

    # Count trainable parameters
    total_params = sum(p.numel() for p in transformer.parameters())
    trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")

    return lora_attn_procs


def save_lora_checkpoint(transformer, output_dir: str, global_step: int):
    """Save only LoRA weights"""
    os.makedirs(output_dir, exist_ok=True)

    # Extract LoRA weights
    state_dict = transformer.state_dict()
    lora_state_dict = {k: v for k, v in state_dict.items() if '_lora' in k or 'lora' in k.lower()}

    save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
    os.makedirs(save_path, exist_ok=True)

    save_file(lora_state_dict, os.path.join(save_path, "lora.safetensors"))
    logger.info(f"Saved LoRA checkpoint to {save_path} ({len(lora_state_dict)} keys)")


def find_latest_lora_checkpoint(output_dir: str) -> Optional[str]:
    """Find the highest-step LoRA checkpoint under output_dir."""
    if not output_dir or not os.path.isdir(output_dir):
        return None

    candidates = []
    search_roots = [Path(output_dir), Path(output_dir) / "final"]
    for root in search_roots:
        if not root.exists():
            continue
        for checkpoint_dir in root.glob("checkpoint-*"):
            lora_path = checkpoint_dir / "lora.safetensors"
            if lora_path.exists():
                candidates.append(str(lora_path))

    if not candidates:
        return None

    return max(candidates, key=infer_step_from_checkpoint_path)


def resolve_lora_checkpoint_path(path: str, output_dir: Optional[str] = None) -> str:
    """Return a lora.safetensors path from either a file or checkpoint directory."""
    if path.lower() in {"auto", "latest"}:
        latest = find_latest_lora_checkpoint(output_dir)
        if latest is None:
            raise FileNotFoundError(f"No LoRA checkpoint found under output_dir: {output_dir}")
        return latest

    if path.endswith(".safetensors"):
        return path

    candidate = os.path.join(path, "lora.safetensors")
    if os.path.exists(candidate):
        return candidate

    raise FileNotFoundError(f"Could not find lora.safetensors from resume path: {path}")


def infer_step_from_checkpoint_path(path: str) -> int:
    """Infer training step from paths like checkpoint-7500/lora.safetensors."""
    matches = re.findall(r"checkpoint-(\d+)", path)
    return int(matches[-1]) if matches else 0


def validate_timestep_range(args):
    if not (0.0 <= args.timestep_min < args.timestep_max <= 1.0):
        raise ValueError(
            f"Expected 0 <= timestep_min < timestep_max <= 1, got "
            f"{args.timestep_min} and {args.timestep_max}"
        )


def compute_class_balance_weights(samples) -> Dict[str, float]:
    """Return inverse-sqrt class-frequency weights normalized around 1."""
    counts: Dict[str, int] = {}
    for sample in samples:
        for label in sample.get("caption", [])[1:]:
            if label:
                counts[label] = counts.get(label, 0) + 1
    if not counts:
        return {}

    mean_count = sum(counts.values()) / float(len(counts))
    weights = {}
    for label, count in counts.items():
        weights[label] = math.sqrt(mean_count / max(float(count), 1.0))
    return weights


def build_object_loss_mask(
    raw_samples,
    prediction_shape,
    device,
    dtype,
    extra_weight: float,
    dilation: int,
    small_object_weight: float = 0.0,
    small_object_reference_area: float = 0.01,
    small_object_weight_max: float = 4.0,
    class_balance_weight: float = 0.0,
    class_weights: Optional[Dict[str, float]] = None,
    boundary_weight: float = 0.0,
    boundary_width: int = 1,
):
    """Build a [B, 1, H, W] MSE multiplier from normalized object boxes.

    ObjLoss-v3 keeps the old uniform box weighting but can add two remote-sensing
    priors: small objects get stronger supervision, and rare classes get a
    modest inverse-frequency boost. Boundary weighting sharpens box placement
    without forcing the entire background to follow the layout condition.
    """
    if (
        extra_weight <= 0
        and small_object_weight <= 0
        and class_balance_weight <= 0
        and boundary_weight <= 0
    ):
        return None

    batch_size, _, height, width = prediction_shape
    mask = torch.ones((batch_size, 1, height, width), device=device, dtype=dtype)
    dilation = max(0, int(dilation))
    boundary_width = max(1, int(boundary_width))
    small_object_reference_area = max(float(small_object_reference_area), 1e-8)
    small_object_weight_max = max(float(small_object_weight_max), 1.0)
    class_weights = class_weights or {}

    for batch_idx, raw in enumerate(raw_samples):
        labels = raw.get("caption", [])[1:]
        boxes = raw.get("bndboxes", [])
        for label, box in zip(labels, boxes):
            if not label:
                continue
            x0, y0, x1, y1 = [float(v) for v in box]
            x0 = max(0.0, min(1.0, x0))
            y0 = max(0.0, min(1.0, y0))
            x1 = max(0.0, min(1.0, x1))
            y1 = max(0.0, min(1.0, y1))
            if x1 <= x0 or y1 <= y0:
                continue

            area = max((x1 - x0) * (y1 - y0), 1e-8)
            object_weight = float(extra_weight)
            if small_object_weight > 0:
                small_factor = math.sqrt(small_object_reference_area / area)
                small_factor = min(max(small_factor, 1.0), small_object_weight_max)
                object_weight += float(small_object_weight) * (small_factor - 1.0)
            if class_balance_weight > 0:
                class_factor = max(float(class_weights.get(str(label), 1.0)), 1.0)
                object_weight += float(class_balance_weight) * (class_factor - 1.0)

            left = max(0, int(math.floor(x0 * width)) - dilation)
            top = max(0, int(math.floor(y0 * height)) - dilation)
            right = min(width, int(math.ceil(x1 * width)) + dilation)
            bottom = min(height, int(math.ceil(y1 * height)) + dilation)
            if right <= left or bottom <= top:
                continue
            if object_weight > 0:
                mask[batch_idx, :, top:bottom, left:right] += object_weight

            if boundary_weight > 0:
                inner_left = min(right, left + boundary_width)
                inner_right = max(left, right - boundary_width)
                inner_top = min(bottom, top + boundary_width)
                inner_bottom = max(top, bottom - boundary_width)
                mask[batch_idx, :, top:inner_top, left:right] += boundary_weight
                mask[batch_idx, :, inner_bottom:bottom, left:right] += boundary_weight
                mask[batch_idx, :, top:bottom, left:inner_left] += boundary_weight
                mask[batch_idx, :, top:bottom, inner_right:right] += boundary_weight

    return mask


def load_lora_checkpoint(transformer, resume_lora_path: str, output_dir: Optional[str] = None):
    """Load LoRA weights saved by save_lora_checkpoint."""
    lora_path = resolve_lora_checkpoint_path(resume_lora_path, output_dir)
    lora_state_dict = load_file(lora_path, device="cpu")

    missing_keys, unexpected_keys = transformer.load_state_dict(lora_state_dict, strict=False)
    loaded_keys = len(lora_state_dict) - len(unexpected_keys)
    resume_step = infer_step_from_checkpoint_path(lora_path)

    logger.info(f"Loaded LoRA checkpoint from {lora_path}")
    logger.info(f"Loaded {loaded_keys}/{len(lora_state_dict)} LoRA tensors")
    if unexpected_keys:
        logger.warning(f"Ignored {len(unexpected_keys)} unexpected LoRA keys")
    lora_missing = [k for k in missing_keys if "lora" in k.lower()]
    if lora_missing:
        logger.warning(f"Missing {len(lora_missing)} LoRA keys while resuming")

    return resume_step


def expand_lora_branches_from_first(transformer, lora_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Copy branch-0 LoRA tensors into any missing extra branches in the current transformer."""
    target_keys = set(transformer.state_dict().keys())
    expanded = dict(lora_state_dict)
    branch_markers = ("q_loras", "k_loras", "v_loras", "proj_loras")
    for key, value in list(lora_state_dict.items()):
        if not any(f"{marker}.0." in key for marker in branch_markers):
            continue
        for branch_idx in range(1, 8):
            new_key = key
            for marker in branch_markers:
                new_key = new_key.replace(f"{marker}.0.", f"{marker}.{branch_idx}.")
            if new_key in target_keys and new_key not in expanded:
                expanded[new_key] = value.clone()
    return expanded


def load_lora_checkpoint_with_optional_expansion(
    transformer,
    resume_lora_path: str,
    output_dir: Optional[str],
    init_extra_loras_from_first: bool,
):
    """Load LoRA weights and optionally initialize newly added condition branches."""
    lora_path = resolve_lora_checkpoint_path(resume_lora_path, output_dir)
    lora_state_dict = load_file(lora_path, device="cpu")
    if init_extra_loras_from_first:
        lora_state_dict = expand_lora_branches_from_first(transformer, lora_state_dict)

    missing_keys, unexpected_keys = transformer.load_state_dict(lora_state_dict, strict=False)
    loaded_keys = len(lora_state_dict) - len(unexpected_keys)
    resume_step = infer_step_from_checkpoint_path(lora_path)

    logger.info(f"Loaded LoRA checkpoint from {lora_path}")
    logger.info(f"Loaded {loaded_keys}/{len(lora_state_dict)} LoRA tensors")
    if init_extra_loras_from_first:
        logger.info("Initialized missing extra condition LoRA branches from branch 0 where possible")
    if unexpected_keys:
        logger.warning(f"Ignored {len(unexpected_keys)} unexpected LoRA keys")
    lora_missing = [k for k in missing_keys if "lora" in k.lower()]
    if lora_missing:
        logger.warning(f"Missing {len(lora_missing)} LoRA keys while resuming")

    return resume_step


def main():
    args = parse_args()
    validate_timestep_range(args)
    if isinstance(args.report_to, str) and args.report_to.lower() in {"none", "null", ""}:
        args.report_to = None

    # Setup accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=args.report_to,
        project_config=ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir),
    )

    if args.seed is not None:
        set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    if args.resume_lora_path is None and not args.no_auto_resume:
        latest_lora = find_latest_lora_checkpoint(args.output_dir)
        if latest_lora is not None:
            args.resume_lora_path = latest_lora

    # Determine dtype
    weight_dtype = torch.bfloat16 if args.torch_dtype == "bf16" else \
                   torch.float16 if args.torch_dtype == "fp16" else torch.float32

    # Load FLUX pipeline
    logger.info("Loading RS-FLUX pipeline...")
    pipe = FluxPipeline.from_pretrained(args.pretrained_model_name_or_path, torch_dtype=weight_dtype)

    # Freeze base model
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)
    pipe.transformer.requires_grad_(True)  # Will set specific layers trainable

    pipe.vae.eval()
    pipe.text_encoder.eval()
    pipe.text_encoder_2.eval()

    # Setup LoRA processors
    setup_lora_processors(pipe.transformer, args, accelerator.device, weight_dtype)
    resume_global_step = 0
    if args.resume_lora_path:
        resume_global_step = load_lora_checkpoint_with_optional_expansion(
            pipe.transformer,
            args.resume_lora_path,
            args.output_dir,
            args.init_extra_loras_from_first,
        )

    # Only LoRA parameters are trainable
    for name, param in pipe.transformer.named_parameters():
        if '_lora' in name or 'lora' in name.lower():
            param.requires_grad = True
        else:
            param.requires_grad = False

    if args.gradient_checkpointing:
        pipe.transformer.enable_gradient_checkpointing()

    pipe.transformer.train()

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in pipe.transformer.parameters() if p.requires_grad],
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset
    render_style = LayoutRenderStyle[args.render_style.upper()]
    train_dataset = RSLayoutDataset(
        args.train_data_dir,
        resolution=args.resolution,
        render_style=render_style,
        prompt_mode=args.prompt_mode,
        max_samples=args.max_train_samples,
        draw_arrows=args.draw_arrows,
    )
    class_balance_weights = compute_class_balance_weights(train_dataset.samples)
    if args.class_balance_loss_weight > 0 and accelerator.is_main_process:
        top_weights = sorted(class_balance_weights.items(), key=lambda item: item[1], reverse=True)[:5]
        logger.info(f"Class-balance loss enabled; largest weights: {top_weights}")

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_rslayout_batch,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler. In distributed training each rank only sees roughly
    # len(dataloader) / num_processes batches per epoch after accelerator.prepare.
    num_batches_per_epoch = math.ceil(len(train_dataloader) / accelerator.num_processes)
    num_update_steps_per_epoch = math.ceil(num_batches_per_epoch / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare for distributed training
    pipe.transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        pipe.transformer, optimizer, train_dataloader, lr_scheduler
    )

    pipe.vae.to(accelerator.device, dtype=weight_dtype)
    pipe.text_encoder.to(accelerator.device, dtype=weight_dtype)
    pipe.text_encoder_2.to(accelerator.device, dtype=weight_dtype)

    if resume_global_step > 0:
        for _ in range(resume_global_step):
            lr_scheduler.step()

    # Training info
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size = {args.train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    if resume_global_step > 0:
        logger.info(f"  Resuming LoRA from global step = {resume_global_step}")

    if accelerator.is_main_process:
        accelerator.init_trackers("rslayout-dit", config=vars(args))
        with open(os.path.join(args.output_dir, "training_config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    global_step = resume_global_step
    progress_bar = tqdm(total=args.max_train_steps, initial=global_step, disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    interrupted = {"flag": False, "signal": None}

    def _handle_interrupt(signum, frame):
        interrupted["flag"] = True
        interrupted["signal"] = signum
        if accelerator.is_main_process:
            logger.warning(f"Received signal {signum}; will save a checkpoint and stop after this step.")

    signal.signal(signal.SIGTERM, _handle_interrupt)
    signal.signal(signal.SIGINT, _handle_interrupt)

    # Training loop
    try:
        for epoch in range(args.num_train_epochs):
            for batch in train_dataloader:
                with accelerator.accumulate(pipe.transformer):
                    # Get images and conditions
                    pixel_values = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
                    cond_pixel_values = batch["cond_pixel_values"].to(accelerator.device, dtype=weight_dtype)

                    with torch.no_grad(), accelerator.autocast():
                        # Encode images to latents
                        latents = pipe.vae.encode(pixel_values).latent_dist.sample()
                        latents = (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor

                        # Encode condition images to latents
                        cond_latents = pipe.vae.encode(cond_pixel_values).latent_dist.sample()
                        cond_latents = (cond_latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
                        cond_latents_list = [cond_latents]

                        if args.multi_scale_condition:
                            coarse_cond = F.interpolate(
                                cond_pixel_values.float(),
                                size=(args.coarse_cond_size, args.coarse_cond_size),
                                mode="bilinear",
                                align_corners=False,
                            )
                            coarse_cond = F.interpolate(
                                coarse_cond,
                                size=(args.resolution, args.resolution),
                                mode="bilinear",
                                align_corners=False,
                            ).to(dtype=weight_dtype)
                            coarse_cond_latents = pipe.vae.encode(coarse_cond).latent_dist.sample()
                            coarse_cond_latents = (
                                coarse_cond_latents - pipe.vae.config.shift_factor
                            ) * pipe.vae.config.scaling_factor
                            cond_latents_list.append(coarse_cond_latents)

                        # Add noise
                        noise = torch.randn_like(latents)
                        bsz = latents.shape[0]

                        # Sample timesteps
                        u = compute_density_for_timestep_sampling(
                            weighting_scheme=args.weighting_scheme,
                            batch_size=bsz,
                            logit_mean=args.logit_mean,
                            logit_std=args.logit_std,
                            mode_scale=args.mode_scale,
                            device=latents.device,
                        )
                        if args.timestep_min != 0.0 or args.timestep_max != 1.0:
                            u = args.timestep_min + (args.timestep_max - args.timestep_min) * u
                        sigmas = u.to(dtype=latents.dtype)
                        while sigmas.ndim < latents.ndim:
                            sigmas = sigmas.unsqueeze(-1)

                        noisy_latents = (1.0 - sigmas) * latents + sigmas * noise

                        # Pack latents (FLUX specific)
                        packed_noisy_latents = pipe._pack_latents(
                            noisy_latents, bsz, latents.shape[1], latents.shape[2], latents.shape[3]
                        )
                        packed_cond_latents_list = [
                            pipe._pack_latents(
                                item, bsz, item.shape[1], item.shape[2], item.shape[3]
                            )
                            for item in cond_latents_list
                        ]

                        # Prepare latent IDs for image
                        latent_image_ids = pipe._prepare_latent_image_ids(
                            bsz, latents.shape[2] // 2, latents.shape[3] // 2,
                            device=latents.device, dtype=latents.dtype
                        )

                        # Prepare latent IDs for condition
                        cond_image_ids_list = [
                            pipe._prepare_latent_image_ids(
                                bsz, item.shape[2] // 2, item.shape[3] // 2,
                                device=item.device, dtype=item.dtype
                            )
                            for item in cond_latents_list
                        ]

                        # Concatenate image and condition IDs
                        combined_img_ids = torch.cat([latent_image_ids] + cond_image_ids_list, dim=0)

                        # Encode prompts
                        t5_prompts = batch["clip_prompts"] if args.t5_prompt_from_clip_prompt else batch["prompts"]
                        prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
                            prompt=batch["clip_prompts"],
                            prompt_2=t5_prompts,
                            device=latents.device,
                            num_images_per_prompt=1,
                            max_sequence_length=args.max_sequence_length,
                        )

                        # Prepare timestep
                        timestep = u.to(device=latents.device, dtype=packed_noisy_latents.dtype)

                        # Guidance
                        guidance = None
                        transformer_config = accelerator.unwrap_model(pipe.transformer).config
                        if transformer_config.guidance_embeds:
                            guidance = torch.full(
                                (bsz,), args.guidance_scale,
                                device=packed_noisy_latents.device,
                                dtype=packed_noisy_latents.dtype
                            )

                        # Concatenate image and condition latents (RSLayout-DiT approach)
                        combined_latents = torch.cat([packed_noisy_latents] + packed_cond_latents_list, dim=1)

                    # Forward pass with LoRA
                    with accelerator.autocast():
                        # Use unwrapped model for forward pass
                        transformer = accelerator.unwrap_model(pipe.transformer)
                        combined_pred = transformer(
                            hidden_states=combined_latents,
                            timestep=timestep,
                            guidance=guidance,
                            pooled_projections=pooled_prompt_embeds,
                            encoder_hidden_states=prompt_embeds,
                            txt_ids=text_ids,
                            img_ids=combined_img_ids,
                            return_dict=False,
                        )[0]

                        # Extract only the image prediction (first part)
                        model_pred = combined_pred[:, :packed_noisy_latents.shape[1], :]

                        # Unpack
                        model_pred = pipe._unpack_latents(
                            model_pred, args.resolution, args.resolution, pipe.vae_scale_factor
                        )

                        # Compute loss
                        target = noise - latents
                        loss_values = (model_pred.float() - target.float()) ** 2

                        if args.weighting_scheme == "cosmap":
                            loss_weights = compute_loss_weighting_for_sd3(args.weighting_scheme, u.to(model_pred.device, dtype=model_pred.dtype))
                            while loss_weights.ndim < model_pred.ndim:
                                loss_weights = loss_weights.unsqueeze(-1)
                            loss_values = loss_weights.float() * loss_values

                        object_loss_mask = build_object_loss_mask(
                            batch["raw"],
                            model_pred.shape,
                            model_pred.device,
                            torch.float32,
                            args.object_loss_weight,
                            args.object_mask_dilation,
                            small_object_weight=args.small_object_loss_weight,
                            small_object_reference_area=args.small_object_reference_area,
                            small_object_weight_max=args.small_object_weight_max,
                            class_balance_weight=args.class_balance_loss_weight,
                            class_weights=class_balance_weights,
                            boundary_weight=args.boundary_loss_weight,
                            boundary_width=args.boundary_width,
                        )
                        if object_loss_mask is not None:
                            loss_values = loss_values * object_loss_mask
                            loss = loss_values.mean() / object_loss_mask.mean().clamp_min(1.0)
                        else:
                            loss = loss_values.mean()

                    # Backward
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(pipe.transformer.parameters(), args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    # Update progress
                    if accelerator.sync_gradients:
                        global_step += 1
                        progress_bar.update(1)
                        accelerator.log({"train_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}, step=global_step)

                        # Save checkpoint
                        if global_step % args.checkpointing_steps == 0:
                            if accelerator.is_main_process:
                                cleanup_old_checkpoints(args.output_dir, args.checkpoints_total_limit)
                                save_lora_checkpoint(accelerator.unwrap_model(pipe.transformer), args.output_dir, global_step)

                        if interrupted["flag"]:
                            if accelerator.is_main_process:
                                interrupt_step = global_step
                                save_lora_checkpoint(accelerator.unwrap_model(pipe.transformer), args.output_dir, interrupt_step)
                                logger.warning(f"Interrupted at global step {interrupt_step}, checkpoint saved.")
                            break

                        if global_step >= args.max_train_steps:
                            break

            if interrupted["flag"] or global_step >= args.max_train_steps:
                break
    finally:
        # Save final checkpoint
        if accelerator.is_main_process:
            final_dir = os.path.join(args.output_dir, "final")
            save_lora_checkpoint(accelerator.unwrap_model(pipe.transformer), final_dir, global_step)
            logger.info(f"Saved final LoRA to {final_dir}")

        accelerator.end_training()


if __name__ == "__main__":
    main()
