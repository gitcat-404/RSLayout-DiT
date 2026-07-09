#!/usr/bin/env python
import argparse
import json
import math
import os
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import FluxPipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from rslayout_utils.data import DiorLayoutDataset, collate_layout_batch
from rslayout_utils.layout_condition import channel_count
from rslayout_utils.models import (
    create_flux_controlnet,
    dtype_from_string,
    metadata_for_training,
    save_controlnet,
    save_controlnet_trainable,
)


logger = get_logger(__name__, log_level="INFO")
warnings.filterwarnings("ignore", message="`FluxPosEmbed` is deprecated.*", category=FutureWarning)


def parse_args():
    parser = argparse.ArgumentParser("Train the FLUX-ControlNet layout-control ablation on RSLayout metadata.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="/path/to/FLUX-RS")
    parser.add_argument("--train_data_dir", type=str, default="path_to_data/DIOR/train")
    parser.add_argument("--validation_data_dir", type=str, default="path_to_data/DIOR/val")
    parser.add_argument("--output_dir", type=str, default="checkpoint-rslayout-controlnet")
    parser.add_argument(
        "--method",
        type=str,
        choices=["simple_control", "layout_control", "layout_control_v2"],
        default="layout_control",
    )
    parser.add_argument("--prompt_mode", type=str, choices=["scene", "scene_with_objects"], default="scene")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--checkpointing_steps", type=int, default=300)
    parser.add_argument("--checkpoints_total_limit", type=int, default=3)
    parser.add_argument(
        "--save_full_controlnet",
        action="store_true",
        help="Save the full 5GB ControlNet. By default only trainable control layers are saved.",
    )
    parser.add_argument(
        "--resume_controlnet_path",
        type=str,
        default=None,
        help="Path to a saved controlnet directory, e.g. output_dir/checkpoint-4000/controlnet.",
    )
    parser.add_argument(
        "--resume_global_step",
        type=int,
        default=None,
        help="Global step to continue counting from. If omitted, parsed from checkpoint-N in resume_controlnet_path.",
    )
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch_dtype", type=str, choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--controlnet_num_layers", type=int, default=4)
    parser.add_argument("--controlnet_num_single_layers", type=int, default=8)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--weighting_scheme", type=str, choices=["none", "logit_normal", "mode", "cosmap"], default="none")
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)
    parser.add_argument("--mode_scale", type=float, default=1.29)
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument(
        "--train_controlnet_backbone",
        action="store_true",
        help="Also train copied FLUX blocks inside ControlNet. By default only layout hint and zero-output control layers are trained.",
    )
    return parser.parse_args()


def cleanup_old_checkpoints(output_dir: str, limit: int):
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


def encode_images_to_packed_latents(pipe: FluxPipeline, pixel_values: torch.Tensor):
    latents = pipe.vae.encode(pixel_values).latent_dist.sample()
    shift = getattr(pipe.vae.config, "shift_factor", 0.0)
    latents = (latents - shift) * pipe.vae.config.scaling_factor
    bsz, channels, height, width = latents.shape
    packed = pipe._pack_latents(latents, bsz, channels, height, width)
    latent_image_ids = pipe._prepare_latent_image_ids(
        bsz,
        height // 2,
        width // 2,
        device=latents.device,
        dtype=latents.dtype,
    )
    return latents, packed, latent_image_ids


def pack_like_flux(pipe: FluxPipeline, tensor: torch.Tensor):
    bsz, channels, height, width = tensor.shape
    return pipe._pack_latents(tensor, bsz, channels, height, width)


def save_training_checkpoint(accelerator, controlnet, args, input_channels: int, global_step: int):
    if not accelerator.is_main_process:
        return
    cleanup_old_checkpoints(args.output_dir, args.checkpoints_total_limit)
    ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}", "controlnet")
    unwrapped = accelerator.unwrap_model(controlnet)
    metadata = metadata_for_training(args, input_channels)
    metadata["global_step"] = global_step
    if args.save_full_controlnet:
        save_controlnet(unwrapped, ckpt_dir, metadata)
    else:
        save_controlnet_trainable(unwrapped, ckpt_dir, metadata)
    logger.info(f"Saved controlnet checkpoint to {ckpt_dir}")


def configure_controlnet_trainable(controlnet, train_backbone: bool = False):
    trainable_prefixes = ("input_hint_block", "controlnet_x_embedder", "controlnet_blocks", "controlnet_single_blocks")
    for name, param in controlnet.named_parameters():
        param.requires_grad = train_backbone or name.startswith(trainable_prefixes)


def count_parameters(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def infer_resume_step(path: str) -> int:
    if not path:
        return 0
    parts = Path(path).parts
    for part in reversed(parts):
        if part.startswith("checkpoint-"):
            try:
                return int(part.split("-")[-1])
            except ValueError:
                return 0
    return 0


def load_controlnet_weights(controlnet, controlnet_path: str):
    safetensors_path = os.path.join(controlnet_path, "diffusion_pytorch_model.safetensors")
    bin_path = os.path.join(controlnet_path, "pytorch_model.bin")
    if os.path.exists(safetensors_path):
        state_dict = load_file(safetensors_path, device="cpu")
    elif os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"No controlnet weights found under {controlnet_path}")
    metadata_path = os.path.join(controlnet_path, "rslayout_utils_config.json")
    trainable_only = False
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            trainable_only = bool(json.load(f).get("trainable_only", False))
    missing, unexpected = controlnet.load_state_dict(state_dict, strict=False)
    if missing and not trainable_only:
        raise RuntimeError(f"Missing keys while resuming controlnet: {missing[:8]}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys while resuming controlnet: {unexpected[:8]}")


def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=args.report_to,
        project_config=ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir),
    )
    if args.seed is not None:
        set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    weight_dtype = dtype_from_string(args.torch_dtype)
    input_channels = channel_count(args.method)

    logger.info("Loading RS-FLUX pipeline.")
    pipe = FluxPipeline.from_pretrained(args.pretrained_model_name_or_path, torch_dtype=weight_dtype)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)
    pipe.transformer.requires_grad_(False)
    pipe.vae.eval()
    pipe.text_encoder.eval()
    pipe.text_encoder_2.eval()
    pipe.transformer.eval()

    controlnet = create_flux_controlnet(
        pipe.transformer,
        method=args.method,
        input_channels=input_channels,
        num_layers=args.controlnet_num_layers,
        num_single_layers=args.controlnet_num_single_layers,
    )
    if args.resume_controlnet_path is not None:
        logger.info(f"Loading controlnet checkpoint from {args.resume_controlnet_path}")
        load_controlnet_weights(controlnet, args.resume_controlnet_path)
    controlnet.to(dtype=weight_dtype)
    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()
    configure_controlnet_trainable(controlnet, train_backbone=args.train_controlnet_backbone)
    total_params, trainable_params = count_parameters(controlnet)
    logger.info(f"ControlNet parameters: total={total_params:,}, trainable={trainable_params:,}")
    controlnet.train()

    optimizer = torch.optim.AdamW(
        [p for p in controlnet.parameters() if p.requires_grad],
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataset = DiorLayoutDataset(
        args.train_data_dir,
        resolution=args.resolution,
        control_mode=args.method,
        prompt_mode=args.prompt_mode,
        max_samples=args.max_train_samples,
    )
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_layout_batch,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_dataloader, lr_scheduler
    )
    pipe.vae.to(accelerator.device, dtype=weight_dtype)
    pipe.text_encoder.to(accelerator.device, dtype=weight_dtype)
    pipe.text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    pipe.transformer.to(accelerator.device, dtype=weight_dtype)

    global_step = args.resume_global_step if args.resume_global_step is not None else infer_resume_step(args.resume_controlnet_path)
    num_update_steps_per_epoch = max(1, math.ceil(len(train_dataloader) / args.gradient_accumulation_steps))
    remaining_steps = max(args.max_train_steps - global_step, 0)
    args.num_train_epochs = max(1, math.ceil(remaining_steps / num_update_steps_per_epoch))
    logger.info(
        f"Training schedule: global_step={global_step}, max_train_steps={args.max_train_steps}, "
        f"remaining_steps={remaining_steps}, update_steps_per_epoch={num_update_steps_per_epoch}, "
        f"num_train_epochs={args.num_train_epochs}"
    )

    if accelerator.is_main_process:
        accelerator.init_trackers("rslayout-controlnet", config=vars(args))
        with open(os.path.join(args.output_dir, "training_config.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2)

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    if global_step > 0:
        logger.info(f"Resuming global_step counter from {global_step}")
        progress_bar.update(min(global_step, args.max_train_steps))

    for epoch in range(args.num_train_epochs):
        for batch in train_dataloader:
            with accelerator.accumulate(controlnet):
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
                control = batch["control"].to(accelerator.device, dtype=weight_dtype)

                with torch.no_grad(), accelerator.autocast():
                    latents, _, latent_image_ids = encode_images_to_packed_latents(pipe, pixel_values)
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    u = compute_density_for_timestep_sampling(
                        weighting_scheme=args.weighting_scheme,
                        batch_size=bsz,
                        logit_mean=args.logit_mean,
                        logit_std=args.logit_std,
                        mode_scale=args.mode_scale,
                        device=latents.device,
                    )
                    sigmas = u.to(dtype=latents.dtype)
                    while sigmas.ndim < latents.ndim:
                        sigmas = sigmas.unsqueeze(-1)
                    noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
                    packed_noisy_latents = pack_like_flux(pipe, noisy_latents)
                    target = pack_like_flux(pipe, noise - latents)
                    model_timestep = u.to(device=latents.device, dtype=packed_noisy_latents.dtype)

                    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
                        prompt=batch["clip_prompts"],
                        prompt_2=batch["prompts"],
                        device=latents.device,
                        num_images_per_prompt=1,
                        max_sequence_length=args.max_sequence_length,
                    )

                with accelerator.autocast():
                    guidance = None
                    controlnet_config = accelerator.unwrap_model(controlnet).config
                    if controlnet_config.guidance_embeds:
                        guidance = torch.full(
                            (packed_noisy_latents.shape[0],),
                            args.guidance_scale,
                            device=packed_noisy_latents.device,
                            dtype=packed_noisy_latents.dtype,
                        )

                    controlnet_block_samples, controlnet_single_block_samples = controlnet(
                        hidden_states=packed_noisy_latents,
                        controlnet_cond=control,
                        conditioning_scale=1.0,
                        timestep=model_timestep,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        return_dict=False,
                    )
                    model_pred = pipe.transformer(
                        hidden_states=packed_noisy_latents,
                        timestep=model_timestep,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        controlnet_block_samples=controlnet_block_samples,
                        controlnet_single_block_samples=controlnet_single_block_samples,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        return_dict=False,
                        controlnet_blocks_repeat=True,
                    )[0]

                    if args.weighting_scheme == "cosmap":
                        flat_sigmas = u.to(model_pred.device, dtype=model_pred.dtype)
                        loss_weights = compute_loss_weighting_for_sd3(args.weighting_scheme, flat_sigmas)
                        while loss_weights.ndim < model_pred.ndim:
                            loss_weights = loss_weights.unsqueeze(-1)
                        loss = torch.mean((loss_weights * (model_pred.float() - target.float()) ** 2))
                    else:
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(controlnet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                accelerator.log({"train_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}, step=global_step)
                if global_step % args.checkpointing_steps == 0:
                    save_training_checkpoint(accelerator, controlnet, args, input_channels, global_step)
                if global_step >= args.max_train_steps:
                    break
        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = os.path.join(args.output_dir, "controlnet")
        final_metadata = metadata_for_training(args, input_channels)
        final_metadata["global_step"] = global_step
        if args.save_full_controlnet:
            save_controlnet(accelerator.unwrap_model(controlnet), final_dir, final_metadata)
        else:
            save_controlnet_trainable(
                accelerator.unwrap_model(controlnet), final_dir, final_metadata
            )
        logger.info(f"Saved final controlnet to {final_dir}")
    accelerator.end_training()


if __name__ == "__main__":
    main()
