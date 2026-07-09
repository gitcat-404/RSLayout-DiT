#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL_PATH="${MODEL_PATH:-/path/to/FLUX-RS}"
DATA_ROOT="${DATA_ROOT:-path_to_data/DIOR}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoint-rslayout-dit}"
GPUS="${GPUS:-0}"

CUDA_VISIBLE_DEVICES="$GPUS" python train_rslayout_dit.py \
  --pretrained_model_name_or_path="$MODEL_PATH" \
  --train_data_dir="$DATA_ROOT/train" \
  --validation_data_dir="$DATA_ROOT/val" \
  --resolution=512 \
  --cond_size=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --num_train_epochs=10 \
  --learning_rate=1e-4 \
  --max_grad_norm=1.0 \
  --lr_scheduler=constant \
  --lr_warmup_steps=500 \
  --checkpointing_steps=500 \
  --checkpoints_total_limit=3 \
  --output_dir="$OUTPUT_DIR" \
  --lora_rank=128 \
  --lora_alpha=128 \
  --lora_weight=1.0 \
  --render_style=colored_polygons \
  --draw_arrows \
  --prompt_mode=satellite \
  --seed=42 \
  --torch_dtype=bf16 \
  --dataloader_num_workers=4 \
  --report_to=tensorboard \
  --logging_dir=logs
