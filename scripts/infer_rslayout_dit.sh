#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL_PATH="${MODEL_PATH:-/path/to/FLUX-RS}"
DATA_ROOT="${DATA_ROOT:-path_to_data/DIOR}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs-rslayout-dit}"
GPUS="${GPUS:-0}"
NUM_SAMPLES="${NUM_SAMPLES:-8}"
CHECKPOINT="${1:-checkpoint-rslayout-dit/checkpoint-1000/lora.safetensors}"

CUDA_VISIBLE_DEVICES="$GPUS" python infer_rslayout_dit.py \
  --pretrained_model_name_or_path="$MODEL_PATH" \
  --lora_path="$CHECKPOINT" \
  --data_dir="$DATA_ROOT/val" \
  --num_samples="$NUM_SAMPLES" \
  --resolution=512 \
  --num_inference_steps=28 \
  --guidance_scale=3.5 \
  --lora_weight=1.0 \
  --render_style=colored_polygons \
  --draw_arrows \
  --output_dir="$OUTPUT_DIR" \
  --save_layout \
  --seed=42 \
  --torch_dtype=bf16

echo ""
echo "Inference complete! Check outputs in: $OUTPUT_DIR/"
