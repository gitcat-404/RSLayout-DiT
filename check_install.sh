#!/usr/bin/env bash
set -euo pipefail
# Quick start script for RS-FLUX RSLayout-DiT

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Run test
echo "Running component tests..."
python test_rslayout_dit.py

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Start training:"
echo "   bash scripts/train_rslayout_dit.sh"
echo ""
echo "2. Or run training manually:"
echo "   CUDA_VISIBLE_DEVICES=0 python train_rslayout_dit.py \\"
echo "     --train_data_dir=path_to_data/DIOR/train \\"
echo "     --output_dir=checkpoint-rslayout-dit"
echo ""
echo "3. Run inference after training:"
echo "   python infer_rslayout_dit.py \\"
echo "     --lora_path=checkpoint-rslayout-dit/checkpoint-1000/lora.safetensors"
echo ""
