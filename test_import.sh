#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "Testing imports..."
python -c "from rslayout_dit import *; print('All imports successful!')"

if [ $? -eq 0 ]; then
    echo ""
    echo "Import check passed. Ready to train."
    echo ""
    echo "Run training with:"
    echo "  bash scripts/train_rslayout_dit_multigpu.sh"
else
    echo "Import failed. Check error above."
    exit 1
fi
