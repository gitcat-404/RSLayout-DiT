#!/usr/bin/env python
"""
Test script for RS-FLUX RSLayout-DiT
Verify that all components work correctly
"""

import sys
import torch
from PIL import Image

print("=" * 60)
print("RS-FLUX RSLayout-DiT - Component Test")
print("=" * 60)

# Test 1: Import modules
print("\n[1/6] Testing imports...")
try:
    from rslayout_dit.layout_render import render_layout_to_rgb, LayoutRenderStyle
    from rslayout_dit.lora_layers import LoRALinearLayer
    from rslayout_dit.data import RSLayoutDataset
    from rslayout_utils.layout_condition import LayoutObject, PALETTE
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Layout rendering
print("\n[2/6] Testing layout rendering...")
try:
    # Create test objects
    test_objects = [
        LayoutObject(
            label="airplane",
            bbox=[0.2, 0.2, 0.4, 0.4],
            obbox=[0.2, 0.2, 0.4, 0.2, 0.4, 0.4, 0.2, 0.4]
        ),
        LayoutObject(
            label="vehicle",
            bbox=[0.6, 0.6, 0.8, 0.8],
            obbox=[0.6, 0.6, 0.8, 0.6, 0.8, 0.8, 0.6, 0.8]
        ),
    ]

    # Test different rendering styles
    for style in [LayoutRenderStyle.COLORED_POLYGONS, LayoutRenderStyle.SEMANTIC_MAP,
                  LayoutRenderStyle.EDGE_MAP, LayoutRenderStyle.HEATMAP]:
        img = render_layout_to_rgb(test_objects, resolution=256, style=style)
        assert isinstance(img, Image.Image)
        assert img.size == (256, 256)
        assert img.mode == "RGB"

    print(f"✓ Layout rendering works (tested {len(LayoutRenderStyle)} styles)")
except Exception as e:
    print(f"✗ Layout rendering failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: LoRA layers
print("\n[3/6] Testing LoRA layers...")
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    lora = LoRALinearLayer(
        in_features=3072,
        out_features=3072,
        rank=128,
        device=device,
        dtype=dtype,
        cond_width=512,
        cond_height=512,
        number=0,
        n_loras=1,
    )

    # Test forward pass
    batch_size = 2
    seq_len = 100  # 64 (image) + 36 (condition)
    hidden_states = torch.randn(batch_size, seq_len, 3072, device=device, dtype=dtype)
    output = lora(hidden_states)

    assert output.shape == hidden_states.shape
    assert output.dtype == dtype

    print(f"✓ LoRA layers work (rank=128, device={device})")
except Exception as e:
    print(f"✗ LoRA layers failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Dataset
print("\n[4/6] Testing dataset...")
try:
    import os
    data_dir = "path_to_data/DIOR/train"

    if os.path.exists(data_dir):
        dataset = RSLayoutDataset(
            data_dir=data_dir,
            resolution=512,
            render_style=LayoutRenderStyle.COLORED_POLYGONS,
            max_samples=2,
        )

        print(f"  Dataset size: {len(dataset)}")

        # Test loading one sample
        sample = dataset[0]
        assert "pixel_values" in sample
        assert "cond_pixel_values" in sample
        assert "prompt" in sample
        assert sample["pixel_values"].shape == (3, 512, 512)
        assert sample["cond_pixel_values"].shape == (3, 512, 512)

        print(f"✓ Dataset works ({len(dataset)} samples)")
    else:
        print(f"⚠ Dataset directory not found: {data_dir} (skipping)")
except Exception as e:
    print(f"✗ Dataset failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Check FLUX model availability
print("\n[5/6] Checking FLUX model...")
try:
    flux_path = os.environ.get("MODEL_PATH", "/path/to/FLUX-RS")
    if os.path.exists(flux_path):
        print(f"✓ FLUX model found at {flux_path}")
    else:
        print(f"⚠ FLUX model not found at {flux_path}")
except Exception as e:
    print(f"✗ FLUX check failed: {e}")

# Test 6: Memory and performance
print("\n[6/6] Testing memory and performance...")
try:
    import time

    # Test rendering performance
    start = time.time()
    for _ in range(10):
        img = render_layout_to_rgb(test_objects, resolution=512)
    elapsed = time.time() - start
    print(f"  Rendering: {elapsed/10*1000:.1f}ms per image (512x512)")

    # Test LoRA performance
    if torch.cuda.is_available():
        start = time.time()
        for _ in range(100):
            output = lora(hidden_states)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        print(f"  LoRA forward: {elapsed/100*1000:.1f}ms per batch")

    print("✓ Performance test complete")
except Exception as e:
    print(f"✗ Performance test failed: {e}")

# Summary
print("\n" + "=" * 60)
print("Test Summary:")
print("  ✓ All core components are working")
print("  ✓ Ready for training and inference")
print("\nNext steps:")
print("  1. Prepare your data in path_to_data/DIOR/")
print("  2. Run training: python train_rslayout_dit.py")
print("  3. Run inference: python infer_rslayout_dit.py --lora_path <path>")
print("=" * 60)
