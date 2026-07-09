#!/usr/bin/env python
"""Prepare a full DOTA train split for RS-FLUX RSLayout-DiT.

This reuses the repository's DOTA tiling code but skips CLIP embeddings and
foreground crops, which are not needed by train_rslayout_dit.py.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prepare_dota_dataset import split_mode


def replace_symlink(link_path: Path, target: Path) -> None:
    if link_path.is_symlink() or link_path.exists():
        if link_path.is_dir() and not link_path.is_symlink():
            raise RuntimeError(f"Refusing to replace real directory: {link_path}")
        link_path.unlink()
    link_path.symlink_to(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", default="/path/to/DOTA/extracted")
    parser.add_argument("--out_root", default="path_to_data/DOTA_fulltrain_fullval")
    parser.add_argument("--val_source", default="path_to_data/DOTA_fullval/val")
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--gap", type=int, default=100)
    parser.add_argument("--max_objects", type=int, default=6)
    parser.add_argument("--activate", action="store_true", help="Point path_to_data/DOTA to the prepared dataset.")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    train_meta = out_root / "train" / "metadata.jsonl"
    if train_meta.exists():
        count = sum(1 for _ in train_meta.open())
        print(f"train_metadata_exists={count}")
    else:
        count = split_mode(raw_root, out_root, "train", args.crop_size, args.gap, args.max_objects)
        print(f"train_metadata_created={count}")

    val_link = out_root / "val"
    val_source = Path(args.val_source)
    if not val_link.exists():
        val_link.symlink_to(os.path.relpath(val_source.resolve(), out_root.resolve()))
        print(f"val_link_created={val_link} -> {val_source.resolve()}")
    else:
        print(f"val_exists={val_link}")

    if args.activate:
        replace_symlink(Path("path_to_data/DOTA"), Path(out_root.name))
        print(f"activated=path_to_data/DOTA -> {out_root.name}")


if __name__ == "__main__":
    main()
