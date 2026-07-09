#!/usr/bin/env python
import argparse
import json
import os

from rslayout_utils.layout_condition import parse_layout_objects, prompt_from_caption
from rslayout_utils.prompting import build_satellite_prompts, legacy_clip_prompt


def parse_args():
    parser = argparse.ArgumentParser("Create old/new RS-FLUX prompt backup files for DIOR metadata.")
    parser.add_argument("--data_dir", type=str, default="path_to_data/DIOR")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val"])
    parser.add_argument("--output_name", type=str, default="metadata_rslayout_prompts.jsonl")
    return parser.parse_args()


def process_split(split_dir: str, output_name: str):
    metadata_path = os.path.join(split_dir, "metadata.jsonl")
    output_path = os.path.join(split_dir, output_name)
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(metadata_path)

    count = 0
    with open(metadata_path, "r", encoding="utf-8") as src, open(output_path, "w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            sample = json.loads(line)
            objects = parse_layout_objects(sample["caption"], sample["bndboxes"], sample["obboxes"])
            old_t5 = prompt_from_caption(sample["caption"], "scene")
            old_clip = legacy_clip_prompt(old_t5, objects)
            new_clip, new_t5 = build_satellite_prompts(sample["caption"], objects)
            record = {
                "file_name": sample["file_name"],
                "objects": [obj.label for obj in objects],
                "old_clip_prompt": old_clip,
                "old_t5_prompt": old_t5,
                "new_clip_prompt": new_clip,
                "new_t5_prompt": new_t5,
            }
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return output_path, count


def main():
    args = parse_args()
    for split in args.splits:
        output_path, count = process_split(os.path.join(args.data_dir, split), args.output_name)
        print(f"Wrote {count} prompt records to {output_path}")


if __name__ == "__main__":
    main()
