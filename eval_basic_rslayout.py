#!/usr/bin/env python
import argparse
import json
import os
from collections import Counter

from rslayout_utils.layout_condition import parse_layout_objects


def parse_args():
    parser = argparse.ArgumentParser("Basic dataset/result bookkeeping for RS-FLUX layout-control experiments.")
    parser.add_argument("--metadata", type=str, default="path_to_data/DIOR/val/metadata.jsonl")
    parser.add_argument("--generated_dir", type=str, default=None)
    parser.add_argument("--output_json", type=str, default="rslayout_utils_basic_eval.json")
    return parser.parse_args()


def main():
    args = parse_args()
    samples = []
    class_counter = Counter()
    object_count = 0
    with open(args.metadata, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            sample = json.loads(line)
            objects = parse_layout_objects(sample["caption"], sample["bndboxes"], sample["obboxes"])
            samples.append(sample)
            object_count += len(objects)
            class_counter.update([obj.label for obj in objects])

    result = {
        "num_metadata_samples": len(samples),
        "num_layout_objects": object_count,
        "class_histogram": dict(class_counter),
    }
    if args.generated_dir:
        generated = [name for name in os.listdir(args.generated_dir) if name.endswith("_generated.png")]
        layout = [name for name in os.listdir(args.generated_dir) if name.endswith("_layout.png")]
        comparison = [name for name in os.listdir(args.generated_dir) if name.endswith("_comparison.png")]
        result.update(
            {
                "generated_dir": args.generated_dir,
                "num_generated_images": len(generated),
                "num_layout_images": len(layout),
                "num_comparison_images": len(comparison),
            }
        )

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
