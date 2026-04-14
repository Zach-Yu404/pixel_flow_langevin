#!/usr/bin/env python
"""Debug utility for MRIDataset and split-manifest consistency checks."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pixelflow.datasets import MRIDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect MRIDataset train/val/test splits.")
    parser.add_argument(
        "--split-root",
        type=str,
        required=True,
        help="Root folder containing train/val/test subfolders.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated split names to inspect.",
    )
    parser.add_argument("--pt-key", type=str, default="slices", help="Primary key used for .pt dict payloads.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for dataset indexing.")
    parser.add_argument("--num-preview", type=int, default=3, help="Number of example samples to print per split.")
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Manifest path. Default: <split_root>/split_manifest.json",
    )
    return parser.parse_args()


def load_manifest_counts(manifest_path: Path) -> Dict[str, int]:
    data = json.loads(manifest_path.read_text())
    counts = defaultdict(int)
    for row in data:
        split = row.get("split")
        if split is None:
            continue
        counts[str(split)] += int(row.get("num_slices", 1))
    return dict(counts)


def main() -> None:
    args = parse_args()
    split_root = Path(args.split_root).resolve()
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    if not split_root.exists():
        raise FileNotFoundError(f"split_root not found: {split_root}")

    print(f"Split root: {split_root}")
    print(f"Splits: {splits}")
    print(f"pt_key: {args.pt_key}")

    dataset_len_by_split: Dict[str, int] = {}
    for split in splits:
        root = split_root / split
        if not root.exists():
            print(f"\n[{split}] missing root: {root}")
            continue

        dataset = MRIDataset(
            root=str(root),
            pt_key=args.pt_key,
            max_samples=args.max_samples,
            return_metadata=True,
            target_mode="class_index",
            verbose=False,
        )
        dataset_len_by_split[split] = len(dataset)

        print(f"\n[{split}]")
        print(f"  root: {root}")
        print(f"  classes: {len(dataset.classes)}")
        print(f"  class_to_idx: {dataset.class_to_idx}")
        print(f"  indexed samples: {len(dataset)}")

        preview_count = min(args.num_preview, len(dataset))
        for idx in range(preview_count):
            image, target, metadata = dataset[idx]
            print(
                f"  sample[{idx}] shape={tuple(image.shape)} dtype={image.dtype} "
                f"target={target} class={metadata['class_name']} path={metadata['path']} slice={metadata['slice_index']}"
            )

    manifest_path = Path(args.manifest).resolve() if args.manifest else (split_root / "split_manifest.json")
    if manifest_path.exists():
        manifest_counts = load_manifest_counts(manifest_path)
        print(f"\nManifest: {manifest_path}")
        for split in splits:
            expected = manifest_counts.get(split)
            actual = dataset_len_by_split.get(split)
            if expected is None or actual is None:
                continue
            if expected == actual:
                print(f"  [OK] {split}: manifest slices={expected}, dataset samples={actual}")
            else:
                print(f"  [MISMATCH] {split}: manifest slices={expected}, dataset samples={actual}")
    else:
        print(f"\nManifest not found, skip consistency check: {manifest_path}")


if __name__ == "__main__":
    main()
