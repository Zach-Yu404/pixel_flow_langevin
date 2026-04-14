#!/usr/bin/env python
"""Split MRI processed_data into train/val/test with optional group-aware leakage prevention."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pixelflow.datasets.mri_dataset import inspect_pt_file


_GROUP_PATTERNS = [
    re.compile(r"(sub(?:ject)?[-_][A-Za-z0-9]+)", re.IGNORECASE),
    re.compile(r"(pat(?:ient)?[-_][A-Za-z0-9]+)", re.IGNORECASE),
    re.compile(r"(study[-_][A-Za-z0-9]+)", re.IGNORECASE),
    re.compile(r"(series[-_][A-Za-z0-9]+)", re.IGNORECASE),
    re.compile(r"(vol(?:ume)?[-_][A-Za-z0-9]+)", re.IGNORECASE),
    re.compile(r"(case[-_][A-Za-z0-9]+)", re.IGNORECASE),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split MRI .pt files into train/val/test.")
    parser.add_argument("--input-root", type=str, required=True, help="Path to processed_data/ root.")
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Output root. Default: <input_root>_split in the same parent directory.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Val split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["copy", "symlink"],
        default="copy",
        help="How to materialize files in output split.",
    )
    parser.add_argument("--pt-key", type=str, default="slices", help="Primary key used to read slices from .pt dict.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files under output root.",
    )
    parser.add_argument(
        "--no-count-slices",
        action="store_true",
        help="Skip loading .pt files for slice counting (faster).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only compute split/manifest and print summary; do not write files.",
    )
    return parser.parse_args()


def stable_int_hash(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def infer_group_id(modality_dir: Path, file_path: Path) -> Optional[str]:
    rel = file_path.relative_to(modality_dir)
    # If there are nested subfolders, use the first parent folder as group id.
    if len(rel.parts) > 1:
        return rel.parts[0]

    stem = file_path.stem
    for pattern in _GROUP_PATTERNS:
        match = pattern.search(stem)
        if match:
            return match.group(1).lower()

    tokens = re.split(r"[-_.]", stem)
    while tokens:
        tail = tokens[-1]
        if re.fullmatch(r"(slice|sl|img|frame|s)\d*", tail, flags=re.IGNORECASE):
            tokens.pop()
            continue
        if re.fullmatch(r"\d{1,4}", tail):
            tokens.pop()
            continue
        break

    if not tokens:
        return None
    candidate = "_".join(tokens).strip("_")
    if not candidate or candidate == stem:
        return None
    return candidate.lower()


def assign_groups_to_splits(
    group_to_files: Dict[str, List[Path]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    modality: str,
) -> Dict[str, List[str]]:
    ratios_sum = train_ratio + val_ratio + test_ratio
    if abs(ratios_sum - 1.0) > 1e-8:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratios_sum}")

    group_ids = sorted(group_to_files.keys())
    rng = random.Random(seed + stable_int_hash(modality))
    rng.shuffle(group_ids)

    total_files = sum(len(group_to_files[g]) for g in group_ids)
    train_target = total_files * train_ratio
    val_target = total_files * val_ratio

    split_groups = {"train": [], "val": [], "test": []}
    count_train = 0
    count_val = 0
    for group_id in group_ids:
        group_size = len(group_to_files[group_id])
        if count_train < train_target:
            split = "train"
            count_train += group_size
        elif count_val < val_target:
            split = "val"
            count_val += group_size
        else:
            split = "test"
        split_groups[split].append(group_id)

    # Keep all splits non-empty when possible.
    if len(group_ids) >= 3:
        for split in ["train", "val", "test"]:
            if split_groups[split]:
                continue
            donor = max(["train", "val", "test"], key=lambda s: len(split_groups[s]))
            if len(split_groups[donor]) > 1:
                split_groups[split].append(split_groups[donor].pop())

    return split_groups


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def materialize(src: Path, dst: Path, mode: str, overwrite: bool) -> None:
    ensure_parent(dst)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            raise FileExistsError(f"Destination exists: {dst}")
        dst.unlink()

    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def summarize_counts(summary: Dict[str, Dict[str, Dict[str, int]]], modalities: Iterable[str]) -> None:
    print("\n=== Split Summary (files / slices) ===")
    print("split\tmodality\tfiles\tslices")
    for split in ["train", "val", "test"]:
        for modality in modalities:
            counts = summary[split].get(modality, {"files": 0, "slices": 0})
            print(f"{split}\t{modality}\t{counts['files']}\t{counts['slices']}")


def main() -> None:
    args = parse_args()

    input_root = Path(args.input_root).resolve()
    if args.output_root is None:
        output_root = input_root.parent / f"{input_root.name}_split"
    else:
        output_root = Path(args.output_root).resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")
    if input_root == output_root:
        raise ValueError("Output root must be different from input root for non-destructive split.")

    modality_dirs = sorted(path for path in input_root.iterdir() if path.is_dir())
    if not modality_dirs:
        raise RuntimeError(f"No modality/class folders found in: {input_root}")

    print(f"Input root: {input_root}")
    print(f"Output root: {output_root}")
    print(f"Split ratios: train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio}")
    print(f"Mode: {args.mode}, Seed: {args.seed}, Count slices: {not args.no_count_slices}, Dry run: {args.dry_run}")

    summary: Dict[str, Dict[str, Dict[str, int]]] = {
        "train": defaultdict(lambda: {"files": 0, "slices": 0}),
        "val": defaultdict(lambda: {"files": 0, "slices": 0}),
        "test": defaultdict(lambda: {"files": 0, "slices": 0}),
    }
    manifest_rows: List[Dict[str, object]] = []

    for modality_dir in modality_dirs:
        modality = modality_dir.name
        pt_files = sorted(path for path in modality_dir.rglob("*.pt") if path.is_file())
        if not pt_files:
            print(f"[WARN] No .pt files under modality {modality}; skipping.")
            continue

        inferred_groups = {path: infer_group_id(modality_dir, path) for path in pt_files}
        has_group_signal = any(group_id is not None for group_id in inferred_groups.values())
        if not has_group_signal:
            print(
                f"[WARN] No group signal inferred for modality={modality}. "
                "Fallback to file-level split; potential leakage across related slices/volumes."
            )

        group_to_files: Dict[str, List[Path]] = defaultdict(list)
        for path in pt_files:
            group_id = inferred_groups[path]
            if group_id is None:
                group_id = str(path.relative_to(modality_dir))
            group_to_files[group_id].append(path)

        split_groups = assign_groups_to_splits(
            group_to_files=group_to_files,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            modality=modality,
        )

        group_to_split = {}
        for split_name, group_ids in split_groups.items():
            for group_id in group_ids:
                group_to_split[group_id] = split_name

        for group_id, files in group_to_files.items():
            split_name = group_to_split[group_id]
            for src_path in files:
                rel_inside_modality = src_path.relative_to(modality_dir)
                dst_path = output_root / split_name / modality / rel_inside_modality

                if not args.dry_run:
                    materialize(src_path, dst_path, mode=args.mode, overwrite=args.overwrite)

                num_slices = 1
                if not args.no_count_slices:
                    num_slices = int(inspect_pt_file(src_path, pt_key=args.pt_key).num_slices)

                summary[split_name][modality]["files"] += 1
                summary[split_name][modality]["slices"] += num_slices
                manifest_rows.append(
                    {
                        "split": split_name,
                        "modality": modality,
                        "group_id": group_id,
                        "original_path": str(src_path.resolve()),
                        "new_path": str(dst_path.resolve() if not args.dry_run else dst_path),
                        "num_slices": num_slices,
                    }
                )

    modalities = [p.name for p in modality_dirs]
    summarize_counts(summary, modalities=modalities)

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_json_path = output_root / "split_manifest.json"
    manifest_csv_path = output_root / "split_manifest.csv"
    summary_json_path = output_root / "split_summary.json"

    with manifest_json_path.open("w") as f:
        json.dump(manifest_rows, f, indent=2)

    with manifest_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "modality", "group_id", "original_path", "new_path", "num_slices"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary_serializable = {
        split: {modality: dict(counts) for modality, counts in split_counts.items()}
        for split, split_counts in summary.items()
    }
    with summary_json_path.open("w") as f:
        json.dump(summary_serializable, f, indent=2)

    print(f"\nManifest JSON: {manifest_json_path}")
    print(f"Manifest CSV:  {manifest_csv_path}")
    print(f"Summary JSON:  {summary_json_path}")
    print(f"Total files indexed: {len(manifest_rows)}")


if __name__ == "__main__":
    main()
