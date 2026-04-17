"""Compute per-class FID from already-generated sample dirs (reuses existing PNGs)."""
import argparse
import os
import shutil
from pathlib import Path

from cleanfid import fid


def split_by_class(src_dir, tmp_root, prefix="gen"):
    """Split PNGs like gen_0_000001.png into per-class subdirs."""
    classes = {}
    for f in sorted(Path(src_dir).glob(f"{prefix}_*.png")):
        cls_idx = int(f.stem.split("_")[1])
        classes.setdefault(cls_idx, []).append(f)

    dirs = {}
    for cls_idx, files in classes.items():
        d = Path(tmp_root) / f"class_{cls_idx}"
        d.mkdir(parents=True, exist_ok=True)
        for f in files:
            dst = d / f.name
            if not dst.exists():
                os.symlink(f.resolve(), dst)
        dirs[cls_idx] = str(d)
    return dirs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gen_dir", required=True, help="dir with gen_*.png files")
    p.add_argument("--real_dir", required=True, help="dir with real_*.png files")
    p.add_argument("--log_file", default=None, help="append results to this file")
    p.add_argument("--label", default="", help="checkpoint label for output")
    args = p.parse_args()

    tmp = Path(args.gen_dir).parent / "_perclass_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)

    gen_by_class = split_by_class(args.gen_dir, tmp / "gen", prefix="gen")
    real_by_class = split_by_class(args.real_dir, tmp / "real", prefix="real")

    results = []
    for cls_idx in sorted(gen_by_class.keys()):
        if cls_idx not in real_by_class:
            continue
        n_gen = len(list(Path(gen_by_class[cls_idx]).glob("*.png")))
        n_real = len(list(Path(real_by_class[cls_idx]).glob("*.png")))
        try:
            score = fid.compute_fid(gen_by_class[cls_idx], real_by_class[cls_idx],
                                    mode="clean", num_workers=4)
            results.append((cls_idx, score, n_gen, n_real))
            print(f"  Class {cls_idx}: FID={score:.4f} (gen={n_gen}, real={n_real})")
        except Exception as e:
            print(f"  Class {cls_idx}: FAILED ({e})")
            results.append((cls_idx, None, n_gen, n_real))

    # Clean up symlink dirs
    shutil.rmtree(tmp, ignore_errors=True)

    summary = f"\n=== Per-Class FID {args.label} ===\n"
    for cls_idx, score, ng, nr in results:
        s = f"{score:.4f}" if score is not None else "N/A"
        summary += f"  Class {cls_idx}: FID={s} (gen={ng}, real={nr})\n"
    print(summary)

    if args.log_file:
        with open(args.log_file, "a") as f:
            f.write(summary)
        print(f"Appended to {args.log_file}")


if __name__ == "__main__":
    main()
