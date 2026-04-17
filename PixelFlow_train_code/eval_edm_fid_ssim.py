"""Evaluate FID+SSIM for EDM-generated samples against val set (per-class + overall)."""
import argparse
import os
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from cleanfid import fid


# ---- SSIM ----
def _gaussian_kernel(size, sigma):
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g.unsqueeze(0) * g.unsqueeze(1)).unsqueeze(0).unsqueeze(0)


def compute_ssim(img1, img2, window_size=11, C1=0.01**2, C2=0.03**2):
    ch = img1.shape[1]
    kernel = _gaussian_kernel(window_size, 1.5).to(img1.device, img1.dtype)
    kernel = kernel.expand(ch, 1, window_size, window_size)
    mu1 = F.conv2d(img1, kernel, padding=window_size // 2, groups=ch)
    mu2 = F.conv2d(img2, kernel, padding=window_size // 2, groups=ch)
    sigma1_sq = F.conv2d(img1**2, kernel, padding=window_size // 2, groups=ch) - mu1**2
    sigma2_sq = F.conv2d(img2**2, kernel, padding=window_size // 2, groups=ch) - mu2**2
    sigma12 = F.conv2d(img1 * img2, kernel, padding=window_size // 2, groups=ch) - mu1 * mu2
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean(dim=[1, 2, 3])


def load_gray(path):
    return torch.from_numpy(np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0).unsqueeze(0).unsqueeze(0)


# ---- EDM class -> val class mapping ----
EDM_TO_VAL = {
    "AXFLAIR": "AXFLAIR_normalized",
    "AXT1": "AXT1_normalized",
    "AXT1POST": "AXT1POST_normalized",
    "AXT2": "AXT2_normalized",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--edm_root", default="/data/Sahil_dataset/MRI_processed/edm_gen/Full_fastmri_score_model")
    p.add_argument("--real_dir", default="/data/Sahil_dataset/MRI_processed/_fid_stats_cache/real_pngs_2f0174ce15_res384")
    p.add_argument("--n_per_class", type=int, default=100)
    p.add_argument("--n_axt1", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_log", default="exp0002/fid_edm_baseline.log")
    args = p.parse_args()

    random.seed(args.seed)
    edm_root = Path(args.edm_root)
    real_dir = Path(args.real_dir)

    # Build tmp dirs with sampled EDM images (converted to RGB for FID)
    tmp = Path("_edm_fid_tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    gen_all = tmp / "gen_all"
    gen_all.mkdir(parents=True)

    # Index real PNGs by class idx
    real_by_cls = {}
    for f in real_dir.glob("real_*.png"):
        cls_idx = int(f.stem.split("_")[1])
        real_by_cls.setdefault(cls_idx, []).append(f)

    # Val class order (sorted, matching MRIDataset)
    val_classes = sorted(EDM_TO_VAL.values())
    val_cls_to_idx = {name: i for i, name in enumerate(val_classes)}

    results = []
    overall_ssim = []
    total_sampled = 0

    for edm_cls, val_cls in sorted(EDM_TO_VAL.items()):
        cls_idx = val_cls_to_idx[val_cls]
        n = args.n_axt1 if edm_cls == "AXT1" else args.n_per_class
        edm_dir = edm_root / edm_cls
        all_files = sorted(edm_dir.glob("*.png"))
        sampled = random.sample(all_files, min(n, len(all_files)))

        # Copy to gen_all + per-class tmp dir
        gen_cls = tmp / f"gen_class_{cls_idx}"
        gen_cls.mkdir(parents=True)
        real_cls = tmp / f"real_class_{cls_idx}"
        real_cls.mkdir(parents=True)

        for i, f in enumerate(sampled):
            img = Image.open(f).convert("L").convert("RGB")
            img.save(gen_all / f"gen_{cls_idx}_{total_sampled + i:06d}.png")
            img.save(gen_cls / f"gen_{i:06d}.png")
        total_sampled += len(sampled)

        # Symlink real PNGs for this class
        if cls_idx in real_by_cls:
            for rf in real_by_cls[cls_idx]:
                dst = real_cls / rf.name
                if not dst.exists():
                    os.symlink(rf.resolve(), dst)

        # Per-class FID
        try:
            cls_fid = fid.compute_fid(str(gen_cls), str(real_cls), mode="clean", num_workers=4)
        except Exception as e:
            cls_fid = None
            print(f"  FID failed for {edm_cls}: {e}")

        # Per-class SSIM
        cls_ssim = []
        real_files = sorted(real_cls.glob("*.png"))
        for i, gf in enumerate(sorted(gen_cls.glob("*.png"))):
            if not real_files:
                break
            rf = real_files[i % len(real_files)]
            val = compute_ssim(load_gray(gf), load_gray(rf)).item()
            cls_ssim.append(val)
            overall_ssim.append(val)

        fid_str = f"{cls_fid:.4f}" if cls_fid is not None else "N/A"
        ssim_str = f"{np.mean(cls_ssim):.4f}" if cls_ssim else "N/A"
        print(f"  {edm_cls} (class {cls_idx}): n={len(sampled)}, FID={fid_str}, SSIM={ssim_str}")
        results.append((edm_cls, cls_idx, len(sampled), cls_fid, np.mean(cls_ssim) if cls_ssim else None))

    # Overall FID using cached val stats
    overall_fid = None
    try:
        overall_fid = fid.compute_fid(str(gen_all), dataset_name="pixelflow_val_2f0174ce15_res384",
                                       dataset_res=384, mode="clean", dataset_split="custom", num_workers=4)
    except Exception as e:
        print(f"  Overall FID failed: {e}")

    # Summary
    summary = "\n=== EDM Baseline (edm_gen/Full_fastmri_score_model) ===\n"
    fid_str = f"{overall_fid:.4f}" if overall_fid is not None else "N/A"
    ssim_str = f"{np.mean(overall_ssim):.4f}" if overall_ssim else "N/A"
    summary += f"  Overall: FID={fid_str}, SSIM={ssim_str} (n={total_sampled})\n"
    for edm_cls, cls_idx, n, cfid, cssim in results:
        f_s = f"{cfid:.4f}" if cfid is not None else "N/A"
        s_s = f"{cssim:.4f}" if cssim is not None else "N/A"
        summary += f"  {edm_cls} (class {cls_idx}): n={n}, FID={f_s}, SSIM={s_s}\n"

    print(summary)
    os.makedirs(os.path.dirname(args.out_log), exist_ok=True)
    with open(args.out_log, "w") as f:
        f.write(summary)
    print(f"Saved to {args.out_log}")

    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
