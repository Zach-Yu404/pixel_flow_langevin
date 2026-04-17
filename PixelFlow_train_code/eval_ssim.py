"""Compute per-class and overall SSIM from already-generated sample dirs."""
import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def compute_ssim_batch(img1, img2, window_size=11, C1=0.01**2, C2=0.03**2):
    channel = img1.shape[1]
    kernel = _gaussian_kernel(window_size, 1.5).to(img1.device, img1.dtype)
    kernel = kernel.expand(channel, 1, window_size, window_size)
    mu1 = F.conv2d(img1, kernel, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, kernel, padding=window_size // 2, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = F.conv2d(img1 ** 2, kernel, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 ** 2, kernel, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, padding=window_size // 2, groups=channel) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean(dim=[1, 2, 3])


def _gaussian_kernel(size, sigma):
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g.unsqueeze(0) * g.unsqueeze(1)).unsqueeze(0).unsqueeze(0)


def load_png_gray(path):
    img = np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gen_dir", required=True)
    p.add_argument("--real_dir", required=True)
    p.add_argument("--log_file", default=None)
    p.add_argument("--label", default="")
    args = p.parse_args()

    # Index real images by class
    real_by_class = {}
    for f in sorted(Path(args.real_dir).glob("real_*.png")):
        cls_idx = int(f.stem.split("_")[1])
        real_by_class.setdefault(cls_idx, []).append(f)

    # Compute SSIM per generated image against a random real of same class
    ssim_by_class = {}
    ssim_all = []

    gen_files = sorted(Path(args.gen_dir).glob("gen_*.png"))
    for f in gen_files:
        cls_idx = int(f.stem.split("_")[1])
        if cls_idx not in real_by_class or len(real_by_class[cls_idx]) == 0:
            continue
        gen_img = load_png_gray(f)
        # Cycle through real images
        idx = len(ssim_by_class.get(cls_idx, [])) % len(real_by_class[cls_idx])
        real_img = load_png_gray(real_by_class[cls_idx][idx])
        if real_img.shape != gen_img.shape:
            real_img = F.interpolate(real_img, size=gen_img.shape[-2:], mode='bilinear', align_corners=False)
        val = compute_ssim_batch(gen_img, real_img).item()
        ssim_by_class.setdefault(cls_idx, []).append(val)
        ssim_all.append(val)

    summary = f"\n=== SSIM {args.label} ===\n"
    summary += f"  Overall: SSIM={np.mean(ssim_all):.4f} (n={len(ssim_all)})\n"
    for cls_idx in sorted(ssim_by_class.keys()):
        vals = ssim_by_class[cls_idx]
        summary += f"  Class {cls_idx}: SSIM={np.mean(vals):.4f} (n={len(vals)})\n"

    print(summary)
    if args.log_file:
        with open(args.log_file, "a") as f:
            f.write(summary)
        print(f"Appended to {args.log_file}")


if __name__ == "__main__":
    main()
