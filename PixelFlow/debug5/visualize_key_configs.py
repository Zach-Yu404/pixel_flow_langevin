#!/usr/bin/env python
"""
Generate high-quality image comparisons for key configs.
Runs on cuda:1 so it doesn't interfere with the sweep on cuda:0.

Saves per-config images with:
  - Full GT, measurement, result
  - Zoomed box-region crop (GT vs result side-by-side)
  - Summary grid with all configs
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "debug_IP4"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from inpaintingStart import get_operator
from ms_sampler_v5 import run_ip4, hf_energy
from pixelflow.utils import config as config_utils
from pixelflow.utils.misc import seed_everything

DEVICE = "cuda:1"
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_vis")
os.makedirs(RESULT_DIR, exist_ok=True)

BASE = dict(terminal_replace_weight=1.0, cg_max_iter=20)

# Key configs to visualize (based on sweep findings)
CONFIGS = [
    ("baseline",        dict(terminal_replace_weight=1.0)),
    ("X4_hx07",         dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.7])),
    ("hx03_lp20",       dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.3],
                             lambda_prox=[0, 0, 0, 20])),
    ("hx05_lp20",       dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.5],
                             lambda_prox=[0, 0, 0, 20])),
    ("hx05_lp50",       dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.5],
                             lambda_prox=[0, 0, 0, 50])),
    ("hx07_lp20",       dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.7],
                             lambda_prox=[0, 0, 0, 20])),
    # h_epsilon experiments
    ("heps001",         dict(**BASE, h_epsilon=[0.01, 0.01, 0.01, 0.001])),
    ("hx03_heps001",    dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.3],
                             h_epsilon=[0.01, 0.01, 0.01, 0.001])),
    ("hx05_heps001",    dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.5],
                             h_epsilon=[0.01, 0.01, 0.01, 0.001])),
    # Combined: h_x + h_eps + lambda_prox (three channels)
    ("hx03_heps001_lp10", dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.3],
                                h_epsilon=[0.01, 0.01, 0.01, 0.001],
                                lambda_prox=[0, 0, 0, 10])),
    ("hx03_heps001_lp20", dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.3],
                                h_epsilon=[0.01, 0.01, 0.01, 0.001],
                                lambda_prox=[0, 0, 0, 20])),
    ("hx05_heps001_lp10", dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.5],
                                h_epsilon=[0.01, 0.01, 0.01, 0.001],
                                lambda_prox=[0, 0, 0, 10])),
]


def to_img(t):
    """Tensor (C,H,W) in [-1,1] → numpy (H,W,C) in [0,1]."""
    return (t.cpu().permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1).numpy()


def find_box(mask):
    """Find box region bounds from mask (1=observed, 0=unobserved)."""
    inv = (1 - mask[0, 0].cpu()).numpy()
    rows = np.any(inv > 0.5, axis=1)
    cols = np.any(inv > 0.5, axis=0)
    if not rows.any():
        return 0, 255, 0, 255
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    pad = 5
    return max(0, rmin - pad), min(255, rmax + pad), max(0, cmin - pad), min(255, cmax + pad)


def load_gt(device):
    pt = "trajectory_videos/posterior_sampling/baseline_05/baseline_05_mode-f_x1.pt"
    if os.path.exists(pt):
        d = torch.load(pt, map_location="cpu", weights_only=False)
        return d["gt"][:2].to(device)
    from torchvision import transforms, datasets
    from ms_posterior_sampling_article_version_final_utils import center_crop_arr
    data_dir = "/data/Zach_dataset/imageNet256/ILSVRC/Data/CLS-LOC/train/"
    tf = transforms.Compose([
        transforms.Lambda(lambda p: center_crop_arr(p, 256)),
        transforms.ToTensor(),
        transforms.Normalize([.5]*3, [.5]*3, inplace=True),
    ])
    ds = datasets.ImageFolder(data_dir, tf)
    torch.manual_seed(20000120)
    ci = [i for i, (_, y) in enumerate(ds.samples) if int(y) == 10]
    perm = torch.randperm(len(ci))[:2]
    return torch.stack([ds[ci[i]][0] for i in perm.tolist()]).to(device)


def main():
    print(f"Visualization on {DEVICE}  ({len(CONFIGS)} configs)", flush=True)
    print("Loading model...", flush=True)

    gt = load_gt(DEVICE)
    config = OmegaConf.load("pretrained_models/c2img/config.yaml")
    model = config_utils.instantiate_from_config(config.model).to(DEVICE)
    ckpt = torch.load("pretrained_models/c2img/model.pt",
                      map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt, strict=True)
    model.eval()
    print("Model loaded.\n", flush=True)

    sigma_n = 0.05
    operator = get_operator(
        "inpainting", resolution=256, device=DEVICE, sigma=sigma_n,
        mask_type="box", mask_len_range=(80, 160), mask_prob_range=None,
    )
    y = operator(gt).detach()
    mask = operator.get_mask(x=gt).float().to(DEVICE)
    gt_hf = hf_energy(gt * (1 - mask[0:1]))
    rmin, rmax, cmin, cmax = find_box(mask)
    print(f"BOX region: [{rmin}:{rmax}, {cmin}:{cmax}]  GT HF={gt_hf:.3f}\n", flush=True)

    results = {}
    for idx, (name, kw) in enumerate(CONFIGS, 1):
        print(f"  {idx}/{len(CONFIGS)} {name}...", end=" ", flush=True)
        xf, po, pa, res, t, hf = run_ip4(
            model, config, gt, y, operator, sigma_n, DEVICE, **kw)
        dhf = abs(hf - gt_hf)
        results[name] = dict(x=xf, psnr=pa, hf=hf, dhf=dhf, res=res, time=t)
        print(f"PSNR={pa:.2f}  HF={hf:.3f}  |dHF|={dhf:.3f}  t={t:.0f}s", flush=True)

    # ================================================================
    # Image 1: Per-config comparison (GT crop vs result crop, both images)
    # ================================================================
    for name, v in results.items():
        fig, axes = plt.subplots(2, 4, figsize=(16, 8.5))

        for img_idx in range(2):
            gt_img = to_img(gt[img_idx])
            y_img = to_img((mask[0:1] * gt)[img_idx])
            xf_img = to_img(v["x"][img_idx])

            axes[img_idx, 0].imshow(gt_img)
            axes[img_idx, 0].set_title(f"GT (img {img_idx})", fontsize=8)
            axes[img_idx, 0].axis("off")

            axes[img_idx, 1].imshow(y_img)
            axes[img_idx, 1].set_title("Measurement", fontsize=8)
            axes[img_idx, 1].axis("off")

            axes[img_idx, 2].imshow(xf_img)
            axes[img_idx, 2].set_title(f"Result: {name}", fontsize=8)
            axes[img_idx, 2].axis("off")

            # Side-by-side crop: GT vs Result
            crop_gt = gt_img[rmin:rmax+1, cmin:cmax+1]
            crop_xf = xf_img[rmin:rmax+1, cmin:cmax+1]
            combined = np.concatenate([crop_gt, np.ones((crop_gt.shape[0], 3, 3)), crop_xf], axis=1)
            axes[img_idx, 3].imshow(combined)
            axes[img_idx, 3].set_title("Box crop: GT | Result", fontsize=8)
            axes[img_idx, 3].axis("off")

        plt.suptitle(f"{name}  —  PSNR={v['psnr']:.2f}  HF={v['hf']:.3f}  "
                     f"|dHF|={v['dhf']:.3f}  res={v['res']:.0f}", fontsize=10)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f"{RESULT_DIR}/{name}.png", dpi=150)
        plt.close()

    # ================================================================
    # Image 2: Summary grid — all configs, image 0, full + crop
    # ================================================================
    n = len(results)
    fig, axes = plt.subplots(3, n + 1, figsize=((n + 1) * 2.5, 7.5))

    # Column 0: GT
    gt_img = to_img(gt[0])
    y_img = to_img((mask[0:1] * gt)[0])
    axes[0, 0].imshow(gt_img)
    axes[0, 0].set_title(f"GT\nHF={gt_hf:.3f}", fontsize=6)
    axes[0, 0].axis("off")
    axes[1, 0].imshow(gt_img[rmin:rmax+1, cmin:cmax+1])
    axes[1, 0].set_title("GT box crop", fontsize=6)
    axes[1, 0].axis("off")
    axes[2, 0].imshow(y_img)
    axes[2, 0].set_title("Measurement", fontsize=6)
    axes[2, 0].axis("off")

    for i, (name, v) in enumerate(results.items()):
        xf_img = to_img(v["x"][0])
        axes[0, i + 1].imshow(xf_img)
        axes[0, i + 1].set_title(f"{name}\nPSNR={v['psnr']:.1f}", fontsize=5)
        axes[0, i + 1].axis("off")

        axes[1, i + 1].imshow(xf_img[rmin:rmax+1, cmin:cmax+1])
        axes[1, i + 1].set_title(f"HF={v['hf']:.3f}\n|d|={v['dhf']:.3f}", fontsize=5)
        axes[1, i + 1].axis("off")

        # Difference map (amplified)
        diff = np.abs(xf_img - gt_img).mean(axis=2)
        axes[2, i + 1].imshow(diff, cmap="hot", vmin=0, vmax=0.3)
        axes[2, i + 1].set_title(f"Error map\nres={v['res']:.0f}", fontsize=5)
        axes[2, i + 1].axis("off")

    plt.suptitle(f"debug5 key configs comparison  (GT HF={gt_hf:.3f})", fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{RESULT_DIR}/comparison_grid.png", dpi=200)
    plt.close()

    # ================================================================
    # Image 3: Box-region-only grid (zoomed, larger)
    # ================================================================
    fig, axes = plt.subplots(2, n + 1, figsize=((n + 1) * 2.8, 5.6))

    gt_crop = to_img(gt[0])[rmin:rmax+1, cmin:cmax+1]
    axes[0, 0].imshow(gt_crop)
    axes[0, 0].set_title(f"GT\nHF={gt_hf:.3f}", fontsize=7)
    axes[0, 0].axis("off")
    gt_crop2 = to_img(gt[1])[rmin:rmax+1, cmin:cmax+1]
    axes[1, 0].imshow(gt_crop2)
    axes[1, 0].set_title("GT img2", fontsize=7)
    axes[1, 0].axis("off")

    for i, (name, v) in enumerate(results.items()):
        crop0 = to_img(v["x"][0])[rmin:rmax+1, cmin:cmax+1]
        crop1 = to_img(v["x"][1])[rmin:rmax+1, cmin:cmax+1]
        axes[0, i + 1].imshow(crop0)
        axes[0, i + 1].set_title(f"{name}\nP={v['psnr']:.1f} HF={v['hf']:.3f}", fontsize=5)
        axes[0, i + 1].axis("off")
        axes[1, i + 1].imshow(crop1)
        axes[1, i + 1].set_title(f"|d|={v['dhf']:.3f}", fontsize=5)
        axes[1, i + 1].axis("off")

    plt.suptitle(f"Box region comparison (unobserved area)  GT HF={gt_hf:.3f}", fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{RESULT_DIR}/box_crops_grid.png", dpi=200)
    plt.close()

    print(f"\nSaved {len(results)} per-config images to {RESULT_DIR}/")
    print(f"Saved {RESULT_DIR}/comparison_grid.png")
    print(f"Saved {RESULT_DIR}/box_crops_grid.png")
    print("DONE.", flush=True)


if __name__ == "__main__":
    main()
