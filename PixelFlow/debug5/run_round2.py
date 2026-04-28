#!/usr/bin/env python
"""
debug5/run_round2.py — Focused sweep based on Round 1 findings.

Key insight from Round 1: h_x=0.7 and lambda_prox are REDUNDANT mechanisms
for amplifying prior at stage 3. Using both overshoots.

Round 2 hypothesis: lambda_prox at LOWER h_x can provide texture without
the PSNR penalty of large h_x. The proximal term directly injects model
prediction (with texture) into x1 update, while lower h_x keeps the overall
step size small for better data fidelity.

Additionally tests:
  - lambda_prox annealing within Langevin (high→low per step)
  - h_x=0.1 (default) + lambda_prox as the sole prior channel
  - Fine-grained h_x ∈ {0.15, 0.2, 0.25, 0.3, 0.4} with lambda_prox

Saves per-config images with box-region zoom crops for visual analysis.
"""
import sys, os, json, time
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

DEVICE = "cuda:0"
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_r2")
os.makedirs(RESULT_DIR, exist_ok=True)

BASE = dict(terminal_replace_weight=1.0, cg_max_iter=20)

CONFIGS = [
    # ================================================================
    # REF: Reference configs from Round 1 (for visual comparison)
    # ================================================================
    ("REF_base",        dict(terminal_replace_weight=1.0)),
    ("REF_X4",          dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.7])),

    # ================================================================
    # P: lambda_prox as SOLE prior channel (h_x=0.1 default)
    # This tests: can lambda_prox replace h_x entirely?
    # ================================================================
    ("P1_hx01_lp5",     dict(**BASE, lambda_prox=[0, 0, 0, 5])),
    ("P2_hx01_lp10",    dict(**BASE, lambda_prox=[0, 0, 0, 10])),
    ("P3_hx01_lp20",    dict(**BASE, lambda_prox=[0, 0, 0, 20])),
    ("P4_hx01_lp50",    dict(**BASE, lambda_prox=[0, 0, 0, 50])),
    ("P5_hx01_lp100",   dict(**BASE, lambda_prox=[0, 0, 0, 100])),
    ("P6_hx01_lp200",   dict(**BASE, lambda_prox=[0, 0, 0, 200])),

    # ================================================================
    # Q: Fine-grained h_x + lambda_prox combinations
    # Sweet spot search: moderate h_x + moderate lambda_prox
    # ================================================================
    ("Q1_hx015_lp10",   dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.15],
                             lambda_prox=[0, 0, 0, 10])),
    ("Q2_hx02_lp10",    dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.2],
                             lambda_prox=[0, 0, 0, 10])),
    ("Q3_hx02_lp20",    dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.2],
                             lambda_prox=[0, 0, 0, 20])),
    ("Q4_hx025_lp10",   dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.25],
                             lambda_prox=[0, 0, 0, 10])),
    ("Q5_hx025_lp20",   dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.25],
                             lambda_prox=[0, 0, 0, 20])),
    ("Q6_hx03_lp10",    dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.3],
                             lambda_prox=[0, 0, 0, 10])),
    ("Q7_hx03_lp20",    dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.3],
                             lambda_prox=[0, 0, 0, 20])),
    ("Q8_hx03_lp30",    dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.3],
                             lambda_prox=[0, 0, 0, 30])),
    ("Q9_hx04_lp10",    dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.4],
                             lambda_prox=[0, 0, 0, 10])),
    ("Q10_hx04_lp20",   dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.4],
                             lambda_prox=[0, 0, 0, 20])),
    ("Q11_hx05_lp10",   dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.5],
                             lambda_prox=[0, 0, 0, 10])),
    ("Q12_hx05_lp20",   dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.5],
                             lambda_prox=[0, 0, 0, 20])),

    # ================================================================
    # R: Matched lambda sweep at LOWER h_x (safer than Round 1's D group)
    # ================================================================
    ("R1_hx03_m10",     dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.3],
                             lambda_reg=[50, 50, 50, 10],
                             lambda_prox=[0, 0, 0, 10])),
    ("R2_hx03_m20",     dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.3],
                             lambda_reg=[50, 50, 50, 20],
                             lambda_prox=[0, 0, 0, 20])),
    ("R3_hx02_m10",     dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.2],
                             lambda_reg=[50, 50, 50, 10],
                             lambda_prox=[0, 0, 0, 10])),
    ("R4_hx02_m20",     dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.2],
                             lambda_reg=[50, 50, 50, 20],
                             lambda_prox=[0, 0, 0, 20])),
    ("R5_hx04_m20",     dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.4],
                             lambda_reg=[50, 50, 50, 20],
                             lambda_prox=[0, 0, 0, 20])),

    # ================================================================
    # S: More Langevin steps at lower h_x (accumulate texture gradually)
    # ================================================================
    ("S1_hx02_lp20_L20",  dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.2],
                               lambda_prox=[0, 0, 0, 20],
                               num_langevin=[10, 10, 10, 20])),
    ("S2_hx03_lp20_L15",  dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.3],
                               lambda_prox=[0, 0, 0, 20],
                               num_langevin=[10, 10, 10, 15])),
    ("S3_hx03_lp20_L20",  dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.3],
                               lambda_prox=[0, 0, 0, 20],
                               num_langevin=[10, 10, 10, 20])),

    # ================================================================
    # T: Noise injection with lambda_prox anchoring (stochastic + anchor)
    # ================================================================
    ("T1_hx03_lp20_ns03", dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.3],
                               lambda_prox=[0, 0, 0, 20],
                               noise_scale=[0, 0, 0, 0.3])),
    ("T2_hx03_lp20_ns05", dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.3],
                               lambda_prox=[0, 0, 0, 20],
                               noise_scale=[0, 0, 0, 0.5])),

    # ================================================================
    # W: h_epsilon sweep (IP4 finding: h_eps=0.001 generates texture)
    # Smaller h_eps keeps noise eps_k closer to random initial state
    # → residual randomness → texture in reconstruction
    # ================================================================
    ("W1_heps001",          dict(**BASE, h_epsilon=[0.01, 0.01, 0.01, 0.001])),
    ("W2_heps003",          dict(**BASE, h_epsilon=[0.01, 0.01, 0.01, 0.003])),
    ("W3_heps005",          dict(**BASE, h_epsilon=[0.01, 0.01, 0.01, 0.005])),
    ("W4_hx03_heps001",     dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.3],
                                 h_epsilon=[0.01, 0.01, 0.01, 0.001])),
    ("W5_hx05_heps001",     dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.5],
                                 h_epsilon=[0.01, 0.01, 0.01, 0.001])),
    ("W6_hx07_heps001",     dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.7],
                                 h_epsilon=[0.01, 0.01, 0.01, 0.001])),

    # ================================================================
    # X: h_epsilon + lambda_prox combinations (both inject texture)
    # ================================================================
    ("X1_heps001_lp10",     dict(**BASE, h_epsilon=[0.01, 0.01, 0.01, 0.001],
                                 lambda_prox=[0, 0, 0, 10])),
    ("X2_heps001_lp20",     dict(**BASE, h_epsilon=[0.01, 0.01, 0.01, 0.001],
                                 lambda_prox=[0, 0, 0, 20])),
    ("X3_hx03_heps001_lp10", dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.3],
                                   h_epsilon=[0.01, 0.01, 0.01, 0.001],
                                   lambda_prox=[0, 0, 0, 10])),
    ("X4_hx03_heps001_lp20", dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.3],
                                   h_epsilon=[0.01, 0.01, 0.01, 0.001],
                                   lambda_prox=[0, 0, 0, 20])),
    ("X5_hx05_heps001_lp10", dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.5],
                                   h_epsilon=[0.01, 0.01, 0.01, 0.001],
                                   lambda_prox=[0, 0, 0, 10])),
]


def compute_ssim(x, y, window_size=11):
    x01, y01 = (x + 1) / 2, (y + 1) / 2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device=x.device) - window_size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    win = (g.unsqueeze(0) * g.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    C = x.shape[1]
    win = win.expand(C, 1, -1, -1).contiguous()
    pad = window_size // 2
    mu_x = F.conv2d(x01, win, padding=pad, groups=C)
    mu_y = F.conv2d(y01, win, padding=pad, groups=C)
    s_x = F.conv2d(x01 * x01, win, padding=pad, groups=C) - mu_x ** 2
    s_y = F.conv2d(y01 * y01, win, padding=pad, groups=C) - mu_y ** 2
    s_xy = F.conv2d(x01 * y01, win, padding=pad, groups=C) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * s_xy + C2)) / \
               ((mu_x ** 2 + mu_y ** 2 + C1) * (s_x + s_y + C2))
    return ssim_map.mean().item()


def save_per_config_image(name, xf, gt, mask, y, gt_hf, metrics, result_dir):
    """Save individual config image with GT, measurement, result, and box crop."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    img_gt = (gt[0].cpu().permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1).numpy()
    img_y = ((mask[0:1] * gt)[0].cpu().permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1).numpy()
    img_xf = (xf[0].permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1).numpy()

    # Find box region from mask
    inv_mask = (1 - mask[0, 0].cpu()).numpy()
    rows = np.any(inv_mask > 0.5, axis=1)
    cols = np.any(inv_mask > 0.5, axis=0)
    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        pad = 10
        rmin, cmin = max(0, rmin - pad), max(0, cmin - pad)
        rmax, cmax = min(255, rmax + pad), min(255, cmax + pad)
    else:
        rmin, rmax, cmin, cmax = 0, 255, 0, 255

    axes[0].imshow(img_gt)
    axes[0].set_title(f"GT (HF={gt_hf:.3f})", fontsize=8)
    axes[0].axis("off")

    axes[1].imshow(img_y)
    axes[1].set_title("Measurement", fontsize=8)
    axes[1].axis("off")

    axes[2].imshow(img_xf)
    axes[2].set_title(f"{name}\nPSNR={metrics['psnr']:.2f} SSIM={metrics['ssim']:.4f}\n"
                      f"HF={metrics['hf']:.3f} |dHF|={metrics['dhf']:.3f}",
                      fontsize=7)
    axes[2].axis("off")

    # Zoomed crop of box region
    axes[3].imshow(img_xf[rmin:rmax+1, cmin:cmax+1])
    gt_crop = img_gt[rmin:rmax+1, cmin:cmax+1]
    axes[3].set_title(f"Box region zoom\n(unobserved area)", fontsize=7)
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(f"{result_dir}/{name}.png", dpi=150)
    plt.close()


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
    print(f"Round 2 on {DEVICE}  ({len(CONFIGS)} configs)", flush=True)
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
    print(f"BOX INPAINTING  GT HF_unobs={gt_hf:.3f}", flush=True)

    hdr = "{:<3} {:<24} {:>5} {:>8} {:>8} {:>8} {:>6} {:>6} {:>5}"
    print(hdr.format("#", "name", "res", "PSNR", "PSNR_u", "SSIM",
                     "HF", "|dHF|", "t(s)"), flush=True)
    print("-" * 86, flush=True)

    results = {}
    for idx, (name, kw) in enumerate(CONFIGS, 1):
        try:
            xf, po, pa, res, t, hf = run_ip4(
                model, config, gt, y, operator, sigma_n, DEVICE, **kw)

            xf_gpu = xf.to(DEVICE)
            ssim_val = compute_ssim(xf_gpu, gt)
            inv_mask = 1 - mask[0:1]
            psnr_u_mse = ((inv_mask * (xf_gpu - gt)) ** 2).sum() / \
                         (inv_mask.sum() / gt.shape[1]) / gt.shape[1]
            psnr_u = -10 * torch.log10(psnr_u_mse.clamp(min=1e-10)).item()
            xf_gpu = None

            dhf = abs(hf - gt_hf)
            metrics = dict(psnr=pa, psnr_u=psnr_u, ssim=ssim_val,
                           hf=hf, dhf=dhf, res=res, time=t)

            # Save per-config image with box crop
            save_per_config_image(name, xf, gt, mask, y, gt_hf, metrics, RESULT_DIR)

            results[name] = dict(x=xf, psnr_obs=po, **metrics, kw=kw)
            print(hdr.format(idx, name, f"{res:.0f}", f"{pa:.2f}",
                             f"{psnr_u:.2f}", f"{ssim_val:.4f}",
                             f"{hf:.3f}", f"{dhf:.3f}", f"{t:.0f}"),
                  flush=True)
        except Exception as e:
            import traceback
            print(f"{idx:<3} {name:<24} ERROR: {e}", flush=True)
            traceback.print_exc()

    # ---- Summary grid of ALL configs ----
    n = len(results)
    by_psnr = sorted(results.items(), key=lambda kv: kv[1]["psnr"], reverse=True)

    # Grid: 2 rows per config (full image + box crop)
    cols = min(8, n + 2)
    rows = 2 * ((n + 2 + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.0))
    if rows == 1:
        axes = axes[np.newaxis, :]
    axes = axes.reshape(rows, cols)

    img_gt = (gt[0].cpu().permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1).numpy()
    img_y = ((mask[0:1] * gt)[0].cpu().permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1).numpy()

    # Find box region
    inv_m = (1 - mask[0, 0].cpu()).numpy()
    r_rows = np.any(inv_m > 0.5, axis=1)
    r_cols = np.any(inv_m > 0.5, axis=0)
    rmin, rmax = np.where(r_rows)[0][[0, -1]]
    cmin, cmax = np.where(r_cols)[0][[0, -1]]
    pad = 10
    rmin, cmin = max(0, rmin - pad), max(0, cmin - pad)
    rmax, cmax = min(255, rmax + pad), min(255, cmax + pad)

    axes[0, 0].imshow(img_gt)
    axes[0, 0].set_title(f"GT\nHF={gt_hf:.3f}", fontsize=5)
    axes[0, 0].axis("off")
    axes[1, 0].imshow(img_gt[rmin:rmax+1, cmin:cmax+1])
    axes[1, 0].set_title("GT crop", fontsize=5)
    axes[1, 0].axis("off")

    axes[0, 1].imshow(img_y)
    axes[0, 1].set_title("Meas", fontsize=5)
    axes[0, 1].axis("off")
    axes[1, 1].imshow(img_y[rmin:rmax+1, cmin:cmax+1])
    axes[1, 1].set_title("Meas crop", fontsize=5)
    axes[1, 1].axis("off")

    for i, (k, v) in enumerate(by_psnr):
        row_base = 2 * ((i + 2) // cols)
        col = (i + 2) % cols
        if row_base + 1 >= rows:
            break
        img = (v["x"][0].permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1).numpy()
        axes[row_base, col].imshow(img)
        axes[row_base, col].set_title(
            f"{k}\n{v['psnr']:.1f}dB ss={v['ssim']:.3f}",
            fontsize=4)
        axes[row_base, col].axis("off")
        axes[row_base + 1, col].imshow(img[rmin:rmax+1, cmin:cmax+1])
        axes[row_base + 1, col].set_title(
            f"hf={v['hf']:.3f} |d|={v['dhf']:.3f}",
            fontsize=4)
        axes[row_base + 1, col].axis("off")

    for r in range(rows):
        for c in range(cols):
            if not axes[r, c].has_data():
                axes[r, c].axis("off")

    plt.suptitle(f"Round 2: {n} configs (sorted by PSNR) GT HF={gt_hf:.3f}", fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{RESULT_DIR}/round2_grid.png", dpi=200)
    plt.close()

    # ---- Save JSON ----
    summary = {}
    for k, v in results.items():
        summary[k] = {kk: vv for kk, vv in v.items() if kk not in ("x", "kw")}
        summary[k]["kw"] = v["kw"]
    with open(f"{RESULT_DIR}/round2_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # ---- Rankings ----
    print(f"\n{'=' * 86}")
    print(f"GT HF_unobs = {gt_hf:.3f}")
    print(f"\n===== RANKED BY PSNR =====")
    for i, (n, v) in enumerate(by_psnr, 1):
        print(hdr.format(i, n, f"{v['res']:.0f}", f"{v['psnr']:.2f}",
                         f"{v['psnr_u']:.2f}", f"{v['ssim']:.4f}",
                         f"{v['hf']:.3f}", f"{v['dhf']:.3f}",
                         f"{v['time']:.0f}"))

    by_dhf = sorted(results.items(), key=lambda kv: kv[1]["dhf"])
    print(f"\n===== RANKED BY HF proximity =====")
    for i, (n, v) in enumerate(by_dhf, 1):
        print(hdr.format(i, n, f"{v['res']:.0f}", f"{v['psnr']:.2f}",
                         f"{v['psnr_u']:.2f}", f"{v['ssim']:.4f}",
                         f"{v['hf']:.3f}", f"{v['dhf']:.3f}",
                         f"{v['time']:.0f}"))

    print(f"\nSaved {RESULT_DIR}/round2_grid.png")
    print(f"Saved {len(results)} per-config images to {RESULT_DIR}/")
    print(f"DONE.", flush=True)


if __name__ == "__main__":
    main()
