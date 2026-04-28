#!/usr/bin/env python
"""
debug5/run_sweep.py — Comprehensive PRINCIPLE parameter sweep with proximal term.

The updated pseudocode adds λ(sg(x̂₁) - x₁) to the posterior gradient g_x1.
This proximal term anchors x₁ toward the model's denoised prediction at each
Langevin step.  On unobserved pixels the CG-preconditioned update becomes:

    δ_unobs ≈ h_x/2 · (x̂₁ - x₁) + (h_x / 2λ) · H^T s_flow

so the model prediction passes through WITHOUT λ-damping (unlike the score term).

Tests ~55 configurations combining:
  - lambda_prox (the new pseudocode term)
  - h_x at stage 3 (dominant IP4 lever)
  - lambda_reg at stage 3
  - noise_scale, num_langevin, cg_max_iter
  - Matched λ (lambda_prox = lambda_reg, as in pseudocode)
  - All key interactions

Uses IP4 run_ip4() infrastructure; terminal_replace=1.0 always on.
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
from ms_posterior_sampling_article_version_final_utils import make_Ak_fns
from ms_sampler_v5 import run_ip4, hf_energy
from pixelflow.utils import config as config_utils
from pixelflow.utils.misc import seed_everything

DEVICE = "cuda:0"
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Extra metrics
# ---------------------------------------------------------------------------

def compute_ssim(x, y, window_size=11):
    """SSIM between x, y in [-1,1], shape (B,C,H,W). Returns scalar."""
    x01 = (x + 1) / 2
    y01 = (y + 1) / 2
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


def psnr_region(x, gt, mask):
    """PSNR in masked region (mask=1 where we compute)."""
    n = mask.sum() / x.shape[1]
    if n < 1:
        return float('inf')
    mse = ((mask * (x - gt)) ** 2).sum() / (n * x.shape[1])
    if mse < 1e-10:
        return float('inf')
    return -10 * torch.log10(mse).item()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Configuration definitions
# ---------------------------------------------------------------------------

# Common base: terminal_replace=1, CG=20 (from IP3/IP4 findings)
BASE = dict(terminal_replace_weight=1.0, cg_max_iter=20)

CONFIGS = [
    # ================================================================
    # A: Baselines / IP4 winners (reference points)
    # ================================================================
    ("A0_raw",          dict()),
    ("A1_F1",           dict(terminal_replace_weight=1.0)),
    ("A2_X4",           dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.7])),
    ("A3_CB1",          dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.3],
                             lambda_reg=[50, 50, 50, 20])),

    # ================================================================
    # B: lambda_prox uniform sweep (all stages, on X4 base)
    # ================================================================
    ("B1_lp1",          dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.7], lambda_prox=1)),
    ("B2_lp5",          dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.7], lambda_prox=5)),
    ("B3_lp10",         dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.7], lambda_prox=10)),
    ("B4_lp20",         dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.7], lambda_prox=20)),
    ("B5_lp50",         dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.7], lambda_prox=50)),
    ("B6_lp100",        dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.7], lambda_prox=100)),

    # ================================================================
    # C: lambda_prox stage-3 only (preserve stage 0-2 behavior)
    # ================================================================
    ("C1_s3lp5",        dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.7],
                             lambda_prox=[0, 0, 0, 5])),
    ("C2_s3lp10",       dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.7],
                             lambda_prox=[0, 0, 0, 10])),
    ("C3_s3lp20",       dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.7],
                             lambda_prox=[0, 0, 0, 20])),
    ("C4_s3lp50",       dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.7],
                             lambda_prox=[0, 0, 0, 50])),
    ("C5_s3lp100",      dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.7],
                             lambda_prox=[0, 0, 0, 100])),

    # ================================================================
    # D: Matched lambda (pseudocode canonical: lambda_prox = lambda_reg)
    #    at stage 3, sweep joint value
    # ================================================================
    ("D1_matched5",     dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.7],
                             lambda_reg=[50, 50, 50, 5],
                             lambda_prox=[0, 0, 0, 5])),
    ("D2_matched10",    dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.7],
                             lambda_reg=[50, 50, 50, 10],
                             lambda_prox=[0, 0, 0, 10])),
    ("D3_matched20",    dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.7],
                             lambda_reg=[50, 50, 50, 20],
                             lambda_prox=[0, 0, 0, 20])),
    ("D4_matched50",    dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.7],
                             lambda_reg=[50, 50, 50, 50],
                             lambda_prox=[0, 0, 0, 50])),

    # ================================================================
    # E: Decoupled lambda_prox > lambda_reg (stronger anchor)
    # ================================================================
    ("E1_lr20_lp50",    dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.7],
                             lambda_reg=[50, 50, 50, 20],
                             lambda_prox=[0, 0, 0, 50])),
    ("E2_lr10_lp50",    dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.7],
                             lambda_reg=[50, 50, 50, 10],
                             lambda_prox=[0, 0, 0, 50])),
    ("E3_lr5_lp50",     dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.7],
                             lambda_reg=[50, 50, 50, 5],
                             lambda_prox=[0, 0, 0, 50])),
    ("E4_lr10_lp100",   dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.7],
                             lambda_reg=[50, 50, 50, 10],
                             lambda_prox=[0, 0, 0, 100])),
    ("E5_lr20_lp100",   dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.7],
                             lambda_reg=[50, 50, 50, 20],
                             lambda_prox=[0, 0, 0, 100])),

    # ================================================================
    # F: h_x × lambda_prox interaction grid
    # ================================================================
    ("F1_hx03_lp20",    dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.3],
                             lambda_prox=[0, 0, 0, 20])),
    ("F2_hx05_lp20",    dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.5],
                             lambda_prox=[0, 0, 0, 20])),
    ("F3_hx10_lp20",    dict(**BASE, h_x=[0.1, 0.1, 0.1, 1.0],
                             lambda_prox=[0, 0, 0, 20])),
    ("F4_hx05_lp50",    dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.5],
                             lambda_prox=[0, 0, 0, 50])),
    ("F5_hx10_lp50",    dict(**BASE, h_x=[0.1, 0.1, 0.1, 1.0],
                             lambda_prox=[0, 0, 0, 50])),
    ("F6_hx15_lp50",    dict(**BASE, h_x=[0.1, 0.1, 0.1, 1.5],
                             lambda_prox=[0, 0, 0, 50])),
    ("F7_hx20_lp50",    dict(**BASE, h_x=[0.1, 0.1, 0.1, 2.0],
                             lambda_prox=[0, 0, 0, 50])),
    ("F8_hx10_lp100",   dict(**BASE, h_x=[0.1, 0.1, 0.1, 1.0],
                             lambda_prox=[0, 0, 0, 100])),

    # ================================================================
    # G: noise_scale + lambda_prox (stochastic exploration + anchoring)
    # ================================================================
    ("G1_ns01_lp20",    dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.7],
                             noise_scale=[0, 0, 0, 0.1],
                             lambda_prox=[0, 0, 0, 20])),
    ("G2_ns03_lp20",    dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.7],
                             noise_scale=[0, 0, 0, 0.3],
                             lambda_prox=[0, 0, 0, 20])),
    ("G3_ns05_lp50",    dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.7],
                             noise_scale=[0, 0, 0, 0.5],
                             lambda_prox=[0, 0, 0, 50])),
    ("G4_ns10_lp50",    dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.7],
                             noise_scale=[0, 0, 0, 1.0],
                             lambda_prox=[0, 0, 0, 50])),

    # ================================================================
    # H: num_langevin × lambda_prox
    # ================================================================
    ("H1_L5_lp20",      dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.7],
                             num_langevin=[10, 10, 10, 5],
                             lambda_prox=[0, 0, 0, 20])),
    ("H2_L15_lp20",     dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.7],
                             num_langevin=[10, 10, 10, 15],
                             lambda_prox=[0, 0, 0, 20])),
    ("H3_L20_lp20",     dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.7],
                             num_langevin=[10, 10, 10, 20],
                             lambda_prox=[0, 0, 0, 20])),
    ("H4_L20_lp50",     dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.7],
                             num_langevin=[10, 10, 10, 20],
                             lambda_prox=[0, 0, 0, 50])),
    ("H5_L30_lp50",     dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.7],
                             num_langevin=[10, 10, 10, 30],
                             lambda_prox=[0, 0, 0, 50])),

    # ================================================================
    # I: CG iterations × lambda_prox
    # ================================================================
    ("I1_CG5_lp20",     dict(terminal_replace_weight=1.0,
                             h_x=[0.1, 0.1, 0.1, 0.7], cg_max_iter=5,
                             lambda_prox=[0, 0, 0, 20])),
    ("I2_CG10_lp20",    dict(terminal_replace_weight=1.0,
                             h_x=[0.1, 0.1, 0.1, 0.7], cg_max_iter=10,
                             lambda_prox=[0, 0, 0, 20])),
    ("I3_CG50_lp20",    dict(terminal_replace_weight=1.0,
                             h_x=[0.1, 0.1, 0.1, 0.7], cg_max_iter=50,
                             lambda_prox=[0, 0, 0, 20])),

    # ================================================================
    # J: IP4 winners (CB1, HXL) + lambda_prox
    # ================================================================
    ("J1_CB1_lp20",     dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.3],
                             lambda_reg=[50, 50, 50, 20],
                             lambda_prox=[0, 0, 0, 20])),
    ("J2_CB1_lp50",     dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.3],
                             lambda_reg=[50, 50, 50, 20],
                             lambda_prox=[0, 0, 0, 50])),
    ("J3_HXL_lp20",     dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.5],
                             num_langevin=[10, 10, 10, 15],
                             lambda_prox=[0, 0, 0, 20])),

    # ================================================================
    # K: Aggressive combos (push HF, PSNR may drop)
    # ================================================================
    ("K1_ultra",        dict(**BASE, h_x=[0.1, 0.1, 0.1, 1.0],
                             lambda_reg=[50, 50, 50, 10],
                             lambda_prox=[0, 0, 0, 50],
                             num_langevin=[10, 10, 10, 15])),
    ("K2_ultra_ns",     dict(**BASE, h_x=[0.1, 0.1, 0.1, 1.0],
                             lambda_reg=[50, 50, 50, 10],
                             lambda_prox=[0, 0, 0, 50],
                             noise_scale=[0, 0, 0, 0.3],
                             num_langevin=[10, 10, 10, 15])),
    ("K3_mega",         dict(**BASE, h_x=[0.1, 0.1, 0.1, 1.5],
                             lambda_reg=[50, 50, 50, 5],
                             lambda_prox=[0, 0, 0, 100],
                             num_langevin=[10, 10, 10, 20])),

    # ================================================================
    # L: Per-stage ramp schedules
    # ================================================================
    ("L1_ramp",         dict(**BASE, h_x=[0.1, 0.1, 0.1, 0.7],
                             lambda_prox=[0, 5, 10, 50])),
    ("L2_ramp_hx",      dict(**BASE, h_x=[0.1, 0.1, 0.3, 0.7],
                             lambda_prox=[0, 0, 10, 50])),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"debug5 sweep on {DEVICE}  ({len(CONFIGS)} configs)", flush=True)
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

    # Header
    hdr = "{:<3} {:<24} {:>5} {:>8} {:>8} {:>8} {:>6} {:>6} {:>5}"
    print(hdr.format("#", "name", "res", "PSNR", "PSNR_u", "SSIM",
                     "HF", "|dHF|", "t(s)"), flush=True)
    print("-" * 86, flush=True)

    results = {}
    for idx, (name, kw) in enumerate(CONFIGS, 1):
        try:
            xf, po, pa, res, t, hf = run_ip4(
                model, config, gt, y, operator, sigma_n, DEVICE, **kw)

            # Extra metrics on GPU
            xf_gpu = xf.to(DEVICE)
            ssim_val = compute_ssim(xf_gpu, gt)
            inv_mask = 1 - mask[0:1]  # unobserved=1
            psnr_u = psnr_region(xf_gpu, gt, inv_mask.expand_as(gt))
            xf_gpu = None  # free

            dhf = abs(hf - gt_hf)
            results[name] = dict(
                x=xf, psnr_all=pa, psnr_obs=po, psnr_unobs=psnr_u,
                ssim=ssim_val, res=res, hf=hf, dhf=dhf, time=t, kw=kw,
            )
            print(hdr.format(idx, name, f"{res:.0f}", f"{pa:.2f}",
                             f"{psnr_u:.2f}", f"{ssim_val:.4f}",
                             f"{hf:.3f}", f"{dhf:.3f}", f"{t:.0f}"),
                  flush=True)
        except Exception as e:
            print(f"{idx:<3} {name:<24} ERROR: {e}", flush=True)

    # --------------- Rankings ---------------
    print(f"\n{'=' * 86}")
    print(f"GT HF_unobs = {gt_hf:.3f}")

    # Composite score: balance PSNR and HF proximity
    # Normalize both to [0,1] within sweep range, then average
    psnrs = [v["psnr_all"] for v in results.values()]
    dhfs = [v["dhf"] for v in results.values()]
    pmin, pmax = min(psnrs), max(psnrs)
    dmin, dmax = min(dhfs), max(dhfs)
    for v in results.values():
        pnorm = (v["psnr_all"] - pmin) / max(pmax - pmin, 1e-8)
        dnorm = 1 - (v["dhf"] - dmin) / max(dmax - dmin, 1e-8)  # invert: smaller dhf = better
        v["composite"] = 0.5 * pnorm + 0.5 * dnorm

    for rank_by, label, key, rev in [
        ("PSNR_all", "PSNR", "psnr_all", True),
        ("|dHF|", "HF proximity", "dhf", False),
        ("SSIM", "SSIM", "ssim", True),
        ("Composite", "Balanced (PSNR + HF)", "composite", True),
    ]:
        ranked = sorted(results.items(), key=lambda kv: kv[1][key], reverse=rev)
        print(f"\n===== RANKED BY {label} =====")
        print(hdr.format("#", "name", "res", "PSNR", "PSNR_u", "SSIM",
                         "HF", "|dHF|", "t(s)"))
        for i, (n, v) in enumerate(ranked[:15], 1):
            print(hdr.format(i, n, f"{v['res']:.0f}", f"{v['psnr_all']:.2f}",
                             f"{v['psnr_unobs']:.2f}", f"{v['ssim']:.4f}",
                             f"{v['hf']:.3f}", f"{v['dhf']:.3f}",
                             f"{v['time']:.0f}"))

    # --------------- Save JSON ---------------
    summary = {}
    for k, v in results.items():
        summary[k] = {kk: vv for kk, vv in v.items() if kk != "x"}
    with open(f"{RESULT_DIR}/sweep_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # --------------- Visualization ---------------
    by_composite = sorted(results.items(),
                          key=lambda kv: kv[1]["composite"], reverse=True)
    keys = [k for k, _ in by_composite]
    n = len(keys)
    cols = min(8, n + 1)
    rows = (n + 1 + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.4))
    if rows == 1:
        axes = axes[np.newaxis, :]
    axes = axes.flatten()

    img_gt = (gt[0].cpu().permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1).numpy()
    axes[0].imshow(img_gt)
    axes[0].set_title(f"GT (HF={gt_hf:.3f})", fontsize=5)
    axes[0].axis("off")

    for i, k in enumerate(keys):
        img = (results[k]["x"][0].permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1).numpy()
        axes[i + 1].imshow(img)
        v = results[k]
        axes[i + 1].set_title(
            f"{k}\n{v['psnr_all']:.1f}dB ss={v['ssim']:.3f}\nhf={v['hf']:.3f} |d|={v['dhf']:.3f}",
            fontsize=4)
        axes[i + 1].axis("off")

    for j in range(len(keys) + 1, len(axes)):
        axes[j].axis("off")
    plt.suptitle(f"debug5 sweep ({n} configs)  GT HF={gt_hf:.3f}", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/sweep_all.png", dpi=200)
    plt.close()

    # Detail comparison of top-10
    top10 = by_composite[:10]
    fig2, axes2 = plt.subplots(2, 6, figsize=(18, 6.5))
    axes2 = axes2.flatten()
    axes2[0].imshow(img_gt)
    axes2[0].set_title(f"GT\nHF={gt_hf:.3f}", fontsize=7)
    axes2[0].axis("off")
    # y (measurement)
    img_y = ((mask[0:1] * gt)[0].cpu().permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1).numpy()
    axes2[1].imshow(img_y)
    axes2[1].set_title("Measurement", fontsize=7)
    axes2[1].axis("off")
    for i, (k, v) in enumerate(top10):
        img = (v["x"][0].permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1).numpy()
        axes2[i + 2].imshow(img)
        axes2[i + 2].set_title(
            f"#{i+1} {k}\nPSNR={v['psnr_all']:.2f} SSIM={v['ssim']:.4f}\n"
            f"HF={v['hf']:.3f} |dHF|={v['dhf']:.3f}",
            fontsize=5)
        axes2[i + 2].axis("off")
    plt.suptitle("Top-10 by composite score (PSNR + HF proximity)", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/top10_detail.png", dpi=200)
    plt.close()

    print(f"\nSaved {RESULT_DIR}/sweep_all.png")
    print(f"Saved {RESULT_DIR}/top10_detail.png")
    print(f"Saved {RESULT_DIR}/sweep_results.json")
    print(f"\nDONE. Total configs: {len(results)}", flush=True)


if __name__ == "__main__":
    main()
