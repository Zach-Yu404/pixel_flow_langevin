#!/usr/bin/env python
"""
IP4 Round-6 — lambda_prox locked at 50 (paper original), grid h_x × h_epsilon,
and four novel g_eps augmentation terms:

  N1 (norm preservation):   g_eps += λ_norm · (1 − ‖eps‖²/N) · eps
  N2 (mask-aware eps):      in unobs region, drop Tweedie term → only −eps (pure prior)
  N3 (proximal to eps_0):   g_eps += λ_prox_ε · (eps_0 − eps_k)
  N4 (noise injection):     g_eps += λ_inj · ξ (fresh Gaussian per step)

All have terminal_replace_weight=1.0 to zero res. We judge by visual + PSNR + HF.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from inpaintingStart import get_operator
from pixelflow.utils import config as config_utils

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "debug_IP4"))
from ms_sampler_v5 import run_ip4, hf_energy

DEVICE = "cuda:0"
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_round6")
os.makedirs(RESULT_DIR, exist_ok=True)


def load_gt():
    pt = "trajectory_videos/posterior_sampling/baseline_05/baseline_05_mode-f_x1.pt"
    d = torch.load(pt, map_location="cpu", weights_only=False)
    return d["gt"][:2].to(DEVICE)


def main():
    print(f"Round-6 on {DEVICE}", flush=True)
    gt = load_gt()
    config = OmegaConf.load("pretrained_models/c2img/config.yaml")
    model = config_utils.instantiate_from_config(config.model).to(DEVICE)
    ckpt = torch.load("pretrained_models/c2img/model.pt",
                      map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt, strict=True); model.eval()

    sigma_n = 0.05
    operator = get_operator("inpainting", resolution=256, device=DEVICE,
                             sigma=sigma_n, mask_type="box",
                             mask_len_range=(80, 160), mask_prob_range=None)
    y = operator(gt).detach()
    mask = operator.get_mask(x=gt).float().to(DEVICE)
    gt_hf = hf_energy(gt * (1 - mask[0:1]))
    print(f"GT HF_unobs={gt_hf:.3f}", flush=True)

    # lambda_prox locked at 50 across all configs; terminal_replace=1 always on
    LP = 50.0
    TR = 1.0
    # All configs inherit: lambda_prox=LP, terminal_replace_weight=TR

    def cfg(**kw):
        out = dict(lambda_prox=LP, terminal_replace_weight=TR)
        out.update(kw); return out

    configs = [
        # ─── Reference ───
        ("R6_base",                       cfg()),
        ("R6_prox_off",                   cfg(lambda_prox=0.0)),

        # ─── Grid: h_x stage-3 × h_epsilon (lambda_prox=50 fixed) ───
        # h_x ∈ {0.1, 0.3, 0.5, 0.7, 1.0} × h_eps ∈ {0.01, 0.001, 0.0001}
        ("G_hx0.1_heps0.01",   cfg(h_x=0.1, h_epsilon=0.01)),     # = R6_base
        ("G_hx0.1_heps0.001",  cfg(h_x=0.1, h_epsilon=0.001)),
        ("G_hx0.1_heps0.0001", cfg(h_x=0.1, h_epsilon=0.0001)),
        ("G_hx0.3_heps0.01",   cfg(h_x=[0.1,0.1,0.1,0.3], h_epsilon=0.01)),
        ("G_hx0.3_heps0.001",  cfg(h_x=[0.1,0.1,0.1,0.3], h_epsilon=0.001)),
        ("G_hx0.3_heps0.0001", cfg(h_x=[0.1,0.1,0.1,0.3], h_epsilon=0.0001)),
        ("G_hx0.5_heps0.01",   cfg(h_x=[0.1,0.1,0.1,0.5], h_epsilon=0.01)),
        ("G_hx0.5_heps0.001",  cfg(h_x=[0.1,0.1,0.1,0.5], h_epsilon=0.001)),
        ("G_hx0.5_heps0.0001", cfg(h_x=[0.1,0.1,0.1,0.5], h_epsilon=0.0001)),
        ("G_hx0.7_heps0.01",   cfg(h_x=[0.1,0.1,0.1,0.7], h_epsilon=0.01)),
        ("G_hx0.7_heps0.001",  cfg(h_x=[0.1,0.1,0.1,0.7], h_epsilon=0.001)),   # WINNER3 + prox
        ("G_hx0.7_heps0.0001", cfg(h_x=[0.1,0.1,0.1,0.7], h_epsilon=0.0001)),
        ("G_hx1.0_heps0.001",  cfg(h_x=[0.1,0.1,0.1,1.0], h_epsilon=0.001)),

        # ─── N1: eps norm preservation ───
        ("N1_norm_0.01_base",   cfg(lambda_eps_norm=0.01)),
        ("N1_norm_0.1_base",    cfg(lambda_eps_norm=0.1)),
        ("N1_norm_1.0_base",    cfg(lambda_eps_norm=1.0)),
        ("N1_norm_10_base",     cfg(lambda_eps_norm=10.0)),
        ("N1_norm_1_hx0.7",     cfg(lambda_eps_norm=1.0, h_x=[0.1,0.1,0.1,0.7])),
        ("N1_norm_10_hx0.7",    cfg(lambda_eps_norm=10.0, h_x=[0.1,0.1,0.1,0.7])),

        # ─── N2: mask-aware eps (unobs region: no Tweedie, pure prior) ───
        ("N2_mask_aware_base",       cfg(mask_aware_eps=True)),
        ("N2_mask_aware_hx0.3",      cfg(mask_aware_eps=True, h_x=[0.1,0.1,0.1,0.3])),
        ("N2_mask_aware_hx0.5",      cfg(mask_aware_eps=True, h_x=[0.1,0.1,0.1,0.5])),
        ("N2_mask_aware_hx0.7",      cfg(mask_aware_eps=True, h_x=[0.1,0.1,0.1,0.7])),
        ("N2_mask_aware_hx0.7_heps0.001",
                                     cfg(mask_aware_eps=True, h_x=[0.1,0.1,0.1,0.7],
                                         h_epsilon=0.001)),

        # ─── N3: proximal to initial eps_0 ───
        ("N3_eps_prox_0.1_base",  cfg(lambda_eps_prox=0.1)),
        ("N3_eps_prox_1_base",    cfg(lambda_eps_prox=1.0)),
        ("N3_eps_prox_10_base",   cfg(lambda_eps_prox=10.0)),
        ("N3_eps_prox_50_base",   cfg(lambda_eps_prox=50.0)),
        ("N3_eps_prox_1_hx0.7",   cfg(lambda_eps_prox=1.0, h_x=[0.1,0.1,0.1,0.7])),
        ("N3_eps_prox_10_hx0.7",  cfg(lambda_eps_prox=10.0, h_x=[0.1,0.1,0.1,0.7])),

        # ─── N4: active noise injection ───
        ("N4_inj_0.01_base",  cfg(lambda_eps_inject=0.01)),
        ("N4_inj_0.1_base",   cfg(lambda_eps_inject=0.1)),
        ("N4_inj_0.3_base",   cfg(lambda_eps_inject=0.3)),
        ("N4_inj_1.0_base",   cfg(lambda_eps_inject=1.0)),
        ("N4_inj_0.1_hx0.7",  cfg(lambda_eps_inject=0.1, h_x=[0.1,0.1,0.1,0.7])),
        ("N4_inj_0.3_hx0.7",  cfg(lambda_eps_inject=0.3, h_x=[0.1,0.1,0.1,0.7])),

        # ─── ULTRA combos: best from each term ───
        ("U1_N2_N3",     cfg(mask_aware_eps=True, lambda_eps_prox=1.0,
                              h_x=[0.1,0.1,0.1,0.7])),
        ("U2_N2_N4",     cfg(mask_aware_eps=True, lambda_eps_inject=0.1,
                              h_x=[0.1,0.1,0.1,0.7])),
        ("U3_N2_N3_heps", cfg(mask_aware_eps=True, lambda_eps_prox=1.0,
                              h_x=[0.1,0.1,0.1,0.7], h_epsilon=0.001)),
        ("U4_N1_N2_hx0.7", cfg(mask_aware_eps=True, lambda_eps_norm=1.0,
                                h_x=[0.1,0.1,0.1,0.7])),
    ]

    results = {}
    header = "{:<3} {:<36} {:>5} {:>7} {:>7} {:>6} {:>5}"
    print(header.format("#", "name", "res", "psnrO", "psnrAll", "HF", "t(s)"), flush=True)
    print("-"*80, flush=True)

    for i, (name, kw) in enumerate(configs, 1):
        try:
            xf, po, pa, res, t, hf = run_ip4(model, config, gt, y, operator, sigma_n, DEVICE, **kw)
            # Correct PSNR
            pa_fix = (-10 * torch.log10((((xf.to(DEVICE)+1)/2 - (gt+1)/2) ** 2).mean())).item()
            results[name] = dict(x=xf.to(DEVICE), psnr_obs=po, psnr_all=pa, psnr_all_fix=pa_fix,
                                  res=res, time=t, hf=hf, kw=kw)
            print(header.format(i, name, f"{res:.0f}", f"{po:.2f}", f"{pa_fix:.2f}",
                                 f"{hf:.3f}", f"{t:.0f}"), flush=True)
        except Exception as e:
            print(f"{i:<3} {name:<36} ERROR: {e}", flush=True)

    # Save summary
    summary = {k: {kk: vv for kk, vv in v.items() if kk != "x"} for k, v in results.items()}
    for k in summary:
        summary[k]["kw"] = {kk: vv for kk, vv in summary[k]["kw"].items()}
    with open(f"{RESULT_DIR}/round6_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Rankings
    by_psnr = sorted(results.items(), key=lambda kv: kv[1]["psnr_all_fix"], reverse=True)
    by_hf = sorted(results.items(), key=lambda kv: abs(kv[1]["hf"] - gt_hf))

    print(f"\n===== RANKED BY PSNR_all [0,1] =====", flush=True)
    for i, (n, v) in enumerate(by_psnr, 1):
        print(header.format(i, n, f"{v['res']:.0f}", f"{v['psnr_obs']:.2f}",
                             f"{v['psnr_all_fix']:.2f}", f"{v['hf']:.3f}", f"{v['time']:.0f}"), flush=True)
    print(f"\n===== RANKED BY HF proximity (GT={gt_hf:.3f}) =====", flush=True)
    for i, (n, v) in enumerate(by_hf, 1):
        d = abs(v["hf"] - gt_hf)
        print(f"{i:<3} {n:<36} HF={v['hf']:.3f} |Δ|={d:.3f} PSNR={v['psnr_all_fix']:.2f}", flush=True)

    # Pareto
    pts = [(v["psnr_all_fix"], abs(v["hf"] - gt_hf), n) for n, v in results.items()]
    pts_sorted = sorted(pts, key=lambda t: t[0], reverse=True)
    pareto = []; best_hf = float("inf")
    for ps, hfd, n in pts_sorted:
        if hfd < best_hf:
            pareto.append(n); best_hf = hfd
    print(f"\n===== PARETO FRONTIER =====", flush=True)
    for n in pareto:
        v = results[n]; d = abs(v["hf"] - gt_hf)
        print(f"{n:<36} PSNR={v['psnr_all_fix']:.2f} HF|Δ|={d:.3f} res={v['res']:.0f}", flush=True)

    # Visualization
    keys = [k for k, _ in by_psnr]
    cols = 8; rows = (len(keys) + 1 + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.8, rows*2.8))
    if rows == 1: axes = axes[np.newaxis, :]
    axes = axes.flatten()
    img_gt = ((gt[0]*0.5+0.5).clamp(0,1).permute(1,2,0).cpu().numpy())
    axes[0].imshow(img_gt); axes[0].set_title("GT", fontsize=6); axes[0].axis("off")
    for i, k in enumerate(keys):
        img = ((results[k]["x"][0]*0.5+0.5).clamp(0,1).permute(1,2,0).cpu().numpy())
        axes[i+1].imshow(img)
        axes[i+1].set_title(f"{k}\n{results[k]['psnr_all_fix']:.1f}dB hf={results[k]['hf']:.3f}",
                             fontsize=5)
        axes[i+1].axis("off")
    for j in range(i+2, len(axes)): axes[j].axis("off")
    plt.suptitle(f"IP4 round-6 (lambda_prox=50 + g_eps augmentation)  GT HF={gt_hf:.3f}", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/round6.png", dpi=140)
    plt.close()
    print(f"Saved {RESULT_DIR}/round6.png", flush=True)


if __name__ == "__main__":
    main()
