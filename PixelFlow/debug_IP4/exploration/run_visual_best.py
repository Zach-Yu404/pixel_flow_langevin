"""Ignore PSNR — aggressive box visual search.
Combines multiple perceptual levers to maximize semantic content generation."""
import os, sys, json
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
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_ana", "visual_best_box")
os.makedirs(OUT, exist_ok=True)
os.makedirs(f"{OUT}/configs", exist_ok=True)
os.makedirs(f"{OUT}/images", exist_ok=True)


def load_gt():
    pt = "trajectory_videos/posterior_sampling/baseline_05/baseline_05_mode-f_x1.pt"
    return torch.load(pt, map_location="cpu", weights_only=False)["gt"][:2].to(DEVICE)


def denorm(x):
    return (x*0.5+0.5).clamp(0,1).permute(1,2,0).cpu().numpy()


# All visually aggressive configs — no regard for PSNR
# All enable F1 terminal_replace for data consistency (free, doesn't hurt visual)
CONFIGS = [
    ("A0_reference_WINNER3", {
        "desc": "Reference: WINNER3 (W1 + X4 h_x=0.7 + F1). Baseline of best known visual.",
        "kw": dict(h_epsilon=0.001, h_x=[0.1, 0.1, 0.1, 0.7], terminal_replace_weight=1.0),
    }),
    # ═══ Push h_x stage-3 further ═══
    ("B1_W1+X_hx1.0", {
        "desc": "More aggressive h_x stg3 = 1.0 (doubles the prior-score amplification)",
        "kw": dict(h_epsilon=0.001, h_x=[0.1, 0.1, 0.1, 1.0], terminal_replace_weight=1.0),
    }),
    ("B2_W1+X_hx1.5", {
        "desc": "Even larger h_x stg3 = 1.5",
        "kw": dict(h_epsilon=0.001, h_x=[0.1, 0.1, 0.1, 1.5], terminal_replace_weight=1.0),
    }),
    # ═══ Multi-stage h_x ramp ═══
    ("C1_W1+hx_multistage", {
        "desc": "h_x progressively increases per stage: [0.1, 0.2, 0.4, 0.7]",
        "kw": dict(h_epsilon=0.001, h_x=[0.1, 0.2, 0.4, 0.7], terminal_replace_weight=1.0),
    }),
    ("C2_W1+hx_aggressive_multistage", {
        "desc": "h_x [0.2, 0.4, 0.7, 1.0] — amplify across all stages",
        "kw": dict(h_epsilon=0.001, h_x=[0.2, 0.4, 0.7, 1.0], terminal_replace_weight=1.0),
    }),
    # ═══ V (h_eps=1e-4) + X4 ═══
    ("D1_V+X4", {
        "desc": "V (h_eps=1e-4) + h_x stg3=0.7. Maximum eps entropy + X4 amplification.",
        "kw": dict(h_epsilon=0.0001, h_x=[0.1, 0.1, 0.1, 0.7], terminal_replace_weight=1.0),
    }),
    ("D2_V+hx1.0", {
        "desc": "V + h_x stg3=1.0 — ultra aggressive",
        "kw": dict(h_epsilon=0.0001, h_x=[0.1, 0.1, 0.1, 1.0], terminal_replace_weight=1.0),
    }),
    # ═══ W1 + lambda_reg drop stg3 (CG softening) ═══
    ("E1_W1+lam20", {
        "desc": "W1 + lambda_reg stg3=20 (CG softer → Langevin steps bigger)",
        "kw": dict(h_epsilon=0.001, lambda_reg=[50, 50, 50, 20], terminal_replace_weight=1.0),
    }),
    ("E2_W1+lam10", {
        "desc": "W1 + lambda_reg stg3=10 (even softer CG)",
        "kw": dict(h_epsilon=0.001, lambda_reg=[50, 50, 50, 10], terminal_replace_weight=1.0),
    }),
    # ═══ Combined h_x + lambda + h_eps ═══
    ("F1_V+X4+lam20", {
        "desc": "V (h_eps=1e-4) + h_x stg3=0.7 + lambda_reg stg3=20 — 3-lever push",
        "kw": dict(h_epsilon=0.0001, h_x=[0.1, 0.1, 0.1, 0.7],
                   lambda_reg=[50, 50, 50, 20], terminal_replace_weight=1.0),
    }),
    ("F2_W1+X4+lam20", {
        "desc": "W1 + X4 h_x=0.7 + lambda_reg stg3=20 — more moderate 3-lever",
        "kw": dict(h_epsilon=0.001, h_x=[0.1, 0.1, 0.1, 0.7],
                   lambda_reg=[50, 50, 50, 20], terminal_replace_weight=1.0),
    }),
    # ═══ More Langevin at stage 3 ═══
    ("G1_W1+X4+L20_stg3", {
        "desc": "WINNER3 + 20 Langevin iters at stage 3 (2x more refinement)",
        "kw": dict(h_epsilon=0.001, h_x=[0.1, 0.1, 0.1, 0.7],
                   num_langevin=[10, 10, 10, 20], terminal_replace_weight=1.0),
    }),
    # ═══ W1 + noise injection stg3 ═══
    ("H1_W1+X4+noise0.3", {
        "desc": "WINNER3 + noise_scale=0.3 at stage 3 — stochastic texture",
        "kw": dict(h_epsilon=0.001, h_x=[0.1, 0.1, 0.1, 0.7],
                   noise_scale=[0, 0, 0, 0.3], terminal_replace_weight=1.0),
    }),
    # ═══ CFG — quick retest on top of W1 ═══
    ("I1_W1+X4+cfg2", {
        "desc": "WINNER3 with CFG=2.0 (was dead in earlier tests; quick retest at new recipe)",
        "kw": dict(h_epsilon=0.001, h_x=[0.1, 0.1, 0.1, 0.7],
                   guidance_scale=2.0, terminal_replace_weight=1.0),
    }),
    # ═══ Ultimate combo ═══
    ("Z_ULTIMATE", {
        "desc": "All-in: V (1e-4) + h_x multi-stage [0.2,0.4,0.7,1.0] + lam_stg3=20 + L=15 + F1",
        "kw": dict(h_epsilon=0.0001,
                   h_x=[0.2, 0.4, 0.7, 1.0],
                   lambda_reg=[50, 50, 50, 20],
                   num_langevin=[10, 10, 10, 15],
                   terminal_replace_weight=1.0),
    }),
]


def main():
    gt = load_gt()
    config = OmegaConf.load("pretrained_models/c2img/config.yaml")
    model = config_utils.instantiate_from_config(config.model).to(DEVICE)
    ckpt = torch.load("pretrained_models/c2img/model.pt",
                      map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt, strict=True); model.eval()

    for name, info in CONFIGS:
        with open(f"{OUT}/configs/{name}.json", "w") as f:
            json.dump({"name": name, **info}, f, indent=2, default=str)

    sigma_n = 0.05
    operator = get_operator("inpainting", resolution=256, device=DEVICE,
                             sigma=sigma_n, mask_type="box",
                             mask_len_range=(80, 160), mask_prob_range=None)
    y = operator(gt).detach()
    mask = operator.get_mask(x=gt).float().to(DEVICE)
    gt_hf = hf_energy(gt * (1 - mask[0:1]))
    print(f"BOX GT HF = {gt_hf:.3f}", flush=True)

    results = {}
    print(f"{'name':<34} {'PSNR_all':>9} {'PSNR_unobs':>11} {'HF':>7} {'|Δ|':>7}", flush=True)
    print("-"*75, flush=True)
    for name, info in CONFIGS:
        kw = info["kw"]
        try:
            xf, po, pa, res, t, hf = run_ip4(model, config, gt, y, operator, sigma_n, DEVICE, **kw)
            xfD = xf.to(DEVICE)
            mu = 1 - mask; nu = mu.sum().item() * xfD.shape[1]
            mse_u = ((mu * ((xfD+1)/2 - (gt+1)/2))**2).sum() / (xfD.shape[0] * max(nu, 1))
            pu = (-10 * torch.log10(mse_u.clamp(min=1e-12))).item()
            results[name] = dict(x=xfD, psnr_all=pa, psnr_unobs=pu,
                                  hf=hf, hf_delta=abs(hf - gt_hf), res=res)
            print(f"{name:<34} {pa:>9.2f} {pu:>11.2f} {hf:>7.3f} {abs(hf-gt_hf):>7.3f}",
                  flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"{name} ERROR: {e}", flush=True)

    # save PNGs
    plt.imsave(f"{OUT}/images/GT.png", denorm(gt[0]))
    for name, v in results.items():
        plt.imsave(f"{OUT}/images/{name}.png", denorm(v["x"][0]))

    # grid
    n = len(results) + 1
    cols = 4; rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    axes = axes.flatten()
    inv = 1 - mask[0, 0].cpu().numpy()
    ys, xs = np.where(inv > 0.5)
    if len(ys):
        y0, y1 = ys.min(), ys.max()+1; x0, x1 = xs.min(), xs.max()+1

    axes[0].imshow(denorm(gt[0])); axes[0].set_title(f"GT  (HF={gt_hf:.3f})", fontsize=10)
    axes[0].axis("off")
    for i, (name, _) in enumerate(CONFIGS):
        if name not in results: continue
        v = results[name]
        axes[i+1].imshow(denorm(v["x"][0]))
        axes[i+1].set_title(f"{name}\nPSNR={v['psnr_all']:.1f} unobs={v['psnr_unobs']:.1f}\n"
                              f"HF={v['hf']:.3f} |Δ|={v['hf_delta']:.3f}", fontsize=7)
        if len(ys):
            axes[i+1].add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0,
                                                ec="red", fc="none", lw=2))
        axes[i+1].axis("off")
    for j in range(i+2, len(axes)): axes[j].axis("off")
    plt.suptitle(f"Visual-first aggressive configs — BOX  (GT HF={gt_hf:.3f})  PSNR not prioritized",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{OUT}/comparison.png", dpi=140, bbox_inches="tight")
    plt.close()

    # metrics json
    summary = {k: {kk: vv for kk, vv in v.items() if kk != "x"} for k, v in results.items()}
    with open(f"{OUT}/metrics.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nSaved → {OUT}/", flush=True)


if __name__ == "__main__":
    main()
