"""Reproduce the most critical 4 configs in R2's frame for direct visual comparison.
Saves individual PNGs + a compact side-by-side panel focused on the box region.

Configs to compare (R2 GT HF=0.623):
  - baseline (R2_base-equivalent, use Z3 defaults)
  - W1 h_epsilon=0.001        (sampling-only; the "meaningful content" finding)
  - X4 h_x stg3=0.7           (sampling-only; the round-3 sampling winner)
  - F1 equivalent terminal_replace=1.0 (projection; preserves baseline texture)
"""
import os, sys
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

DEVICE = "cuda:3"    # GPU 3 idle
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_w1_visual")
os.makedirs(OUT, exist_ok=True)


def load_gt():
    pt = "trajectory_videos/posterior_sampling/baseline_05/baseline_05_mode-f_x1.pt"
    d = torch.load(pt, map_location="cpu", weights_only=False)
    return d["gt"][:2].to(DEVICE)


def denorm(x):
    return (x * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def psnr_fixed(xf, gt, mask, obs=True):
    """Correct PSNR in [0,1] convention with proper mask denominator."""
    xf01 = (xf + 1) / 2
    gt01 = (gt + 1) / 2
    if obs:
        m = mask
        n_valid = m.sum().item() * xf.shape[1]  # C channels
    else:
        m = 1 - mask
        n_valid = m.sum().item() * xf.shape[1]
    mse = ((m * (xf01 - gt01)) ** 2).sum() / (xf.shape[0] * n_valid)
    return -10 * torch.log10(mse.clamp(min=1e-12)).item()


def main():
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
    mask = operator.get_mask(x=gt).float().to(DEVICE)      # 1=observed
    gt_hf = hf_energy(gt * (1 - mask[0:1]))
    print(f"GT HF_unobs={gt_hf:.3f}", flush=True)

    # locate box corners for zoomed view
    inv = 1 - mask[0, 0].cpu().numpy()
    ys, xs = np.where(inv > 0.5)
    if len(ys):
        y0, y1 = ys.min(), ys.max()+1
        x0, x1 = xs.min(), xs.max()+1
    else:
        y0, y1, x0, x1 = 64, 192, 64, 192
    print(f"box bbox: y=[{y0},{y1}]  x=[{x0},{x1}]")

    configs = [
        ("baseline",     dict()),
        ("W1_heps_0.001",dict(h_epsilon=0.001)),
        ("H1_heps_stg3_0.001", dict(h_epsilon=[0.01,0.01,0.01,0.001])),
        ("K1_skip_eps_all",   dict(skip_eps_update=True)),
        ("K2_skip_eps_stg3",  dict(skip_eps_update=[False,False,False,True])),
        ("J1_reset_eps_all",  dict(reset_eps_per_ode_step=True)),
        ("J2_reset_eps_stg3", dict(reset_eps_per_ode_step=[False,False,False,True])),
        ("X4_hx0.7",     dict(h_x=[0.1,0.1,0.1,0.7])),
        ("F1_termrep",   dict(terminal_replace_weight=1.0)),
        ("WC3_heps+hx0.7", dict(h_epsilon=[0.01,0.01,0.01,0.001], h_x=[0.1,0.1,0.1,0.7])),
    ]

    results = {}
    for name, kw in configs:
        print(f"running {name}…", flush=True)
        xf, po, pa, res, t, hf = run_ip4(model, config, gt, y, operator, sigma_n, DEVICE, **kw)
        results[name] = dict(x=xf.to(DEVICE), psnr_obs=po, psnr_all=pa, res=res, t=t, hf=hf)

    # Correct PSNR recomputation
    print(f"\n{'config':<22} {'PSNR_all ([0,1])':>18} {'PSNR_obs':>12} {'PSNR_unobs':>12} {'HF':>7} {'res':>7}")
    print("-"*85)
    for n, v in results.items():
        pa_fix = psnr_fixed(v["x"], gt, mask, obs=False) if False else (
            -10 * torch.log10((((v["x"]+1)/2 - (gt+1)/2) ** 2).mean()).item())
        po_fix = psnr_fixed(v["x"], gt, mask, obs=True)
        pu_fix = psnr_fixed(v["x"], gt, mask, obs=False)
        print(f"{n:<22} {pa_fix:>18.2f} {po_fix:>12.2f} {pu_fix:>12.2f} {v['hf']:>7.3f} {v['res']:>7.1f}")

    # Full-image comparison — sample 0
    n = len(configs) + 1
    cols = 4; rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3.5, rows*3.5))
    axes = axes.flatten()
    axes[0].imshow(denorm(gt[0])); axes[0].set_title("GT", fontsize=10); axes[0].axis("off")
    for i, (name, _) in enumerate(configs):
        img = denorm(results[name]["x"][0])
        axes[i+1].imshow(img)
        axes[i+1].set_title(
            f"{name}\nHF={results[name]['hf']:.3f}", fontsize=8)
        # red bbox on box region
        axes[i+1].add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0,
                                           ec="red", fc="none", lw=1.5))
        axes[i+1].axis("off")
    for j in range(i+2, len(axes)): axes[j].axis("off")
    plt.suptitle(f"Full image view — box region outlined in red (GT HF={gt_hf:.3f})",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{OUT}/full_comparison.png", dpi=130, bbox_inches="tight")
    plt.close()

    # Zoomed BOX region
    pad = 8
    y0z, y1z = max(0, y0-pad), min(256, y1+pad)
    x0z, x1z = max(0, x0-pad), min(256, x1+pad)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3.5, rows*3.5))
    axes = axes.flatten()
    gt_zoom = denorm(gt[0])[y0z:y1z, x0z:x1z]
    axes[0].imshow(gt_zoom); axes[0].set_title(f"GT (box region)", fontsize=10)
    axes[0].axis("off")
    for i, (name, _) in enumerate(configs):
        img = denorm(results[name]["x"][0])[y0z:y1z, x0z:x1z]
        axes[i+1].imshow(img)
        axes[i+1].set_title(f"{name}\nHF={results[name]['hf']:.3f}", fontsize=8)
        axes[i+1].axis("off")
    for j in range(i+2, len(axes)): axes[j].axis("off")
    plt.suptitle("ZOOMED to box region — direct visual inspection of filled content",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{OUT}/box_zoom_comparison.png", dpi=160, bbox_inches="tight")
    plt.close()

    # Save individual images
    for name, _ in configs:
        img = denorm(results[name]["x"][0])
        plt.figure(figsize=(6, 6))
        plt.imshow(img); plt.axis("off")
        plt.title(f"{name} — HF={results[name]['hf']:.3f}")
        plt.tight_layout()
        plt.savefig(f"{OUT}/{name}.png", dpi=150, bbox_inches="tight")
        plt.close()
    plt.figure(figsize=(6, 6)); plt.imshow(denorm(gt[0])); plt.axis("off")
    plt.title("GT"); plt.tight_layout()
    plt.savefig(f"{OUT}/GT.png", dpi=150, bbox_inches="tight"); plt.close()

    print(f"\nSaved → {OUT}/")


if __name__ == "__main__":
    main()
