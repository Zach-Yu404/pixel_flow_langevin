"""Targeted comparison: does combining global h_eps=0.001 with X4's h_x help
OR do they conflict? Also test W1 + F1 = the proposed practical winner.
Same mask/GT as visualize_w1_effect.py (pull from the SAME seed path)."""
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

DEVICE = "cuda:3"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_winner_visual")
os.makedirs(OUT, exist_ok=True)


def load_gt():
    pt = "trajectory_videos/posterior_sampling/baseline_05/baseline_05_mode-f_x1.pt"
    d = torch.load(pt, map_location="cpu", weights_only=False)
    return d["gt"][:2].to(DEVICE)


def denorm(x):
    return (x * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def psnr_fixed(xf, gt, mask, obs):
    xf01 = (xf + 1) / 2; gt01 = (gt + 1) / 2
    m = mask if obs else (1 - mask)
    n = m.sum().item() * xf.shape[1]
    mse = ((m * (xf01 - gt01)) ** 2).sum() / (xf.shape[0] * n)
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
    mask = operator.get_mask(x=gt).float().to(DEVICE)
    gt_hf = hf_energy(gt * (1 - mask[0:1]))
    print(f"GT HF_unobs={gt_hf:.3f}", flush=True)

    # Targeted head-to-head of PROPOSED WINNERS vs established baselines
    configs = [
        ("GT_reference", None),
        ("baseline",                 dict()),
        ("W1_heps_global",           dict(h_epsilon=0.001)),
        ("WINNER1_W1_F1",            dict(h_epsilon=0.001, terminal_replace_weight=1.0)),
        ("WINNER2_W1_X4",            dict(h_epsilon=0.001, h_x=[0.1,0.1,0.1,0.7])),
        ("WINNER3_W1_X4_F1",         dict(h_epsilon=0.001, h_x=[0.1,0.1,0.1,0.7],
                                          terminal_replace_weight=1.0)),
        ("W1_with_lower_heps_0.0001",dict(h_epsilon=0.0001)),
        ("W1_lower_heps_F1",         dict(h_epsilon=0.0001, terminal_replace_weight=1.0)),
        ("X4_F1",                    dict(h_x=[0.1,0.1,0.1,0.7], terminal_replace_weight=1.0)),
    ]

    results = {}
    for name, kw in configs:
        if kw is None:
            continue
        print(f"running {name}…", flush=True)
        xf, po, pa, res, t, hf = run_ip4(model, config, gt, y, operator, sigma_n, DEVICE, **kw)
        results[name] = dict(x=xf.to(DEVICE), hf=hf, res=res, t=t)

    print(f"\n{'config':<28} {'PSNR_all':>9} {'PSNR_obs':>10} {'PSNR_unobs':>11} {'HF':>6} {'res':>7}")
    print("-"*80)
    for n, v in results.items():
        pa = -10 * torch.log10((((v["x"]+1)/2 - (gt+1)/2) ** 2).mean()).item()
        po = psnr_fixed(v["x"], gt, mask, obs=True)
        pu = psnr_fixed(v["x"], gt, mask, obs=False)
        print(f"{n:<28} {pa:>9.2f} {po:>10.2f} {pu:>11.2f} {v['hf']:>6.3f} {v['res']:>7.1f}")

    # render all
    n = len(results) + 1
    cols = 3; rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    axes = axes.flatten()
    axes[0].imshow(denorm(gt[0])); axes[0].set_title("GT", fontsize=10); axes[0].axis("off")
    # box bbox
    inv = 1 - mask[0, 0].cpu().numpy()
    ys, xs = np.where(inv > 0.5)
    if len(ys):
        y0, y1 = ys.min(), ys.max()+1; x0, x1 = xs.min(), xs.max()+1
    for i, (name, _) in enumerate([c for c in configs if c[0] != "GT_reference"]):
        img = denorm(results[name]["x"][0])
        axes[i+1].imshow(img)
        pa = -10 * torch.log10((((results[name]["x"]+1)/2 - (gt+1)/2) ** 2).mean()).item()
        pu = psnr_fixed(results[name]["x"], gt, mask, obs=False)
        axes[i+1].set_title(f"{name}\nHF={results[name]['hf']:.3f} "
                             f"PSNRall={pa:.1f} PSNRunobs={pu:.1f}", fontsize=9)
        axes[i+1].add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0,
                                           ec="red", fc="none", lw=2))
        axes[i+1].axis("off")
    for j in range(i+2, len(axes)): axes[j].axis("off")
    plt.suptitle(f"Winner head-to-head (GT HF={gt_hf:.3f})", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{OUT}/winner_comparison.png", dpi=140, bbox_inches="tight")
    plt.close()

    # individual high-res
    for name in results:
        img = denorm(results[name]["x"][0])
        plt.figure(figsize=(7, 7)); plt.imshow(img); plt.axis("off")
        plt.title(name); plt.tight_layout()
        plt.savefig(f"{OUT}/{name}.png", dpi=160, bbox_inches="tight")
        plt.close()
    plt.figure(figsize=(7,7)); plt.imshow(denorm(gt[0])); plt.axis("off")
    plt.title("GT"); plt.tight_layout()
    plt.savefig(f"{OUT}/GT.png", dpi=160, bbox_inches="tight"); plt.close()
    print(f"\nSaved → {OUT}/")


if __name__ == "__main__":
    main()
