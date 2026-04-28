"""Final visual comparison of Round-6 top performers: does N2 mask-aware
actually produce semantic content like W1 did, or is it another metric trick?"""
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

DEVICE = "cuda:1"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_N2_visual")
os.makedirs(OUT, exist_ok=True)


def load_gt():
    pt = "trajectory_videos/posterior_sampling/baseline_05/baseline_05_mode-f_x1.pt"
    d = torch.load(pt, map_location="cpu", weights_only=False)
    return d["gt"][:2].to(DEVICE)


def denorm(x):
    return (x*0.5+0.5).clamp(0,1).permute(1,2,0).cpu().numpy()


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
    print(f"GT HF={gt_hf:.3f}", flush=True)

    configs = [
        ("baseline",                dict(terminal_replace_weight=1.0)),
        ("W1_heps_0.001",           dict(h_epsilon=0.001, terminal_replace_weight=1.0)),
        ("WINNER3_W1+X4+F1",        dict(h_epsilon=0.001, h_x=[0.1,0.1,0.1,0.7],
                                           terminal_replace_weight=1.0)),
        ("N2_mask_aware_alone",     dict(mask_aware_eps=True, terminal_replace_weight=1.0)),
        ("N2_mask_aware_hx0.3",     dict(mask_aware_eps=True, h_x=[0.1,0.1,0.1,0.3],
                                           terminal_replace_weight=1.0)),
        ("N2_mask_aware_hx0.5",     dict(mask_aware_eps=True, h_x=[0.1,0.1,0.1,0.5],
                                           terminal_replace_weight=1.0)),
        ("N2_mask_aware_hx0.7",     dict(mask_aware_eps=True, h_x=[0.1,0.1,0.1,0.7],
                                           terminal_replace_weight=1.0)),
        ("N2_plus_W1_hx0.5",        dict(mask_aware_eps=True, h_epsilon=0.001,
                                           h_x=[0.1,0.1,0.1,0.5], terminal_replace_weight=1.0)),
    ]

    results = {}
    for name, kw in configs:
        print(f"running {name}…", flush=True)
        xf, po, pa, res, t, hf = run_ip4(model, config, gt, y, operator, sigma_n, DEVICE, **kw)
        results[name] = dict(x=xf.to(DEVICE), hf=hf, res=res)

    print(f"\n{'config':<28} {'PSNR_all':>9} {'PSNR_unobs':>11} {'HF':>7} {'|Δ|':>7}")
    print("-"*80)
    for n, v in results.items():
        pa = -10 * torch.log10((((v["x"]+1)/2 - (gt+1)/2) ** 2).mean()).item()
        m_u = 1 - mask
        n_unobs = m_u.sum().item() * v["x"].shape[1]
        mse_u = ((m_u * ((v["x"]+1)/2 - (gt+1)/2)) ** 2).sum() / (v["x"].shape[0] * n_unobs)
        pu = -10 * torch.log10(mse_u.clamp(min=1e-12)).item()
        d = abs(v["hf"] - gt_hf)
        print(f"{n:<28} {pa:>9.2f} {pu:>11.2f} {v['hf']:>7.3f} {d:>7.3f}")

    inv = 1 - mask[0, 0].cpu().numpy()
    ys, xs = np.where(inv > 0.5)
    y0, y1 = ys.min(), ys.max()+1; x0, x1 = xs.min(), xs.max()+1

    cols = 3; rows = 3
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    axes = axes.flatten()
    axes[0].imshow(denorm(gt[0])); axes[0].set_title("GT", fontsize=10); axes[0].axis("off")
    for i, (name, _) in enumerate(configs):
        img = denorm(results[name]["x"][0])
        axes[i+1].imshow(img)
        pa = -10 * torch.log10((((results[name]["x"]+1)/2 - (gt+1)/2) ** 2).mean()).item()
        axes[i+1].set_title(f"{name}\nHF={results[name]['hf']:.3f} PSNR={pa:.1f}", fontsize=9)
        axes[i+1].add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0, ec="red", fc="none", lw=2))
        axes[i+1].axis("off")
    for j in range(i+2, len(axes)): axes[j].axis("off")
    plt.suptitle(f"N2 mask-aware-eps head-to-head vs W1/WINNER3 — GT HF={gt_hf:.3f}", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{OUT}/n2_vs_w1.png", dpi=140, bbox_inches="tight")
    plt.close()

    # individual high-res
    for name in results:
        img = denorm(results[name]["x"][0])
        plt.figure(figsize=(7,7)); plt.imshow(img); plt.axis("off")
        plt.title(name); plt.tight_layout()
        plt.savefig(f"{OUT}/{name}.png", dpi=160, bbox_inches="tight")
        plt.close()
    plt.figure(figsize=(7,7)); plt.imshow(denorm(gt[0])); plt.axis("off")
    plt.title("GT"); plt.tight_layout()
    plt.savefig(f"{OUT}/GT.png", dpi=160, bbox_inches="tight"); plt.close()
    print(f"\nSaved → {OUT}/")

if __name__ == "__main__":
    main()
