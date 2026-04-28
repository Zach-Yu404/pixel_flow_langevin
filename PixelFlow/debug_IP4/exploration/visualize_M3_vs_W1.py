"""R5 metric ranking has M3 (lambda_stg3=15) ranked #1 for HF proximity,
yet W1 (global h_eps=0.001) is much lower in PSNR + comparable HF.
Key question: does M3 actually produce semantic content like W1, or is
HF being gamed by mask-boundary edges / micro-grain? Visual inspection decides."""
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
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_m3_vs_w1")
os.makedirs(OUT, exist_ok=True)


def load_gt():
    pt = "trajectory_videos/posterior_sampling/baseline_05/baseline_05_mode-f_x1.pt"
    d = torch.load(pt, map_location="cpu", weights_only=False)
    return d["gt"][:2].to(DEVICE)


def denorm(x):
    return (x * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy()


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
        ("baseline",        dict()),
        ("M3_lam15",        dict(lambda_reg=[50,50,50,15])),
        ("M3_lam15_F1",     dict(lambda_reg=[50,50,50,15], terminal_replace_weight=1.0)),
        ("W1_heps_global",  dict(h_epsilon=0.001)),
        ("W1_heps_F1",      dict(h_epsilon=0.001, terminal_replace_weight=1.0)),
        ("WINNER3",         dict(h_epsilon=0.001, h_x=[0.1,0.1,0.1,0.7],
                                  terminal_replace_weight=1.0)),
        ("M3_plus_W1",      dict(lambda_reg=[50,50,50,15], h_epsilon=0.001)),
        ("M3_plus_W1_F1",   dict(lambda_reg=[50,50,50,15], h_epsilon=0.001,
                                  terminal_replace_weight=1.0)),
    ]

    results = {}
    for name, kw in configs:
        print(f"running {name}…", flush=True)
        xf, po, pa, res, t, hf = run_ip4(model, config, gt, y, operator, sigma_n, DEVICE, **kw)
        results[name] = dict(x=xf.to(DEVICE), hf=hf, res=res)

    print(f"\n{'config':<22} {'PSNR_all':>9} {'PSNR_unobs':>11} {'HF':>7} {'|Δ|':>7}")
    print("-"*72)
    for n, v in results.items():
        pa = -10 * torch.log10((((v["x"]+1)/2 - (gt+1)/2) ** 2).mean()).item()
        m_u = 1 - mask
        mse_u = ((m_u * ((v["x"]+1)/2 - (gt+1)/2)) ** 2).sum() / (v["x"].shape[0] * m_u.sum().item() * v["x"].shape[1])
        pu = -10 * torch.log10(mse_u.clamp(min=1e-12)).item()
        d = abs(v["hf"] - gt_hf)
        print(f"{n:<22} {pa:>9.2f} {pu:>11.2f} {v['hf']:>7.3f} {d:>7.3f}")

    # Visualization
    inv = 1 - mask[0, 0].cpu().numpy()
    ys, xs = np.where(inv > 0.5)
    y0, y1 = ys.min(), ys.max()+1; x0, x1 = xs.min(), xs.max()+1

    cols = 3; rows = 3
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    axes = axes.flatten()
    axes[0].imshow(denorm(gt[0])); axes[0].set_title("GT"); axes[0].axis("off")
    for i, (name, _) in enumerate(configs):
        img = denorm(results[name]["x"][0])
        axes[i+1].imshow(img)
        pa = -10 * torch.log10((((results[name]["x"]+1)/2 - (gt+1)/2) ** 2).mean()).item()
        axes[i+1].set_title(f"{name}\nHF={results[name]['hf']:.3f} PSNR={pa:.1f}",
                             fontsize=9)
        axes[i+1].add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0, ec="red", fc="none", lw=2))
        axes[i+1].axis("off")
    for j in range(i+2, len(axes)): axes[j].axis("off")
    plt.suptitle(f"M3 (lambda reduction) vs W1 (h_eps) vs combined — GT HF={gt_hf:.3f}",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{OUT}/m3_vs_w1.png", dpi=140, bbox_inches="tight")
    plt.close()

    # Zoomed box view
    pad = 8
    y0z = max(0, y0-pad); y1z = min(256, y1+pad)
    x0z = max(0, x0-pad); x1z = min(256, x1+pad)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    axes = axes.flatten()
    axes[0].imshow(denorm(gt[0])[y0z:y1z, x0z:x1z])
    axes[0].set_title("GT (box zoom)"); axes[0].axis("off")
    for i, (name, _) in enumerate(configs):
        img = denorm(results[name]["x"][0])[y0z:y1z, x0z:x1z]
        axes[i+1].imshow(img)
        axes[i+1].set_title(f"{name}\nHF={results[name]['hf']:.3f}",
                             fontsize=9)
        axes[i+1].axis("off")
    for j in range(i+2, len(axes)): axes[j].axis("off")
    plt.suptitle("ZOOM to box region: does M3 actually show bird structure?",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{OUT}/m3_vs_w1_zoom.png", dpi=160, bbox_inches="tight")
    plt.close()
    print(f"\nSaved → {OUT}/")


if __name__ == "__main__":
    main()
