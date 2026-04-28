"""Re-run the distilled TOP configurations from all rounds on box AND random.
Saves to final_ana/top_configs/{box,random}/ with corrected PSNR + visual grids."""
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
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_ana", "top_configs")
os.makedirs(OUT, exist_ok=True)
os.makedirs(f"{OUT}/box",    exist_ok=True)
os.makedirs(f"{OUT}/random", exist_ok=True)
os.makedirs(f"{OUT}/configs", exist_ok=True)


def load_gt():
    pt = "trajectory_videos/posterior_sampling/baseline_05/baseline_05_mode-f_x1.pt"
    d = torch.load(pt, map_location="cpu", weights_only=False)
    return d["gt"][:2].to(DEVICE)


def denorm(x):
    return (x*0.5+0.5).clamp(0,1).permute(1,2,0).cpu().numpy()


# ═══════════════════════════════════════════════════════════════════════
# TOP configs: distilled from 220+ experiments, visually verified
# ═══════════════════════════════════════════════════════════════════════
# ID scheme: 0 = ref, 1+ = ordered by increasing perceptual strength
TOP_CONFIGS = [
    ("0_ref_baseline_F1", {
        "rationale": "Reference: no eps tempering, only terminal replacement (free data consistency)",
        "kw": dict(terminal_replace_weight=1.0),
    }),
    ("1_S1_stage1_only_F1", {
        "rationale": "Minimal perturbation: low h_eps only at stage 1 (most critical per-stage). Nearly baseline PSNR, mild perceptual shift.",
        "kw": dict(h_epsilon=[0.01, 0.001, 0.01, 0.01], terminal_replace_weight=1.0),
    }),
    ("2_P12_stages12_F1", {
        "rationale": "Balanced: stages 1+2 LO. PSNR cost ~0.8 dB, HF shifts toward GT.",
        "kw": dict(h_epsilon=[0.01, 0.001, 0.001, 0.01], terminal_replace_weight=1.0),
    }),
    ("3_T012_LO_LO_LO_HI_F1", {
        "rationale": "W1-equivalent perceptual, 25% faster (stage-3 stays HI since it's free to skip). Best 'free efficiency' variant.",
        "kw": dict(h_epsilon=[0.001, 0.001, 0.001, 0.01], terminal_replace_weight=1.0),
    }),
    ("4_W1_global_heps_0.001_F1", {
        "rationale": "Classic W1: global low h_eps. Strong semantic bird in box. PSNR -3 to -4 dB.",
        "kw": dict(h_epsilon=0.001, terminal_replace_weight=1.0),
    }),
    ("5_WINNER3_W1+X4+F1", {
        "rationale": "W1 + h_x stage-3 boost. Sharpest bird texture. PSNR -4 dB. Most detail, sometimes over-sharp.",
        "kw": dict(h_epsilon=0.001, h_x=[0.1, 0.1, 0.1, 0.7], terminal_replace_weight=1.0),
    }),
    ("6_V_all_1e-4_F1", {
        "rationale": "Extreme: h_eps=1e-4 globally. HF bullseye, most grain. Highest texture, lowest PSNR.",
        "kw": dict(h_epsilon=0.0001, terminal_replace_weight=1.0),
    }),
]


def psnr_region(xf, gt, mask, obs):
    m = mask if obs else (1 - mask)
    n = m.sum().item() * xf.shape[1]
    mse = ((m * ((xf+1)/2 - (gt+1)/2)) ** 2).sum() / (xf.shape[0] * max(n, 1))
    return (-10 * torch.log10(mse.clamp(min=1e-12))).item()


def psnr_all_01(xf, gt):
    return -10 * torch.log10((((xf+1)/2 - (gt+1)/2) ** 2).mean()).item()


def main():
    gt = load_gt()
    config = OmegaConf.load("pretrained_models/c2img/config.yaml")
    model = config_utils.instantiate_from_config(config.model).to(DEVICE)
    ckpt = torch.load("pretrained_models/c2img/model.pt",
                      map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt, strict=True); model.eval()

    # Save config JSONs
    for name, info in TOP_CONFIGS:
        with open(f"{OUT}/configs/{name}.json", "w") as f:
            json.dump({"name": name, **info}, f, indent=2, default=str)

    sigma_n = 0.05
    all_metrics = {}

    for task_name, op_kw in [
        ("box",    dict(mask_type="box",    mask_len_range=(80, 160), mask_prob_range=None)),
        ("random", dict(mask_type="random", mask_len_range=None,       mask_prob_range=(0.8, 0.8))),
    ]:
        print(f"\n===== {task_name.upper()} =====", flush=True)
        operator = get_operator("inpainting", resolution=256, device=DEVICE,
                                 sigma=sigma_n, **op_kw)
        y = operator(gt).detach()
        mask = operator.get_mask(x=gt).float().to(DEVICE)
        if mask.shape[0] == 1 and gt.shape[0] > 1:
            mask_b = mask.expand(gt.shape[0], -1, -1, -1)
        else:
            mask_b = mask
        gt_hf = hf_energy(gt * (1 - mask[0:1]))
        print(f"GT HF = {gt_hf:.3f}", flush=True)

        metrics = {}
        images = {}
        print(f"{'name':<34} {'PSNR_all':>9} {'PSNR_obs':>9} {'PSNR_unobs':>11} {'HF':>7} {'|Δ|':>7} {'res':>7}", flush=True)
        print("-"*90, flush=True)

        for name, info in TOP_CONFIGS:
            kw = info["kw"]
            try:
                xf, po, pa, res, t, hf = run_ip4(model, config, gt, y, operator, sigma_n, DEVICE, **kw)
                xfD = xf.to(DEVICE)
                pa_01 = psnr_all_01(xfD, gt)
                po_fix = psnr_region(xfD, gt, mask_b, True)
                pu_fix = psnr_region(xfD, gt, mask_b, False)
                metrics[name] = dict(psnr_all=pa_01, psnr_obs=po_fix, psnr_unobs=pu_fix,
                                      hf=hf, hf_delta=abs(hf - gt_hf), res=res)
                images[name] = xfD
                print(f"{name:<34} {pa_01:>9.2f} {po_fix:>9.2f} {pu_fix:>11.2f} {hf:>7.3f} {abs(hf-gt_hf):>7.3f} {res:>7.1f}",
                      flush=True)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"{name} ERROR: {e}", flush=True)

        # save individual PNGs
        subdir = f"{OUT}/{task_name}"
        plt.imsave(f"{subdir}/GT.png", denorm(gt[0]))
        for name, img in images.items():
            plt.imsave(f"{subdir}/{name}.png", denorm(img[0]))

        # grid
        n = len(images) + 1
        cols = 4; rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
        axes = axes.flatten()
        inv = 1 - mask[0, 0].cpu().numpy()
        ys, xs = np.where(inv > 0.5)
        if len(ys) and task_name == "box":
            y0, y1 = ys.min(), ys.max()+1; x0, x1 = xs.min(), xs.max()+1

        axes[0].imshow(denorm(gt[0])); axes[0].set_title(f"GT  (HF={gt_hf:.3f})", fontsize=10)
        axes[0].axis("off")
        for i, (name, _) in enumerate(TOP_CONFIGS):
            img = images[name]; m = metrics[name]
            axes[i+1].imshow(denorm(img[0]))
            axes[i+1].set_title(
                f"{name}\nPSNR={m['psnr_all']:.2f}  unobs={m['psnr_unobs']:.2f}\n"
                f"HF={m['hf']:.3f}  |Δ|={m['hf_delta']:.3f}",
                fontsize=8)
            if task_name == "box" and len(ys):
                axes[i+1].add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0,
                                                    ec="red", fc="none", lw=2))
            axes[i+1].axis("off")
        for j in range(i+2, len(axes)): axes[j].axis("off")
        plt.suptitle(f"TOP configs — {task_name.upper()} inpainting  (GT HF={gt_hf:.3f})", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{subdir}/comparison.png", dpi=140, bbox_inches="tight")
        plt.close()

        all_metrics[task_name] = dict(gt_hf=gt_hf, per_config=metrics)

    with open(f"{OUT}/metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    print(f"\nAll saved under {OUT}/", flush=True)


if __name__ == "__main__":
    main()
