"""
Final analysis comparing 5 selected candidates + baseline on BOTH box and
random inpainting. Saves everything under debug_IP4/final_ana/:
  - configs/*.json  — exact settings per candidate
  - box/*.png       — full-image reconstruction PNGs
  - random/*.png    — same for random mask
  - box/comparison.png, random/comparison.png — side-by-side grid
  - metrics.json    — all PSNR (fixed [0,1] convention), HF, res

Candidates listed by user:
  A_baseline               : ms_posterior_sampling_article_version_final defaults (IP3)
  B_G_hx0.1_heps0.0001     : R6 config 5 (global h_eps=0.0001, lam_prox=50, termrep=1)
  C_W1_heps_0.001_F1       : R5/N2 config (global h_eps=0.001, termrep=1)
  D_WINNER3                : WINNER3 (h_eps=0.001 global, h_x stg3=0.7, termrep=1)
  E_N2_plus_W1_hx0.5       : N2 mask-aware-eps + global h_eps=0.001 + h_x stg3=0.5 + termrep=1
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
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_ana")
os.makedirs(OUT, exist_ok=True)
os.makedirs(f"{OUT}/configs", exist_ok=True)
os.makedirs(f"{OUT}/box",    exist_ok=True)
os.makedirs(f"{OUT}/random", exist_ok=True)


def load_gt():
    pt = "trajectory_videos/posterior_sampling/baseline_05/baseline_05_mode-f_x1.pt"
    d = torch.load(pt, map_location="cpu", weights_only=False)
    return d["gt"][:2].to(DEVICE)


def denorm(x):
    return (x*0.5+0.5).clamp(0,1).permute(1,2,0).cpu().numpy()


def psnr_all_01(xf, gt):
    return -10 * torch.log10((((xf+1)/2 - (gt+1)/2) ** 2).mean()).item()


def psnr_region(xf, gt, mask, obs):
    m = mask if obs else (1 - mask)
    n = m.sum().item() * xf.shape[1]
    mse = ((m * ((xf+1)/2 - (gt+1)/2)) ** 2).sum() / (xf.shape[0] * n)
    return -10 * torch.log10(mse.clamp(min=1e-12)).item()


# All 5 candidates + IP3 baseline. A = IP3 default, not the IP4 sampler defaults.
# To reproduce IP3 final we must: lambda_prox=0.0, terminal_replace=0, h_eps=0.01, h_x=0.1.
CANDIDATES = [
    ("A_baseline_IP3final", dict(
        h_x=0.1, h_epsilon=0.01, lambda_prox=0.0,
        terminal_replace_weight=0.0)),
    ("B_G_hx0.1_heps0.0001", dict(
        h_x=0.1, h_epsilon=0.0001, lambda_prox=50.0,
        terminal_replace_weight=1.0)),
    ("C_W1_heps_0.001_F1", dict(
        h_x=0.1, h_epsilon=0.001, lambda_prox=0.0,
        terminal_replace_weight=1.0)),
    ("D_WINNER3", dict(
        h_x=[0.1, 0.1, 0.1, 0.7], h_epsilon=0.001, lambda_prox=0.0,
        terminal_replace_weight=1.0)),
    ("E_N2_plus_W1_hx0.5", dict(
        mask_aware_eps=True, h_epsilon=0.001,
        h_x=[0.1, 0.1, 0.1, 0.5], lambda_prox=0.0,
        terminal_replace_weight=1.0)),
]


def format_cfg_for_json(name, kw):
    """Build a standalone JSON matching ms_posterior_sampling_article_version_final.json
    schema plus the IP4 extra knobs."""
    base = {
        "_comment": f"Candidate: {name}",
        "exp_name":       name,
        "seed":           42,
        "num_stages":     4,
        "resolution":     256,
        "num_examples":   4,
        "sigma_n":        0.05,
        "class_label":    10,
        "shift":          1.0,
        "inference_each_step": 10,
        "guidance_scale": 0.0,
        "num_langevin":   10,
        "h_x":            kw.get("h_x", 0.1),
        "h_epsilon":      kw.get("h_epsilon", 0.01),
        "lambda_x":       0.01,
        "lambda_reg":     50.0,
        "lambda_prox":    kw.get("lambda_prox", 0.0),
        "rho_s":          1.0,
        "rho_e":          1.0,
        "terminal_replace_weight": kw.get("terminal_replace_weight", 0.0),
        "soft_replace_weight":     kw.get("soft_replace_weight", 0.0),
        "dps_kick_zeta":           kw.get("dps_kick_zeta", 0.0),
        "h_x_obs_ratio":           kw.get("h_x_obs_ratio", 1.0),
        "mask_aware_eps":          kw.get("mask_aware_eps", False),
        "lambda_eps_norm":         kw.get("lambda_eps_norm", 0.0),
        "lambda_eps_prox":         kw.get("lambda_eps_prox", 0.0),
        "lambda_eps_inject":       kw.get("lambda_eps_inject", 0.0),
        "skip_eps_update":         kw.get("skip_eps_update", False),
        "reset_eps_per_ode_step":  kw.get("reset_eps_per_ode_step", False),
        "warm_restart":    True,
        "g_bypass_stage3": True,
        "x1_init_mode":    "model",
        "cg_tol":          1e-5,
        "cg_max_iter":     50,
        "device":      "cuda:0",
        "data_dir":    "/data/Zach_dataset/imageNet256/ILSVRC/Data/CLS-LOC/train/",
        "model_dir":   "./pretrained_models/c2img",
        "dict_path":   "./trajectory_videos/posterior_sampling",
        "active_operator":    "box",
        "measurement_mode":   "call",
        "latent_update_mode": "f_x1",
        "box_operator": {
            "mask_type": "box",
            "mask_len_range": [80, 160],
            "mask_prob_range": None},
        "random_operator": {
            "mask_type": "random",
            "mask_len_range": None,
            "mask_prob_range": [0.8, 0.8]},
    }
    return base


def main():
    # save configs
    for name, kw in CANDIDATES:
        cfg = format_cfg_for_json(name, kw)
        with open(f"{OUT}/configs/{name}.json", "w") as f:
            json.dump(cfg, f, indent=2, default=str)

    gt = load_gt()
    config_flow = OmegaConf.load("pretrained_models/c2img/config.yaml")
    model = config_utils.instantiate_from_config(config_flow.model).to(DEVICE)
    ckpt = torch.load("pretrained_models/c2img/model.pt",
                      map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt, strict=True); model.eval()

    sigma_n = 0.05

    all_metrics = {}

    for task_name, op_kw in [
        ("box",    dict(mask_type="box",    mask_len_range=(80, 160), mask_prob_range=None)),
        ("random", dict(mask_type="random", mask_len_range=None,       mask_prob_range=(0.8, 0.8))),
    ]:
        print(f"\n===== {task_name.upper()} inpainting =====", flush=True)
        operator = get_operator("inpainting", resolution=256, device=DEVICE,
                                 sigma=sigma_n, **op_kw)
        y = operator(gt).detach()
        mask = operator.get_mask(x=gt).float().to(DEVICE)
        if mask.shape[0] == 1 and gt.shape[0] > 1:
            mask_bcast = mask.expand(gt.shape[0], -1, -1, -1)
        else:
            mask_bcast = mask
        gt_hf = hf_energy(gt * (1 - mask[0:1]))
        print(f"GT HF_unobs = {gt_hf:.3f}", flush=True)

        metrics = {}
        images  = {}

        print(f"{'candidate':<30} {'PSNR_all':>9} {'PSNR_obs':>9} {'PSNR_unobs':>11} {'HF':>7} {'|Δ|':>7} {'res':>7}", flush=True)
        print("-"*90, flush=True)
        for name, kw in CANDIDATES:
            try:
                xf, po_raw, pa_raw, res, t, hf = run_ip4(
                    model, config_flow, gt, y, operator, sigma_n, DEVICE, **kw)
                xfD = xf.to(DEVICE)
                pa  = psnr_all_01(xfD, gt)
                po  = psnr_region(xfD, gt, mask_bcast, obs=True)
                pu  = psnr_region(xfD, gt, mask_bcast, obs=False)
                metrics[name] = dict(psnr_all=pa, psnr_obs=po, psnr_unobs=pu,
                                      hf=hf, hf_delta=abs(hf - gt_hf),
                                      res=res, time=t)
                images[name] = xfD
                print(f"{name:<30} {pa:>9.2f} {po:>9.2f} {pu:>11.2f} {hf:>7.3f} {abs(hf-gt_hf):>7.3f} {res:>7.1f}", flush=True)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"{name} ERROR: {e}", flush=True)

        # save individual PNGs
        subdir = f"{OUT}/{task_name}"
        plt.imsave(f"{subdir}/GT.png", denorm(gt[0]))
        for name, img in images.items():
            plt.imsave(f"{subdir}/{name}.png", denorm(img[0]))

        # comparison grid
        n = len(images) + 1
        cols = 3; rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols*4.5, rows*4.5))
        axes = axes.flatten()
        inv = 1 - mask[0, 0].cpu().numpy()
        ys, xs = np.where(inv > 0.5)
        if len(ys):
            y0, y1 = ys.min(), ys.max()+1; x0, x1 = xs.min(), xs.max()+1

        axes[0].imshow(denorm(gt[0])); axes[0].set_title(f"GT  (HF={gt_hf:.3f})", fontsize=11)
        axes[0].axis("off")
        for i, (name, img) in enumerate(images.items()):
            m = metrics[name]
            axes[i+1].imshow(denorm(img[0]))
            axes[i+1].set_title(
                f"{name}\nPSNRall={m['psnr_all']:.2f}  PSNRunobs={m['psnr_unobs']:.2f}\n"
                f"HF={m['hf']:.3f} |Δ|={m['hf_delta']:.3f}  res={m['res']:.0f}",
                fontsize=9)
            if task_name == "box" and len(ys):
                axes[i+1].add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0,
                                                    ec="red", fc="none", lw=2))
            axes[i+1].axis("off")
        for j in range(i+2, len(axes)): axes[j].axis("off")
        plt.suptitle(f"FINAL COMPARISON — {task_name.upper()} inpainting  (GT HF={gt_hf:.3f})",
                     fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{subdir}/comparison.png", dpi=140, bbox_inches="tight")
        plt.close()

        all_metrics[task_name] = dict(gt_hf=gt_hf, per_config=metrics)

    with open(f"{OUT}/metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"\nAll saved under {OUT}/", flush=True)


if __name__ == "__main__":
    main()
