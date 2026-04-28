#!/usr/bin/env python
"""
IP4 Round-2 — launched on GPU 1 in parallel with Round-1 on GPU 0.

Round-1 finding so far: stage-3 lambda_reg=10 (B3) is the sweet spot,
HF climbs from 0.655 to 0.693 (GT=0.702) at cost of ~0.6 dB PSNR.
CFG axis is completely dead for this model.

Round-2 agenda (all within the pseudo-code framework):
  Z. fine lambda_reg sweep at stage 3: {5, 7, 10, 12, 15, 20, 30}
  Y. release lambda at stage 2 in addition to stage 3
  X. step-wise lambda within stage 3 (only release last N ODE steps)  [impl in sampler]
  W. h_epsilon sweep (NEVER tested in IP3 or IP4-round1)
  V. per-term ablation — zero out likelihood / prior / eps-update one at a time
  U. noise_scale schedules paired with lambda release
  T. DPS zeta high (20, 50) — IP3 z=10 hit HF=0.634 near GT_HF
  S. h_x stage-3 amplified paired with lambda=10
  R. num_langevin stage-3 amplified paired with lambda=10
  Q. combined B3 + soft_replace sweep
  P. combined B3 + terminal_replace sweep
"""
import os, sys, json, time, copy
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

DEVICE = "cuda:1"   # round-2 on GPU 1 (round-1 on GPU 0)
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_round2")
os.makedirs(RESULT_DIR, exist_ok=True)


def load_gt():
    pt = "trajectory_videos/posterior_sampling/baseline_05/baseline_05_mode-f_x1.pt"
    if os.path.exists(pt):
        d = torch.load(pt, map_location="cpu", weights_only=False)
        return d["gt"][:2].to(DEVICE)
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
    return torch.stack([ds[ci[i]][0] for i in perm.tolist()]).to(DEVICE)


def main():
    print(f"Round-2 on {DEVICE}", flush=True)
    gt = load_gt()
    config = OmegaConf.load("pretrained_models/c2img/config.yaml")
    model = config_utils.instantiate_from_config(config.model).to(DEVICE)
    ckpt = torch.load("pretrained_models/c2img/model.pt",
                      map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt, strict=True); model.eval()

    sigma_n = 0.05
    operator = get_operator(
        "inpainting", resolution=256, device=DEVICE, sigma=sigma_n,
        mask_type="box", mask_len_range=(80, 160), mask_prob_range=None,
    )
    y = operator(gt).detach()
    mask = operator.get_mask(x=gt).float().to(DEVICE)
    gt_hf = hf_energy(gt * (1 - mask[0:1]))
    print(f"BOX  GT HF_unobs={gt_hf:.3f}", flush=True)

    configs = [
        # ═══ Z. fine lambda_reg sweep at stage 3 ═══
        ("Z1_lam_stg3_5",   dict(lambda_reg=[50, 50, 50, 5])),
        ("Z2_lam_stg3_7",   dict(lambda_reg=[50, 50, 50, 7])),
        ("Z3_lam_stg3_10",  dict(lambda_reg=[50, 50, 50, 10])),
        ("Z4_lam_stg3_12",  dict(lambda_reg=[50, 50, 50, 12])),
        ("Z5_lam_stg3_15",  dict(lambda_reg=[50, 50, 50, 15])),
        ("Z6_lam_stg3_20",  dict(lambda_reg=[50, 50, 50, 20])),
        ("Z7_lam_stg3_30",  dict(lambda_reg=[50, 50, 50, 30])),

        # ═══ Y. release stages 2 & 3 together ═══
        ("Y1_lam_stg23_20_10", dict(lambda_reg=[50, 50, 20, 10])),
        ("Y2_lam_stg23_10_10", dict(lambda_reg=[50, 50, 10, 10])),
        ("Y3_lam_stg23_10_5",  dict(lambda_reg=[50, 50, 10, 5])),

        # ═══ W. h_epsilon sweep — entirely unexplored axis ═══
        ("W1_heps_0.001", dict(h_epsilon=0.001)),
        ("W2_heps_0.05",  dict(h_epsilon=0.05)),
        ("W3_heps_0.1",   dict(h_epsilon=0.1)),

        # ═══ T. DPS zeta aggressive (IP3 z=10 got HF~GT) ═══
        ("T1_dps_stg3_z10", dict(dps_kick_zeta=[0, 0, 0, 10])),
        ("T2_dps_stg3_z20", dict(dps_kick_zeta=[0, 0, 0, 20])),
        ("T3_dps_stg3_z50", dict(dps_kick_zeta=[0, 0, 0, 50])),
        ("T4_dps_all_z10",  dict(dps_kick_zeta=10.0)),

        # ═══ S. h_x amplified at stage 3 paired with lam=10 ═══
        ("S1_hx0.3_lam10",  dict(h_x=[0.1, 0.1, 0.1, 0.3], lambda_reg=[50, 50, 50, 10])),
        ("S2_hx0.5_lam10",  dict(h_x=[0.1, 0.1, 0.1, 0.5], lambda_reg=[50, 50, 50, 10])),
        ("S3_hx1.0_lam10",  dict(h_x=[0.1, 0.1, 0.1, 1.0], lambda_reg=[50, 50, 50, 10])),

        # ═══ R. num_langevin at stage 3 paired with lam=10 ═══
        ("R1_L20_lam10",   dict(num_langevin=[10, 10, 10, 20], lambda_reg=[50, 50, 50, 10])),
        ("R2_L3_lam10",    dict(num_langevin=[10, 10, 10, 3],  lambda_reg=[50, 50, 50, 10])),

        # ═══ Q. B3 + soft_replace at stage 3 ═══
        ("Q1_B3_softrep0.1", dict(lambda_reg=[50, 50, 50, 10], soft_replace_weight=0.1)),
        ("Q2_B3_softrep0.3", dict(lambda_reg=[50, 50, 50, 10], soft_replace_weight=0.3)),

        # ═══ P. B3 + terminal_replace ═══
        ("P1_B3_term1.0",  dict(lambda_reg=[50, 50, 50, 10], terminal_replace_weight=1.0)),
        ("P2_B3_term0.5",  dict(lambda_reg=[50, 50, 50, 10], terminal_replace_weight=0.5)),

        # ═══ U. B3 + late-stage noise injection ═══
        ("U1_B3_noise0.2", dict(lambda_reg=[50, 50, 50, 10], noise_scale=[0, 0, 0, 0.2])),
        ("U2_B3_noise0.5", dict(lambda_reg=[50, 50, 50, 10], noise_scale=[0, 0, 0, 0.5])),

        # ═══ Ultra-combo candidates ═══
        ("COMBO1_B3+dpsZ10+term1",
         dict(lambda_reg=[50, 50, 50, 10], dps_kick_zeta=[0, 0, 0, 10],
              terminal_replace_weight=1.0)),
        ("COMBO2_Y1+term1",
         dict(lambda_reg=[50, 50, 20, 10], terminal_replace_weight=1.0)),
        ("COMBO3_B3+hx0.3+term1",
         dict(lambda_reg=[50, 50, 50, 10], h_x=[0.1, 0.1, 0.1, 0.3],
              terminal_replace_weight=1.0)),
        ("COMBO4_B3+L3+term1",
         dict(lambda_reg=[50, 50, 50, 10], num_langevin=[10, 10, 10, 3],
              terminal_replace_weight=1.0)),
    ]

    results = {}
    header = "{:<3} {:<32} {:>5} {:>7} {:>7} {:>6} {:>5}"
    print(header.format("#", "name", "res", "psnrO", "psnrAll", "HF", "t(s)"), flush=True)
    print("-"*80, flush=True)
    for i, (name, kw) in enumerate(configs, 1):
        try:
            xf, po, pa, res, t, hf = run_ip4(model, config, gt, y, operator, sigma_n, DEVICE, **kw)
            results[name] = dict(x=xf, psnr_obs=po, psnr_all=pa, res=res, time=t, hf=hf, kw=kw)
            print(header.format(i, name, f"{res:.0f}", f"{po:.2f}", f"{pa:.2f}",
                                f"{hf:.3f}", f"{t:.0f}"), flush=True)
        except Exception as e:
            print(f"{i:<3} {name:<32} ERROR: {e}", flush=True)

    # Save summary
    summary = {k: {kk: vv for kk, vv in v.items() if kk != "x"} for k, v in results.items()}
    for k in summary:
        summary[k]["kw"] = {kk: vv for kk, vv in summary[k]["kw"].items()}
    with open(f"{RESULT_DIR}/round2_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # ranked
    by_psnr = sorted(results.items(), key=lambda kv: kv[1]["psnr_all"], reverse=True)
    by_hf   = sorted(results.items(), key=lambda kv: abs(kv[1]["hf"] - gt_hf))
    print(f"\n===== RANKED BY PSNR_all =====", flush=True)
    for i, (n, v) in enumerate(by_psnr, 1):
        print(header.format(i, n, f"{v['res']:.0f}", f"{v['psnr_obs']:.2f}",
              f"{v['psnr_all']:.2f}", f"{v['hf']:.3f}", f"{v['time']:.0f}"), flush=True)
    print(f"\n===== RANKED BY HF proximity (GT={gt_hf:.3f}) =====", flush=True)
    for i, (n, v) in enumerate(by_hf, 1):
        d = abs(v["hf"] - gt_hf)
        print(f"{i:<3} {n:<32} HF={v['hf']:.3f} |Δ|={d:.3f} PSNR={v['psnr_all']:.2f}", flush=True)

    # viz
    keys = [k for k, _ in by_psnr]
    cols = min(8, len(keys) + 1); rows = (len(keys)+1+cols-1)//cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.5, rows*2.5))
    if rows == 1: axes = axes[np.newaxis, :]
    axes = axes.flatten()
    img_gt = (gt[0].cpu().permute(1,2,0)*0.5+0.5).clamp(0,1).numpy()
    axes[0].imshow(img_gt); axes[0].set_title("GT", fontsize=6); axes[0].axis("off")
    for i, k in enumerate(keys):
        img = (results[k]["x"][0].permute(1,2,0)*0.5+0.5).clamp(0,1).numpy()
        axes[i+1].imshow(img)
        axes[i+1].set_title(f"{k}\n{results[k]['psnr_all']:.2f}dB hf={results[k]['hf']:.3f}", fontsize=5)
        axes[i+1].axis("off")
    for j in range(i+2, len(axes)): axes[j].axis("off")
    plt.suptitle(f"IP4 round-2 (GT HF={gt_hf:.3f})", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/round2.png", dpi=150)
    plt.close()
    print(f"Saved {RESULT_DIR}/round2.png", flush=True)

if __name__ == "__main__":
    main()
