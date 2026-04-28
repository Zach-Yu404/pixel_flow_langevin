#!/usr/bin/env python
"""
IP4 Round-3 — targeted combos of confirmed winners.

Confirmed axes from R1/R2:
  A (CFG)   : DEAD for this model
  B/Z (lambda stage-3): perceptual++ at cost of PSNR (Z2-Z3 / B2-B3)
  C (h_x stage-3)     : +perceptual, -0.4 dB PSNR, res↓ 4x — STRONGEST direction
  D (fewer Langevin)  : hurts res, don't
  E (DPS kick)        : free res=0 w/o perceptual change
  F1 (terminal replace): FREE res=0, psnr_obs=inf, same PSNR/HF
  → best recipe hypothesis: C + F1 (never tested directly)

Round-3 tests:
  1. C2 × F1 : h_x stage-3 ∈ {0.2, 0.3, 0.5, 0.7, 1.0} × termrep {0, 1}
  2. C × B combined: h_x stage-3 boost + lambda stage-3 reduction together
  3. h_x sweep at finer granularity
  4. h_x also applied to stage-2 (non-monotonic curve)
  5. terminal-replace applied to every lambda-reduction config
  6. bulk loss-term ablation: h_x=0 (pure CG/likelihood), lambda_reg very small
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

DEVICE = "cuda:2"
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_round3")
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
    print(f"Round-3 on {DEVICE}", flush=True)
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
        # Reference
        ("R3_base", dict()),
        ("R3_F1_termrep1", dict(terminal_replace_weight=1.0)),

        # ═══ Primary hypothesis: C × F1 ═══
        ("X1_hx0.2_term1", dict(h_x=[0.1,0.1,0.1,0.2], terminal_replace_weight=1.0)),
        ("X2_hx0.3_term1", dict(h_x=[0.1,0.1,0.1,0.3], terminal_replace_weight=1.0)),
        ("X3_hx0.5_term1", dict(h_x=[0.1,0.1,0.1,0.5], terminal_replace_weight=1.0)),
        ("X4_hx0.7_term1", dict(h_x=[0.1,0.1,0.1,0.7], terminal_replace_weight=1.0)),
        ("X5_hx1.0_term1", dict(h_x=[0.1,0.1,0.1,1.0], terminal_replace_weight=1.0)),

        # ═══ C × B combined (h_x stg3 boost + lambda_reg stg3 drop) ═══
        ("CB1_hx0.3_lam20_term1",  dict(h_x=[0.1,0.1,0.1,0.3], lambda_reg=[50,50,50,20], terminal_replace_weight=1.0)),
        ("CB2_hx0.3_lam10_term1",  dict(h_x=[0.1,0.1,0.1,0.3], lambda_reg=[50,50,50,10], terminal_replace_weight=1.0)),
        ("CB3_hx0.5_lam20_term1",  dict(h_x=[0.1,0.1,0.1,0.5], lambda_reg=[50,50,50,20], terminal_replace_weight=1.0)),
        ("CB4_hx0.5_lam10_term1",  dict(h_x=[0.1,0.1,0.1,0.5], lambda_reg=[50,50,50,10], terminal_replace_weight=1.0)),

        # ═══ h_x both stages 2 and 3 ═══
        ("HX1_stg23_0.3_term1",    dict(h_x=[0.1,0.1,0.3,0.3], terminal_replace_weight=1.0)),
        ("HX2_stg23_0.2_0.5_term1",dict(h_x=[0.1,0.1,0.2,0.5], terminal_replace_weight=1.0)),

        # ═══ h_x with higher num_langevin to amplify effect ═══
        ("HXL1_hx0.3_L15_term1",   dict(h_x=[0.1,0.1,0.1,0.3], num_langevin=[10,10,10,15], terminal_replace_weight=1.0)),
        ("HXL2_hx0.5_L15_term1",   dict(h_x=[0.1,0.1,0.1,0.5], num_langevin=[10,10,10,15], terminal_replace_weight=1.0)),

        # ═══ Terminal replace + every B variant (previously untested with termrep) ═══
        ("BT1_lam_stg3_7_term1",   dict(lambda_reg=[50,50,50,7],  terminal_replace_weight=1.0)),
        ("BT2_lam_stg3_10_term1",  dict(lambda_reg=[50,50,50,10], terminal_replace_weight=1.0)),
        ("BT3_lam_stg3_15_term1",  dict(lambda_reg=[50,50,50,15], terminal_replace_weight=1.0)),

        # ═══ Loss-term bulk ablation ═══
        # very small h_x — prior/model dominates, no Langevin update
        ("AB1_hx0.01_term1",       dict(h_x=0.01, terminal_replace_weight=1.0)),
        # very high h_x stage 3 — likelihood domination
        ("AB2_hx2.0_stg3_term1",   dict(h_x=[0.1,0.1,0.1,2.0], terminal_replace_weight=1.0)),
        # eps update step-size sweep
        ("AB3_heps0.1_term1",      dict(h_epsilon=0.1, terminal_replace_weight=1.0)),

        # ═══ Ultra-combo: C + B + DPS + termrep ═══
        ("ULTRA1_hx0.3_lam10_dpsZ10_term1",
         dict(h_x=[0.1,0.1,0.1,0.3], lambda_reg=[50,50,50,10],
              dps_kick_zeta=[0,0,0,10], terminal_replace_weight=1.0)),
        ("ULTRA2_hx0.5_lam10_dpsZ10_term1",
         dict(h_x=[0.1,0.1,0.1,0.5], lambda_reg=[50,50,50,10],
              dps_kick_zeta=[0,0,0,10], terminal_replace_weight=1.0)),
        ("ULTRA3_hx0.5_noise0.3_term1",
         dict(h_x=[0.1,0.1,0.1,0.5], noise_scale=[0,0,0,0.3],
              terminal_replace_weight=1.0)),
        ("ULTRA4_hx0.7_noise0.2_lam15_term1",
         dict(h_x=[0.1,0.1,0.1,0.7], noise_scale=[0,0,0,0.2],
              lambda_reg=[50,50,50,15], terminal_replace_weight=1.0)),
    ]

    results = {}
    header = "{:<3} {:<40} {:>5} {:>7} {:>7} {:>6} {:>5}"
    print(header.format("#","name","res","psnrO","psnrAll","HF","t(s)"), flush=True)
    print("-"*90, flush=True)
    for i, (name, kw) in enumerate(configs, 1):
        try:
            xf, po, pa, res, t, hf = run_ip4(model, config, gt, y, operator, sigma_n, DEVICE, **kw)
            results[name] = dict(x=xf, psnr_obs=po, psnr_all=pa, res=res, time=t, hf=hf, kw=kw)
            print(header.format(i,name,f"{res:.0f}",f"{po:.2f}",f"{pa:.2f}",
                  f"{hf:.3f}",f"{t:.0f}"), flush=True)
        except Exception as e:
            print(f"{i:<3} {name:<40} ERROR: {e}", flush=True)

    summary = {k: {kk: vv for kk,vv in v.items() if kk!="x"} for k, v in results.items()}
    for k in summary:
        summary[k]["kw"] = {kk:vv for kk,vv in summary[k]["kw"].items()}
    with open(f"{RESULT_DIR}/round3_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    by_psnr = sorted(results.items(), key=lambda kv: kv[1]["psnr_all"], reverse=True)
    by_hf   = sorted(results.items(), key=lambda kv: abs(kv[1]["hf"] - gt_hf))
    print(f"\n===== RANKED BY PSNR_all =====", flush=True)
    for i, (n, v) in enumerate(by_psnr, 1):
        print(header.format(i,n,f"{v['res']:.0f}",f"{v['psnr_obs']:.2f}",
              f"{v['psnr_all']:.2f}",f"{v['hf']:.3f}",f"{v['time']:.0f}"), flush=True)
    print(f"\n===== RANKED BY HF proximity (GT={gt_hf:.3f}) =====", flush=True)
    for i, (n, v) in enumerate(by_hf, 1):
        d = abs(v["hf"] - gt_hf)
        print(f"{i:<3} {n:<40} HF={v['hf']:.3f} |Δ|={d:.3f} PSNR={v['psnr_all']:.2f}", flush=True)

    keys = [k for k,_ in by_psnr]
    cols = min(8, len(keys)+1); rows = (len(keys)+1+cols-1)//cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.6, rows*2.6))
    if rows == 1: axes = axes[np.newaxis,:]
    axes = axes.flatten()
    img_gt = (gt[0].cpu().permute(1,2,0)*0.5+0.5).clamp(0,1).numpy()
    axes[0].imshow(img_gt); axes[0].set_title("GT", fontsize=6); axes[0].axis("off")
    for i, k in enumerate(keys):
        img = (results[k]["x"][0].permute(1,2,0)*0.5+0.5).clamp(0,1).numpy()
        axes[i+1].imshow(img)
        axes[i+1].set_title(f"{k}\n{results[k]['psnr_all']:.2f}dB hf={results[k]['hf']:.3f}", fontsize=5)
        axes[i+1].axis("off")
    for j in range(i+2, len(axes)): axes[j].axis("off")
    plt.suptitle(f"IP4 round-3 (GT HF={gt_hf:.3f})", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/round3.png", dpi=150)
    plt.close()
    print(f"\nSaved {RESULT_DIR}/round3.png", flush=True)

if __name__ == "__main__":
    main()
