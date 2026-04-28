#!/usr/bin/env python
"""
IP4 Round-4 — tests the missing pseudo-code term.

The paper's Algorithm step (d) has three terms in g_x1:
    (1) (1/η²) A^T (b - A x1)          -- likelihood
    (2) (H_τ)^T ŝ_flow                 -- prior score
    (3) λ (sg(x̂_1) - x1)               -- proximal pull (MISSING in final_utils)

All IP4 rounds 1-3 ran with term (3) dropped. Round-4 re-enables it via
`lambda_prox` (we decouple from lambda_reg so we can test independently).

Also re-tests the round-3 winners (X4, CB1, F1) with term (3) enabled,
to see if restoring the paper algorithm further improves perceptual.
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

DEVICE = "cuda:0"   # round-4 on GPU 0 (others idle)
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_round4")
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
    print(f"Round-4 on {DEVICE}", flush=True)
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
        ("R4_base",       dict()),
        ("R4_F1",         dict(terminal_replace_weight=1.0)),

        # ═══ L. Proximal-term-only sweep (restore paper algorithm) ═══
        ("L1_prox_1",     dict(lambda_prox=1.0,  terminal_replace_weight=1.0)),
        ("L2_prox_5",     dict(lambda_prox=5.0,  terminal_replace_weight=1.0)),
        ("L3_prox_10",    dict(lambda_prox=10.0, terminal_replace_weight=1.0)),
        ("L4_prox_50",    dict(lambda_prox=50.0, terminal_replace_weight=1.0)),
        ("L5_prox_100",   dict(lambda_prox=100.0, terminal_replace_weight=1.0)),

        # ═══ M. Proximal stage-3 only (mirror the h_x finding) ═══
        ("M1_prox_stg3_1",  dict(lambda_prox=[0,0,0,1.0],  terminal_replace_weight=1.0)),
        ("M2_prox_stg3_10", dict(lambda_prox=[0,0,0,10.0], terminal_replace_weight=1.0)),
        ("M3_prox_stg3_50", dict(lambda_prox=[0,0,0,50.0], terminal_replace_weight=1.0)),

        # ═══ N. Proximal decreases as stage progresses (weight model early, Langevin late) ═══
        ("N1_prox_decr_10to0", dict(lambda_prox=[10.0, 5.0, 1.0, 0.0], terminal_replace_weight=1.0)),
        ("N2_prox_decr_50to0", dict(lambda_prox=[50.0, 10.0, 1.0, 0.0], terminal_replace_weight=1.0)),

        # ═══ O. X4 + proximal (does adding proximal help X4?) ═══
        ("O1_X4_prox1",   dict(h_x=[0.1,0.1,0.1,0.7], lambda_prox=1.0,  terminal_replace_weight=1.0)),
        ("O2_X4_prox10",  dict(h_x=[0.1,0.1,0.1,0.7], lambda_prox=10.0, terminal_replace_weight=1.0)),
        ("O3_X4_prox50",  dict(h_x=[0.1,0.1,0.1,0.7], lambda_prox=50.0, terminal_replace_weight=1.0)),

        # ═══ PR. CB1 + proximal ═══
        ("PR1_CB1_prox1",  dict(h_x=[0.1,0.1,0.1,0.3], lambda_reg=[50,50,50,20], lambda_prox=1.0,  terminal_replace_weight=1.0)),
        ("PR2_CB1_prox10", dict(h_x=[0.1,0.1,0.1,0.3], lambda_reg=[50,50,50,20], lambda_prox=10.0, terminal_replace_weight=1.0)),
        ("PR3_CB1_prox50", dict(h_x=[0.1,0.1,0.1,0.3], lambda_reg=[50,50,50,20], lambda_prox=50.0, terminal_replace_weight=1.0)),

        # ═══ Q. Paper default: lambda_prox = lambda_reg (as written in pseudo-code) ═══
        ("Q1_paper_prox_eq_lamreg_50", dict(lambda_prox=50.0, lambda_reg=50.0, terminal_replace_weight=1.0)),
        ("Q2_paper_X4_prox50",         dict(h_x=[0.1,0.1,0.1,0.7], lambda_prox=50.0, terminal_replace_weight=1.0)),
    ]

    results = {}
    header = "{:<3} {:<38} {:>5} {:>7} {:>7} {:>6} {:>5}"
    print(header.format("#","name","res","psnrO","psnrAll","HF","t(s)"), flush=True)
    print("-"*82, flush=True)
    for i, (name, kw) in enumerate(configs, 1):
        try:
            xf, po, pa, res, t, hf = run_ip4(model, config, gt, y, operator, sigma_n, DEVICE, **kw)
            results[name] = dict(x=xf, psnr_obs=po, psnr_all=pa, res=res, time=t, hf=hf, kw=kw)
            print(header.format(i,name,f"{res:.0f}",f"{po:.2f}",f"{pa:.2f}",
                  f"{hf:.3f}",f"{t:.0f}"), flush=True)
        except Exception as e:
            print(f"{i:<3} {name:<38} ERROR: {e}", flush=True)

    summary = {k: {kk: vv for kk,vv in v.items() if kk!="x"} for k, v in results.items()}
    for k in summary:
        summary[k]["kw"] = {kk:vv for kk,vv in summary[k]["kw"].items()}
    with open(f"{RESULT_DIR}/round4_results.json", "w") as f:
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
        print(f"{i:<3} {n:<38} HF={v['hf']:.3f} |Δ|={d:.3f} PSNR={v['psnr_all']:.2f}", flush=True)

    keys = [k for k,_ in by_psnr]
    cols = min(7, len(keys)+1); rows = (len(keys)+1+cols-1)//cols
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
    plt.suptitle(f"IP4 round-4 (proximal term) GT HF={gt_hf:.3f}", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/round4.png", dpi=150)
    plt.close()
    print(f"\nSaved {RESULT_DIR}/round4.png", flush=True)

if __name__ == "__main__":
    main()
