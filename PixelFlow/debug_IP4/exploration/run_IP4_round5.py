#!/usr/bin/env python
"""
IP4 Round-5 — NO terminal_replace. Pure sampling-dynamics exploration.

User request: "不开这个进行探索"

Without terminal_replace (the post-hoc data-consistency projection), the
observed pixels are left to whatever the Langevin inner loop produces.
Question: can we reach good PSNR AND good perceptual *from sampling alone*?

Axes tested:
  L. Fine h_x stage-3 sweep               — pure perceptual lever
  M. Fine lambda_reg stage-3 sweep        — pure perceptual lever #2
  N. h_x + lambda_prox (restore paper)    — does the missing term help when
                                             we don't have termrep safety net?
  O. num_langevin scaling                 — more steps might self-consistency
  P. CG tightening                        — cg_tol=1e-7, cg_max_iter=100
  Q. soft_replace per-step variants       — mild in-loop projection
  R. Combined: h_x + lambda_prox without termrep
  S. Ultra-fine around R1 best (C2 h_x=0.5 + lambda stg3=20 no term)
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

DEVICE = "cuda:2"
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_round5")
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
    print(f"Round-5 (NO terminal_replace) on {DEVICE}", flush=True)
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

    # NO terminal_replace anywhere. All rely on sampling dynamics.
    configs = [
        # ─── Reference ───
        ("R5_base", dict()),

        # ─── L. fine h_x sweep (no termrep) ───
        ("L1_hx_stg3_0.2", dict(h_x=[0.1,0.1,0.1,0.2])),
        ("L2_hx_stg3_0.3", dict(h_x=[0.1,0.1,0.1,0.3])),
        ("L3_hx_stg3_0.4", dict(h_x=[0.1,0.1,0.1,0.4])),
        ("L4_hx_stg3_0.5", dict(h_x=[0.1,0.1,0.1,0.5])),
        ("L5_hx_stg3_0.6", dict(h_x=[0.1,0.1,0.1,0.6])),
        ("L6_hx_stg3_0.7", dict(h_x=[0.1,0.1,0.1,0.7])),
        ("L7_hx_stg3_0.8", dict(h_x=[0.1,0.1,0.1,0.8])),

        # ─── M. fine lambda_reg sweep (no termrep) ───
        ("M1_lam_stg3_5",  dict(lambda_reg=[50,50,50,5])),
        ("M2_lam_stg3_10", dict(lambda_reg=[50,50,50,10])),
        ("M3_lam_stg3_15", dict(lambda_reg=[50,50,50,15])),
        ("M4_lam_stg3_20", dict(lambda_reg=[50,50,50,20])),
        ("M5_lam_stg3_30", dict(lambda_reg=[50,50,50,30])),

        # ─── N. paper-restore proximal without termrep ───
        ("N1_prox_50",           dict(lambda_prox=50.0)),
        ("N2_prox_eq_lamreg",    dict(lambda_prox=None)),   # uses lambda_reg
        ("N3_prox_100",          dict(lambda_prox=100.0)),
        ("N4_prox_200",          dict(lambda_prox=200.0)),
        ("N5_X4_prox10_notrmrp", dict(h_x=[0.1,0.1,0.1,0.7], lambda_prox=10.0)),
        ("N6_X4_prox50_notrmrp", dict(h_x=[0.1,0.1,0.1,0.7], lambda_prox=50.0)),
        ("N7_X4_prox100_notrmrp",dict(h_x=[0.1,0.1,0.1,0.7], lambda_prox=100.0)),

        # ─── O. num_langevin scaling ───
        ("O1_L15",  dict(num_langevin=15)),
        ("O2_L20",  dict(num_langevin=20)),
        ("O3_L20_hx0.5", dict(num_langevin=20, h_x=[0.1,0.1,0.1,0.5])),

        # ─── P. CG tightening ───
        ("P1_cg_tol1e-7_iter100",  dict(cg_tol=1e-7, cg_max_iter=100)),
        ("P2_cg_hx0.5_tight",      dict(cg_tol=1e-7, cg_max_iter=100, h_x=[0.1,0.1,0.1,0.5])),

        # ─── Q. soft per-step replace (without terminal) ───
        ("Q1_softrep_0.05",  dict(soft_replace_weight=0.05)),
        ("Q2_softrep_0.1",   dict(soft_replace_weight=0.1)),
        ("Q3_softrep_0.2",   dict(soft_replace_weight=0.2)),
        ("Q4_softrep_0.5",   dict(soft_replace_weight=0.5)),

        # ─── R. hx + lambda_prox (no termrep) combined ───
        ("R1_hx0.3_prox50",  dict(h_x=[0.1,0.1,0.1,0.3], lambda_prox=50.0)),
        ("R2_hx0.5_prox50",  dict(h_x=[0.1,0.1,0.1,0.5], lambda_prox=50.0)),
        ("R3_hx0.5_prox100", dict(h_x=[0.1,0.1,0.1,0.5], lambda_prox=100.0)),

        # ─── S. Best R3 combos WITHOUT termrep ───
        ("S1_CB1_no_term", dict(h_x=[0.1,0.1,0.1,0.3], lambda_reg=[50,50,50,20])),
        ("S2_CB1_plus_softrep0.1", dict(h_x=[0.1,0.1,0.1,0.3], lambda_reg=[50,50,50,20], soft_replace_weight=0.1)),
        ("S3_X4_plus_softrep0.1",  dict(h_x=[0.1,0.1,0.1,0.7], soft_replace_weight=0.1)),
        ("S4_X4_plus_softrep0.2",  dict(h_x=[0.1,0.1,0.1,0.7], soft_replace_weight=0.2)),

        # ═══ NEW: eps-entropy preservation experiments ═══
        # W1 repro (h_eps=0.001) — anchor for this axis on R5 mask
        ("W1_repro_heps_0.001", dict(h_epsilon=0.001)),
        ("W2_heps_0.0001",      dict(h_epsilon=0.0001)),

        # H. stage-dependent h_eps schedule — only freeze eps at stg3
        ("H1_heps_stg3_0.001",  dict(h_epsilon=[0.01,0.01,0.01,0.001])),
        ("H2_heps_stg3_0.0001", dict(h_epsilon=[0.01,0.01,0.01,0.0001])),
        ("H3_heps_stg23_0.001", dict(h_epsilon=[0.01,0.01,0.001,0.001])),

        # K. skip step (f) entirely (hardest freeze)
        ("K1_skip_eps_update_all",  dict(skip_eps_update=True)),
        ("K2_skip_eps_stg3",        dict(skip_eps_update=[False,False,False,True])),

        # J. reset eps to fresh Gaussian at each ODE step start
        ("J1_reset_eps_all",  dict(reset_eps_per_ode_step=True)),
        ("J2_reset_eps_stg3", dict(reset_eps_per_ode_step=[False,False,False,True])),

        # W+C combined: preserve eps entropy + amplify x1 step
        ("WC1_heps_stg3_0.001_hx0.3", dict(h_epsilon=[0.01,0.01,0.01,0.001], h_x=[0.1,0.1,0.1,0.3])),
        ("WC2_heps_stg3_0.001_hx0.5", dict(h_epsilon=[0.01,0.01,0.01,0.001], h_x=[0.1,0.1,0.1,0.5])),
        ("WC3_heps_stg3_0.001_hx0.7", dict(h_epsilon=[0.01,0.01,0.01,0.001], h_x=[0.1,0.1,0.1,0.7])),

        # W+B combined: preserve eps entropy + lambda release
        ("WB1_heps_stg3_0.001_lam20", dict(h_epsilon=[0.01,0.01,0.01,0.001], lambda_reg=[50,50,50,20])),
        ("WB2_heps_stg3_0.001_lam10", dict(h_epsilon=[0.01,0.01,0.01,0.001], lambda_reg=[50,50,50,10])),

        # skip-eps + x1 boost
        ("KC1_skip_eps_stg3_hx0.5",  dict(skip_eps_update=[False,False,False,True], h_x=[0.1,0.1,0.1,0.5])),
        ("KC2_skip_eps_stg3_hx0.7",  dict(skip_eps_update=[False,False,False,True], h_x=[0.1,0.1,0.1,0.7])),

        # reset-eps + x1 boost
        ("JC1_reset_eps_stg3_hx0.5", dict(reset_eps_per_ode_step=[False,False,False,True], h_x=[0.1,0.1,0.1,0.5])),
    ]

    results = {}
    header = "{:<3} {:<34} {:>5} {:>7} {:>7} {:>6} {:>5}"
    print(header.format("#","name","res","psnrO","psnrAll","HF","t(s)"), flush=True)
    print("-"*78, flush=True)
    for i, (name, kw) in enumerate(configs, 1):
        try:
            xf, po, pa, res, t, hf = run_ip4(model, config, gt, y, operator, sigma_n, DEVICE, **kw)
            results[name] = dict(x=xf, psnr_obs=po, psnr_all=pa, res=res, time=t, hf=hf, kw=kw)
            print(header.format(i,name,f"{res:.0f}",f"{po:.2f}",f"{pa:.2f}",
                  f"{hf:.3f}",f"{t:.0f}"), flush=True)
        except Exception as e:
            print(f"{i:<3} {name:<34} ERROR: {e}", flush=True)

    summary = {k: {kk: vv for kk,vv in v.items() if kk!="x"} for k, v in results.items()}
    for k in summary:
        summary[k]["kw"] = {kk:vv for kk,vv in summary[k]["kw"].items()}
    with open(f"{RESULT_DIR}/round5_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    by_psnr = sorted(results.items(), key=lambda kv: kv[1]["psnr_all"], reverse=True)
    by_hf   = sorted(results.items(), key=lambda kv: abs(kv[1]["hf"] - gt_hf))

    # Pareto frontier on (psnr, hf_proximity)
    pts = [(v["psnr_all"], abs(v["hf"] - gt_hf), n) for n, v in results.items()]
    pts_sorted = sorted(pts, key=lambda t: t[0], reverse=True)
    pareto = []
    best_hf_delta = float("inf")
    for psnr, hfd, n in pts_sorted:
        if hfd < best_hf_delta:
            pareto.append(n); best_hf_delta = hfd

    print(f"\n===== RANKED BY PSNR_all =====", flush=True)
    for i, (n, v) in enumerate(by_psnr, 1):
        print(header.format(i,n,f"{v['res']:.0f}",f"{v['psnr_obs']:.2f}",
              f"{v['psnr_all']:.2f}",f"{v['hf']:.3f}",f"{v['time']:.0f}"), flush=True)
    print(f"\n===== RANKED BY HF proximity (GT={gt_hf:.3f}) =====", flush=True)
    for i, (n, v) in enumerate(by_hf, 1):
        d = abs(v["hf"] - gt_hf)
        print(f"{i:<3} {n:<34} HF={v['hf']:.3f} |Δ|={d:.3f} PSNR={v['psnr_all']:.2f}", flush=True)
    print(f"\n===== PARETO FRONTIER (PSNR ↑, HF-proximity ↑) =====", flush=True)
    for n in pareto:
        v = results[n]; d = abs(v["hf"] - gt_hf)
        print(f"{n:<34} PSNR={v['psnr_all']:.2f} HF|Δ|={d:.3f} res={v['res']:.0f}", flush=True)

    keys = [k for k, _ in by_psnr]
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
    plt.suptitle(f"IP4 round-5 (no terminal_replace) GT HF={gt_hf:.3f}", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/round5.png", dpi=150)
    plt.close()
    print(f"\nSaved {RESULT_DIR}/round5.png", flush=True)

if __name__ == "__main__":
    main()
