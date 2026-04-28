"""
IP4 Round-7 — systematic per-stage h_epsilon schedule.

Goal: pinpoint WHICH stage's eps-collapse is responsible for the smooth-box
failure. Prior rounds already established:
  - stage-3-only h_eps reduction is NULL (entropy already collapsed by then)
  - global h_eps=0.001 is the winner

Open questions:
  1. Does stage-0 alone suffice? (build coherent x1 from start with entropy)
  2. Does an early-pair (stages 0-1) suffice? (before refinement collapses eps)
  3. Is there a "critical stage" after which entropy cannot be recovered?
  4. Does a ramp (decreasing h_eps as τ advances) preserve more content?
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
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_round7")
os.makedirs(RESULT_DIR, exist_ok=True)

HI = 0.01     # default high (collapses)
LO = 0.001    # low (preserves)


def load_gt():
    pt = "trajectory_videos/posterior_sampling/baseline_05/baseline_05_mode-f_x1.pt"
    d = torch.load(pt, map_location="cpu", weights_only=False)
    return d["gt"][:2].to(DEVICE)


def main():
    print(f"Round-7 per-stage h_epsilon on {DEVICE}", flush=True)
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

    # All with terminal_replace=1.0 for fair comparison (data-consistency decoupled)
    def cfg(schedule, **extra):
        return dict(h_epsilon=schedule, terminal_replace_weight=1.0, **extra)

    configs = [
        # ─── References ───
        ("REF_baseline",         cfg(0.01)),
        ("REF_W1_all_LO",        cfg(0.001)),

        # ─── Single-stage isolations (LO at one stage, HI elsewhere) ───
        ("S0_stage0_only",       cfg([LO, HI, HI, HI])),
        ("S1_stage1_only",       cfg([HI, LO, HI, HI])),
        ("S2_stage2_only",       cfg([HI, HI, LO, HI])),
        ("S3_stage3_only",       cfg([HI, HI, HI, LO])),

        # ─── Pair isolations ───
        ("P01_stages_01",        cfg([LO, LO, HI, HI])),
        ("P12_stages_12",        cfg([HI, LO, LO, HI])),
        ("P23_stages_23",        cfg([HI, HI, LO, LO])),
        ("P02_stages_02",        cfg([LO, HI, LO, HI])),
        ("P03_stages_03",        cfg([LO, HI, HI, LO])),
        ("P13_stages_13",        cfg([HI, LO, HI, LO])),

        # ─── Triple (3-of-4) ───
        ("T012_stages_012",      cfg([LO, LO, LO, HI])),
        ("T013_stages_013",      cfg([LO, LO, HI, LO])),
        ("T023_stages_023",      cfg([LO, HI, LO, LO])),
        ("T123_stages_123",      cfg([HI, LO, LO, LO])),

        # ─── Ramps ───
        ("R_decr_0.01_0.0001",   cfg([0.01, 0.005, 0.002, 0.001])),
        ("R_decr_0.01_0.0005",   cfg([0.01, 0.01, 0.005, 0.0005])),
        ("R_incr_0.0001_0.01",   cfg([0.0001, 0.0005, 0.002, 0.01])),  # ramp up
        ("R_v_shape",            cfg([0.001, 0.01, 0.01, 0.001])),     # V (low on ends)
        ("R_inv_v_shape",        cfg([0.01, 0.001, 0.001, 0.01])),     # Λ (low in middle)

        # ─── Very low variants ───
        ("V_all_1e-4",           cfg(0.0001)),
        ("V_stg012_1e-4_stg3_HI",cfg([1e-4, 1e-4, 1e-4, HI])),
        ("V_stg01_1e-4",         cfg([1e-4, 1e-4, HI, HI])),

        # ─── Critical-stage hypothesis test: HI in ONE stage on top of all-LO ───
        ("INV_stage0_HI",        cfg([HI, LO, LO, LO])),  # negative of S0
        ("INV_stage1_HI",        cfg([LO, HI, LO, LO])),
        ("INV_stage2_HI",        cfg([LO, LO, HI, LO])),
        ("INV_stage3_HI",        cfg([LO, LO, LO, HI])),
    ]

    results = {}
    header = "{:<3} {:<32} {:>5} {:>7} {:>7} {:>6} {:>5}"
    print(header.format("#", "name", "res", "PSNR", "unobs", "HF", "t(s)"), flush=True)
    print("-"*76, flush=True)

    for i, (name, kw) in enumerate(configs, 1):
        try:
            xf, po, pa, res, t, hf = run_ip4(model, config, gt, y, operator, sigma_n, DEVICE, **kw)
            xfD = xf.to(DEVICE)
            diff2 = (xfD - gt) ** 2
            C = xfD.shape[1]
            m = mask if mask.shape[0] == xfD.shape[0] else mask.expand_as(xfD[:, :1])
            mse_unobs = ((1-m) * diff2).sum() / max((1-m).sum().item() * C, 1)
            pu = (10 * torch.log10(4.0 / mse_unobs.clamp(min=1e-12))).item()
            results[name] = dict(x=xfD, psnr_all=pa, psnr_unobs=pu,
                                  psnr_obs=po, res=res, time=t, hf=hf,
                                  hf_delta=abs(hf - gt_hf), sched=kw["h_epsilon"])
            print(header.format(i, name, f"{res:.0f}", f"{pa:.2f}", f"{pu:.2f}",
                                 f"{hf:.3f}", f"{t:.0f}"), flush=True)
        except Exception as e:
            print(f"{i:<3} {name:<32} ERROR: {e}", flush=True)

    # save metrics
    summary = {k: {kk: (vv.tolist() if hasattr(vv, 'tolist') else vv)
                    for kk, vv in v.items() if kk != "x"}
                for k, v in results.items()}
    with open(f"{RESULT_DIR}/round7_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Rankings
    by_psnr = sorted(results.items(), key=lambda kv: kv[1]["psnr_all"], reverse=True)
    by_hf = sorted(results.items(), key=lambda kv: kv[1]["hf_delta"])

    print(f"\n===== RANKED BY PSNR_all =====", flush=True)
    for i, (n, v) in enumerate(by_psnr, 1):
        print(header.format(i, n, f"{v['res']:.0f}", f"{v['psnr_all']:.2f}", f"{v['psnr_unobs']:.2f}",
                             f"{v['hf']:.3f}", f"{v['time']:.0f}"), flush=True)

    print(f"\n===== RANKED BY HF proximity (GT={gt_hf:.3f}) =====", flush=True)
    for i, (n, v) in enumerate(by_hf, 1):
        print(f"{i:<3} {n:<32} HF={v['hf']:.3f} |Δ|={v['hf_delta']:.3f} PSNR={v['psnr_all']:.2f}", flush=True)

    # Visualization grid
    keys = [k for k, _ in by_psnr]
    cols = 6; rows = (len(keys)+1+cols-1)//cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    axes = axes.flatten()
    img_gt = ((gt[0]*0.5+0.5).clamp(0,1).permute(1,2,0).cpu().numpy())
    axes[0].imshow(img_gt); axes[0].set_title(f"GT HF={gt_hf:.3f}", fontsize=7); axes[0].axis("off")
    for i, k in enumerate(keys):
        img = ((results[k]["x"][0]*0.5+0.5).clamp(0,1).permute(1,2,0).cpu().numpy())
        axes[i+1].imshow(img)
        axes[i+1].set_title(f"{k}\nPSNR={results[k]['psnr_all']:.1f} HF={results[k]['hf']:.3f}", fontsize=6)
        axes[i+1].axis("off")
    for j in range(i+2, len(axes)): axes[j].axis("off")
    plt.suptitle(f"IP4 round-7 — per-stage h_epsilon schedule  GT HF={gt_hf:.3f}", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/round7.png", dpi=140)
    plt.close()
    print(f"Saved {RESULT_DIR}/round7.png", flush=True)

if __name__ == "__main__":
    main()
