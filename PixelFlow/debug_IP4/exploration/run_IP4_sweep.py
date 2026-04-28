#!/usr/bin/env python
"""
debug_IP4 sweep — explores all unopened axes from IP3:

  A. CFG guidance_scale schedule (never tested in IP3)
  B. Per-stage lambda_reg schedule  (release late stages for texture)
  C. Per-stage h_x schedule         (larger step at stage 3 where G is identity)
  D. Per-stage num_langevin         (fewer Langevin late => less smoothing)
  E. DPS pre-kick hybrid            (stage-targeted zeta)
  F. Terminal / soft replacement    (hard data consistency)
  G. Mask-aware h_x (observed vs unobserved)
  H. More ODE steps late stage
  I. Per-stage noise_scale injection (unleash texture stochasticity)
  J. WLS vs direct x1 init

Every run uses the SAME gt/y as IP3 methods sweep for direct comparability.
"""
import os, sys, json, time, copy
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from inpaintingStart import get_operator
from pixelflow.utils import config as config_utils
from pixelflow.utils.misc import seed_everything

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "debug_IP4"))
from ms_sampler_v5 import run_ip4, hf_energy

DEVICE = "cuda:0"
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_sweep")
os.makedirs(RESULT_DIR, exist_ok=True)


def load_same_gt():
    """Match IP3 methods sweep gt for direct comparison."""
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
        transforms.Normalize([.5] * 3, [.5] * 3, inplace=True),
    ])
    ds = datasets.ImageFolder(data_dir, tf)
    torch.manual_seed(20000120)
    ci = [i for i, (_, y) in enumerate(ds.samples) if int(y) == 10]
    perm = torch.randperm(len(ci))[:2]
    return torch.stack([ds[ci[i]][0] for i in perm.tolist()]).to(DEVICE)


def main():
    print("Loading model & data…", flush=True)
    gt = load_same_gt()
    config = OmegaConf.load("pretrained_models/c2img/config.yaml")
    model = config_utils.instantiate_from_config(config.model).to(DEVICE)
    ckpt = torch.load("pretrained_models/c2img/model.pt",
                      map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt, strict=True); model.eval()
    print("Ready.\n", flush=True)

    sigma_n = 0.05
    operator = get_operator(
        "inpainting", resolution=256, device=DEVICE, sigma=sigma_n,
        mask_type="box", mask_len_range=(80, 160), mask_prob_range=None,
    )
    # Freeze mask by pre-building it from gt (same as IP3)
    y = operator(gt).detach()
    mask = operator.get_mask(x=gt).float().to(DEVICE)
    gt_hf = hf_energy(gt * (1 - mask[0:1]))
    print(f"BOX  GT HF_unobs={gt_hf:.3f}", flush=True)

    # ───────────────────────── experiment configs ─────────────────────────
    configs = [
        # ----- reference baseline (= IP3 baseline_L10) -----
        ("A0_baseline", dict()),

        # ═══ A. CFG guidance_scale (never tested in IP3) ═══
        ("A1_cfg_1.0",  dict(guidance_scale=1.0)),
        ("A2_cfg_2.0",  dict(guidance_scale=2.0)),
        ("A3_cfg_4.0",  dict(guidance_scale=4.0)),
        ("A4_cfg_1.5",  dict(guidance_scale=1.5)),

        # ═══ B. per-stage lambda_reg (release late stages) ═══
        ("B1_lam_50to5",   dict(lambda_reg=[50, 50, 50, 5])),
        ("B2_lam_50to10",  dict(lambda_reg=[50, 50, 50, 10])),
        ("B3_lam_50to20",  dict(lambda_reg=[50, 50, 30, 10])),
        ("B4_lam_ramp",    dict(lambda_reg=[50, 40, 30, 20])),

        # ═══ C. per-stage h_x (larger at stage 3 where G=I) ═══
        ("C1_hx_upramp",  dict(h_x=[0.1, 0.1, 0.1, 0.3])),
        ("C2_hx_stage3",  dict(h_x=[0.1, 0.1, 0.1, 0.5])),
        ("C3_hx_dualup",  dict(h_x=[0.1, 0.1, 0.2, 0.4])),

        # ═══ D. per-stage num_langevin (less smoothing late) ═══
        ("D1_L_10_10_10_3", dict(num_langevin=[10, 10, 10, 3])),
        ("D2_L_10_10_5_3",  dict(num_langevin=[10, 10, 5, 3])),
        ("D3_L_10_10_10_1", dict(num_langevin=[10, 10, 10, 1])),

        # ═══ E. DPS pre-kick hybrid (targeted) ═══
        ("E1_dps_stg3_z5",  dict(dps_kick_zeta=[0, 0, 0, 5])),
        ("E2_dps_stg23_z3", dict(dps_kick_zeta=[0, 0, 3, 3])),
        ("E3_dps_all_z3",   dict(dps_kick_zeta=3.0)),

        # ═══ F. terminal / soft replacement ═══
        ("F1_termrep_1.0",   dict(terminal_replace_weight=1.0)),
        ("F2_termrep_0.7",   dict(terminal_replace_weight=0.7)),
        ("F3_softrep_0.3",   dict(soft_replace_weight=0.3)),
        ("F4_softrep_0.1",   dict(soft_replace_weight=0.1)),

        # ═══ G. mask-aware h_x (observed region ratio) ═══
        ("G1_obs_0.3",   dict(h_x_obs_ratio=0.3)),
        ("G2_obs_2.0",   dict(h_x_obs_ratio=2.0)),

        # ═══ H. more ODE steps late ═══
        ("H1_steps_10_10_10_20", dict(ode_steps_per_stage=[10, 10, 10, 20])),
        ("H2_steps_10_10_20_20", dict(ode_steps_per_stage=[10, 10, 20, 20])),

        # ═══ I. per-stage noise_scale injection (late) ═══
        ("I1_noise_stg3_0.3", dict(noise_scale=[0, 0, 0, 0.3])),
        ("I2_noise_stg3_0.5", dict(noise_scale=[0, 0, 0, 0.5])),
        ("I3_noise_stg23_0.3",dict(noise_scale=[0, 0, 0.3, 0.3])),

        # ═══ J. x1_init WLS ═══
        ("J1_wls_init", dict(x1_init_mode="wls")),

        # ═══ K. combined candidates ═══
        ("K1_combo_cfg2_termrep1",
         dict(guidance_scale=2.0, terminal_replace_weight=1.0)),
        ("K2_combo_cfg2_L10-10-10-3_termrep1",
         dict(guidance_scale=2.0,
              num_langevin=[10, 10, 10, 3],
              terminal_replace_weight=1.0)),
        ("K3_combo_cfg2_lam50to10_termrep1",
         dict(guidance_scale=2.0,
              lambda_reg=[50, 50, 50, 10],
              terminal_replace_weight=1.0)),
        ("K4_combo_cfg2_dpsstg3_termrep1",
         dict(guidance_scale=2.0,
              dps_kick_zeta=[0, 0, 0, 3],
              terminal_replace_weight=1.0)),
        ("K5_combo_cfg2_noise_stg3_termrep1",
         dict(guidance_scale=2.0,
              noise_scale=[0, 0, 0, 0.3],
              terminal_replace_weight=1.0)),
        ("K6_combo_cfg2_hxstg3_termrep1",
         dict(guidance_scale=2.0,
              h_x=[0.1, 0.1, 0.1, 0.3],
              terminal_replace_weight=1.0)),
        ("K7_combo_cfg1.5_L10-10-10-3",
         dict(guidance_scale=1.5,
              num_langevin=[10, 10, 10, 3])),
    ]

    results = {}
    total = len(configs)
    header_fmt = "{:<3} {:<34} {:>6} {:>7} {:>9} {:>9} {:>6} {:>6}"
    print(header_fmt.format(
        "#", "name", "res", "psnrO", "psnrAll", "HF", "time", "ok"),
        flush=True)
    print("-" * 90, flush=True)

    for i, (name, kw) in enumerate(configs, 1):
        t0 = time.time()
        try:
            xf, po, pa, res, t, hf = run_ip4(
                model, config, gt, y, operator, sigma_n, DEVICE, **kw)
            ok = "✓"
            results[name] = dict(
                x=xf, psnr_obs=po, psnr_all=pa, res=res,
                time=t, hf=hf, kw=kw,
            )
        except Exception as e:
            print(f"{i:<3} {name:<34}   ERROR: {e}", flush=True)
            continue
        print(header_fmt.format(
            i, name, f"{res:.0f}", f"{po:.2f}", f"{pa:.2f}",
            f"{hf:.3f}", f"{t:.0f}s", ok),
            flush=True)

    # Save JSON summary
    summary = {
        k: {kk: vv for kk, vv in v.items() if kk != "x"}
        for k, v in results.items()
    }
    for k in summary:
        summary[k]["kw"] = {
            kk: vv for kk, vv in summary[k]["kw"].items()
        }
    with open(f"{RESULT_DIR}/sweep_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # ─────────────── ranked tables ───────────────
    by_psnr = sorted(results.items(), key=lambda kv: kv[1]["psnr_all"], reverse=True)
    by_hf   = sorted(results.items(),
                     key=lambda kv: abs(kv[1]["hf"] - gt_hf))
    print(f"\n{'=' * 90}\nRANKED BY PSNR_all", flush=True)
    print(header_fmt.format("#", "name", "res", "psnrO", "psnrAll", "HF", "time", "ok"),
          flush=True); print("-" * 90, flush=True)
    for i, (n, v) in enumerate(by_psnr, 1):
        print(header_fmt.format(i, n, f"{v['res']:.0f}",
              f"{v['psnr_obs']:.2f}", f"{v['psnr_all']:.2f}",
              f"{v['hf']:.3f}", f"{v['time']:.0f}s", "✓"), flush=True)

    print(f"\n{'=' * 90}\nRANKED BY HF proximity to GT ({gt_hf:.3f})", flush=True)
    for i, (n, v) in enumerate(by_hf, 1):
        d = abs(v["hf"] - gt_hf)
        print(f"{i:<3} {n:<34} HF={v['hf']:.3f}  |Δ|={d:.3f}  "
              f"PSNR={v['psnr_all']:.2f}", flush=True)

    # ─────────────── visualization ───────────────
    keys = [k for k, _ in by_psnr]
    n = len(keys)
    cols = min(7, n + 1); rows = (n + 1 + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.8, rows * 2.8))
    if rows == 1: axes = axes[np.newaxis, :]
    axes = axes.flatten()
    img_gt = (gt[0].cpu().permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1).numpy()
    axes[0].imshow(img_gt); axes[0].set_title("GT", fontsize=6)
    axes[0].axis("off")
    for i, k in enumerate(keys):
        img = (results[k]["x"][0].permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1).numpy()
        axes[i + 1].imshow(img)
        axes[i + 1].set_title(
            f"{k}\n{results[k]['psnr_all']:.2f}dB  hf={results[k]['hf']:.3f}",
            fontsize=5)
        axes[i + 1].axis("off")
    for j in range(i + 2, len(axes)):
        axes[j].axis("off")
    plt.suptitle("IP4 sweep — all axes beyond IP3", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/IP4_sweep.png", dpi=150)
    plt.close()
    print(f"\nSaved {RESULT_DIR}/IP4_sweep.png", flush=True)
    print("DONE.", flush=True)


if __name__ == "__main__":
    main()
