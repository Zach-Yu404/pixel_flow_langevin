#!/usr/bin/env python
"""One-step x1-hat prediction MSE vs t  (PixelFlowICLR experiment 1).

For each (task, image, stage k):
  1. x1_gt^k  = chain-of-bilinear-halving downsample of the 256 GT to the stage
     resolution — EXACTLY the training-time clean stage target
     (pixelflow/data_in1k.py collate, lines 63-68).
  2. For every within-stage t on the stage's inference grid
     (PixelFlowScheduler.set_timesteps with the task's LPIPS_king
     ode_steps_per_stage + shift):
       x_t^k = H_t(x1_gt^k) + sigma_t * eps          (sampler line 634 convention;
                                                      ONE eps per (image, stage),
                                                      matching the training collate
                                                      where xs/xe share one noise)
  3. ONE velocity call mu = v(x_t^k, T):
       xs_hat = x_t - t*mu ;  xe_hat = x_t + (1-t)*mu
       x1_model = direct_estimate_x1(xs_hat, xe_hat, s_k, e_k)      # "Model"
       x1_wls   = wls_estimate_x1(xs_hat, xe_hat, s_k, e_k, ...)    # "WLS"
  4. MSE(x1_*, x1_gt^k) per image.

No Langevin, no iterative refinement, no ODE rollout.

All math is REUSED from IP_package/ms_posterior_sampling_article_version_final_utils
(apply_H_tau / compute_sigma_tau / direct_estimate_x1 / wls_estimate_x1 /
make_velocity_fn) and pixelflow.scheduling_pixelflow.PixelFlowScheduler.
GT images = the 7 demo images used by ALL existing playground_runs experiments
(union of playground_runs/*/meta.json "images"; loader = demo_runner.load_demo_images).
"""

import argparse
import copy
import json
import os
import sys
import time
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
IP_PACKAGE = os.path.abspath(os.path.join(HERE, "..", "PixelFlow", "IP_package"))
sys.path.insert(0, IP_PACKAGE)
# demo_runner (imported below) does os.chdir(IP_PACKAGE) at import time; capture
# the caller's CWD NOW so relative --out never lands inside the original tree.
ORIG_CWD = os.getcwd()

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ms_posterior_sampling_article_version_final_utils import (  # noqa: E402
    apply_H_tau, compute_sigma_tau, direct_estimate_x1, wls_estimate_x1,
    make_velocity_fn,
)
from pixelflow.scheduling_pixelflow import PixelFlowScheduler  # noqa: E402
from pixelflow.utils import config as config_utils             # noqa: E402
from demo_runner import load_demo_images                       # noqa: E402

TASKS = ["box_inpainting", "random_inpainting", "gaussian_blur",
         "motion_blur", "superresolution"]
# The image set every playground run used (union over playground_runs/*/meta.json).
PLAYGROUND_IMAGES = ["breastplate_armor", "crane_structure", "ibex_horns",
                     "junco", "lakeside_beach", "sea_anemone", "shetland_sheepdog"]


def find_lpips_king_config(task):
    """tasks/<task>/configs/LPIPS_king.json, falling back to the task's
    LPIPS_king_* variant (box_inpainting ships LPIPS_king_srs1e-4__fd1_tr1)."""
    cdir = os.path.join(IP_PACKAGE, "tasks", task, "configs")
    exact = os.path.join(cdir, "LPIPS_king.json")
    if os.path.isfile(exact):
        return exact
    cands = sorted(f for f in os.listdir(cdir)
                   if f.startswith("LPIPS_king") and f.endswith(".json"))
    if not cands:
        raise FileNotFoundError(f"no LPIPS_king config for {task}")
    return os.path.join(cdir, cands[0])


def gt_stage_pyramid(gt, num_stages):
    """x1_gt^k for k=0..S-1 via chain-of-bilinear-halving (data_in1k.py:63-68).
    k = sampler stage index (0 coarsest .. S-1 = full res)."""
    pyr = {num_stages - 1: gt}
    cur = gt
    for k in range(num_stages - 2, -1, -1):
        cur = F.interpolate(cur, size=(cur.shape[-2] // 2, cur.shape[-1] // 2),
                            mode="bilinear")
        pyr[k] = cur
    return pyr


def eps_for(short_names, stage_idx, shape):
    """One eps per (image, stage), deterministic, task-independent so the same
    x_t^k is fed to every task's config (differences come from kw only)."""
    outs = []
    for name in short_names:
        seed = (zlib.crc32(f"{name}|stage{stage_idx}".encode()) ^ 42) & 0x7FFFFFFF
        g = torch.Generator(device="cpu").manual_seed(seed)
        outs.append(torch.randn(shape[1:], generator=g))
    return torch.stack(outs, dim=0)


def rope_for(model, h, w, device):
    from diffusers.models.embeddings import get_2d_rotary_pos_embed
    size_tensor = torch.tensor([h // model.patch_size], dtype=torch.int32, device=device)
    pos = get_2d_rotary_pos_embed(
        embed_dim=model.attention_head_dim,
        crops_coords=((0, 0), (h // model.patch_size, w // model.patch_size)),
        grid_size=(h // model.patch_size, w // model.patch_size),
        device=device, output_type="pt",
    )
    return size_tensor, torch.stack(pos, -1)


def per_stage(val, k, num_stages):
    if isinstance(val, (list, tuple)):
        assert len(val) == num_stages
        return val[k]
    return val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "results", "onestep_mse"))
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    ap.add_argument("--images", nargs="*", default=PLAYGROUND_IMAGES)
    ap.add_argument("--chunk", type=int, default=4, help="images per model forward")
    ap.add_argument("--self_test", action="store_true",
                    help="CPU shape/scheduler/WLS test with a fake velocity (no model)")
    args = ap.parse_args()
    if not os.path.isabs(args.out):
        args.out = os.path.join(ORIG_CWD, args.out)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    # ── demo images (playground GT set) ─────────────────────────────────────
    demos_all = load_demo_images(resolution=256, demo_dir=os.path.join(IP_PACKAGE, "demo"))
    by_short = {d["short_name"]: d for d in demos_all}
    missing = [s for s in args.images if s not in by_short]
    if missing:
        raise KeyError(f"images not in demo set: {missing}")
    demos = [by_short[s] for s in args.images]
    shorts = [d["short_name"] for d in demos]
    classes = [int(d["class_idx"]) for d in demos]
    gt = torch.stack([d["gt"] for d in demos], dim=0).to(device)  # [B,3,256,256]
    B = gt.shape[0]
    print(f"[setup] device={device}  images={shorts}  classes={classes}", flush=True)

    # ── model (skipped in self-test) ────────────────────────────────────────
    model_dir = os.path.join(IP_PACKAGE, "pretrained_models", "c2img")
    config = OmegaConf.load(os.path.join(model_dir, "config.yaml"))
    num_stages = int(config.scheduler.num_stages)
    if args.self_test:
        model = None
        patch_size, head_dim, num_classes = 4, 72, 1000
    else:
        model = config_utils.instantiate_from_config(config.model).to(device)
        ckpt = torch.load(os.path.join(model_dir, "model.pt"),
                          map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt, strict=True)
        model.eval()
        patch_size = model.patch_size
        num_classes = int(model.num_classes)
        print(f"[setup] model loaded ({sum(p.numel() for p in model.parameters())/1e6:.0f}M params)", flush=True)

    scheduler = PixelFlowScheduler(
        config.scheduler.num_train_timesteps, num_stages=num_stages, gamma=-1 / 3)

    pyr = gt_stage_pyramid(gt, num_stages)

    all_rows = []
    for task in args.tasks:
        cfg_path = find_lpips_king_config(task)
        cfg = json.load(open(cfg_path))
        kw = cfg["kw"]
        ode_steps = kw["ode_steps_per_stage"]
        shift = float(kw.get("shift", 1.0))
        guidance_scale = float(kw.get("guidance_scale", 0.0))
        g_bypass = bool(kw.get("g_bypass_stage3", True))
        lambda_x = float(kw.get("lambda_x", 0.01))
        cg_tol = float(kw.get("cg_tol", 1e-5))
        cg_max_iter = int(kw.get("cg_max_iter", 50))
        do_cfg = guidance_scale > 0
        print(f"[task {task}] cfg={os.path.basename(cfg_path)} steps={ode_steps} "
              f"shift={shift} gs={guidance_scale}", flush=True)

        task_rows = []
        t0 = time.time()
        for si in range(num_stages):
            sc = copy.deepcopy(scheduler)
            steps_si = int(per_stage(ode_steps, si, num_stages))
            sc.set_timesteps(steps_si, si, device=device, shift=shift)
            sk, ek = float(sc.start_t[si]), float(sc.end_t[si])
            eff_si = si if g_bypass else None
            rho_s_si = float(per_stage(kw.get("rho_s", 1.0), si, num_stages))
            rho_e_si = float(per_stage(kw.get("rho_e", 1.0), si, num_stages))

            x1_gt = pyr[si]                      # [B,3,h,w]
            h, w = x1_gt.shape[-2:]
            eps = eps_for(shorts, si, x1_gt.shape).to(device)

            if model is not None:
                size_tensor, rope_pos = rope_for(model, h, w, device)

            for step_idx in range(len(sc.Timesteps)):
                T = sc.Timesteps[step_idx]
                tau = float(sc.t[step_idx])
                sigma_t = compute_sigma_tau(tau, sk, ek)
                x_tau = apply_H_tau(x1_gt, tau, sk, ek, stage_idx=eff_si) + sigma_t * eps

                # ── ONE velocity evaluation, chunked over images ──
                mus = []
                for lo in range(0, B, args.chunk):
                    hi_ = min(lo + args.chunk, B)
                    if model is None:
                        mu_c = torch.zeros_like(x_tau[lo:hi_])
                    else:
                        pe = torch.tensor(classes[lo:hi_], dtype=torch.int32, device=device)
                        if do_cfg:
                            neg = num_classes * torch.ones_like(pe)
                            pe = torch.cat([neg, pe], dim=0)
                        vfn = make_velocity_fn(model, T, pe, size_tensor, rope_pos,
                                               do_cfg, guidance_scale, si)
                        with torch.no_grad():
                            mu_c = vfn(x_tau[lo:hi_])
                    mus.append(mu_c)
                mu = torch.cat(mus, dim=0)

                xs_hat = x_tau - tau * mu
                xe_hat = x_tau + (1.0 - tau) * mu
                x1_model = direct_estimate_x1(xs_hat, xe_hat, sk, ek)
                x1_wls = wls_estimate_x1(xs_hat, xe_hat, sk, ek,
                                         rho_s_si, rho_e_si, lambda_x,
                                         cg_tol, cg_max_iter, stage_idx=eff_si)

                mse_model = (x1_model - x1_gt).pow(2).mean(dim=(1, 2, 3))
                mse_wls = (x1_wls - x1_gt).pow(2).mean(dim=(1, 2, 3))
                for bi, name in enumerate(shorts):
                    task_rows.append(dict(
                        task=task, image=name, stage=si, step=step_idx,
                        tau=tau, T=float(T), sigma_tau=float(sigma_t),
                        s_k=sk, e_k=ek, resolution=h,
                        mse_wls=float(mse_wls[bi]), mse_model=float(mse_model[bi]),
                    ))
            print(f"  stage {si} ({h}x{w}, s={sk:.4f} e={ek:.4f}, {steps_si} steps) done "
                  f"[{time.time()-t0:.0f}s]", flush=True)

        # ── save raw per task ──
        tdir = os.path.join(args.out, task)
        os.makedirs(tdir, exist_ok=True)
        with open(os.path.join(tdir, "raw_mse.json"), "w") as f:
            json.dump(dict(config_path=cfg_path, kw=kw, images=shorts,
                           classes=classes, rows=task_rows), f, indent=1)
        all_rows += task_rows

        # ── figures: one per (image, stage), WLS + Model on the same axes ──
        fig_dir = os.path.join(tdir, "curves")
        os.makedirs(fig_dir, exist_ok=True)
        for name in shorts:
            for si in range(num_stages):
                rows = [r for r in task_rows if r["image"] == name and r["stage"] == si]
                if not rows:
                    continue
                taus = [r["tau"] for r in rows]
                plt.figure(figsize=(5, 3.4))
                plt.semilogy(taus, [r["mse_wls"] for r in rows], "o-", label=r"WLS $\hat{x}_1$")
                plt.semilogy(taus, [r["mse_model"] for r in rows], "s-", label=r"Model $\hat{x}_1$")
                plt.xlabel(r"$t$ (within-stage)")
                plt.ylabel(r"MSE vs $x_1^k$ (GT)")
                plt.title(f"{task} · {name} · stage {si} ({rows[0]['resolution']}px)",
                          fontsize=10)
                plt.legend(fontsize=9)
                plt.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(fig_dir, f"{name}_stage{si}.png"), dpi=140)
                plt.close()

        # ── per-task overview grid: mean over images, 1 panel per stage ──
        fig, axes = plt.subplots(1, num_stages, figsize=(4.2 * num_stages, 3.2))
        for si in range(num_stages):
            ax = axes[si]
            rows_s = [r for r in task_rows if r["stage"] == si]
            taus = sorted({r["tau"] for r in rows_s})
            mw = [np.mean([r["mse_wls"] for r in rows_s if r["tau"] == t]) for t in taus]
            mm = [np.mean([r["mse_model"] for r in rows_s if r["tau"] == t]) for t in taus]
            ax.semilogy(taus, mw, "o-", label="WLS")
            ax.semilogy(taus, mm, "s-", label="Model")
            ax.set_title(f"stage {si}", fontsize=10)
            ax.set_xlabel(r"$t$")
            ax.grid(alpha=0.3)
            if si == 0:
                ax.set_ylabel("MSE (mean over images)")
                ax.legend(fontsize=9)
        fig.suptitle(f"{task} — one-step $\\hat{{x}}_1$ MSE vs $t$ (LPIPS_king)", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(tdir, "overview_mean.png"), dpi=150)
        plt.close(fig)
        print(f"[task {task}] saved raw + {len(shorts)*num_stages} curves + overview", flush=True)

    # ── combined CSV ──
    import csv
    csv_path = os.path.join(args.out, "all_mse.csv")
    with open(csv_path, "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(all_rows)
    print(f"[done] {len(all_rows)} rows -> {csv_path}", flush=True)


if __name__ == "__main__":
    main()
