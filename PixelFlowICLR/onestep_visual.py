#!/usr/bin/env python
"""One-step x1-hat prediction VISUAL frames (PixelFlowICLR experiment 1b).

Companion to onestep_mse_vs_t.py: same pipeline (x1_gt^k -> x_t^k -> ONE
velocity call -> WLS / Model x1-hat), but instead of MSE numbers it renders,
for every (stage, t), one landscape frame:

    rows = [GT x1^k | x_t^k | WLS x1-hat | Model x1-hat]
    cols = the 7 playground images

Frames go to <out>/frames_tmp/frame_%03d.png ordered by (stage, step); a login
node then encodes them into ONE mp4 with ffmpeg (see run_onestep_visual.sbatch
+ the encode step) and deletes frames_tmp.

Verified fact (all_mse.csv): all per-image MSE values are bitwise identical
across the 5 task configs; combined with identical consumed LPIPS_king kw,
shared eps, and the operator/y never entering the one-step call path, the
experiment is task-independent (prediction tensors themselves were not
directly compared). So frames are computed ONCE with the box_inpainting
LPIPS_king config and labeled task-independent.

All math is imported from onestep_mse_vs_t (which re-exports the original
IP_package utils). No Langevin / iteration / rollout. playground_runs untouched.
"""

import argparse
import copy
import json
import os
import time

ORIG_CWD = os.getcwd()

import onestep_mse_vs_t as base  # noqa: E402  (sets sys.path, chdirs to IP_package)

import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402
import torch.nn.functional as F                               # noqa: E402
from omegaconf import OmegaConf                               # noqa: E402
import matplotlib                                             # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                               # noqa: E402

from ms_posterior_sampling_article_version_final_utils import (  # noqa: E402
    apply_H_tau, compute_sigma_tau, direct_estimate_x1, wls_estimate_x1,
    make_velocity_fn,
)
from pixelflow.scheduling_pixelflow import PixelFlowScheduler  # noqa: E402
from pixelflow.utils import config as config_utils             # noqa: E402
from demo_runner import load_demo_images                       # noqa: E402

ROW_LABELS = [r"GT $x_1^k$", r"$x_t^k$", r"WLS $\hat{x}_1$", r"Model $\hat{x}_1$"]


def to_img(x):
    """[-1,1] tensor [3,h,w] -> display array [256,256,3] (nearest upscale)."""
    if x.shape[-1] != 256:
        x = F.interpolate(x[None], size=(256, 256), mode="nearest")[0]
    return ((x.clamp(-1, 1) + 1) / 2).permute(1, 2, 0).float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(base.HERE, "results", "onestep_mse"))
    ap.add_argument("--images", nargs="*", default=base.PLAYGROUND_IMAGES)
    ap.add_argument("--chunk", type=int, default=4)
    args = ap.parse_args()
    if not os.path.isabs(args.out):
        args.out = os.path.join(ORIG_CWD, args.out)
    frames_dir = os.path.join(args.out, "frames_tmp")
    os.makedirs(frames_dir, exist_ok=True)

    device = "cuda:0"
    cfg = json.load(open(base.find_lpips_king_config("box_inpainting")))
    kw = cfg["kw"]
    shift = float(kw.get("shift", 1.0))
    guidance_scale = float(kw.get("guidance_scale", 0.0))
    do_cfg = guidance_scale > 0
    g_bypass = bool(kw.get("g_bypass_stage3", True))
    lambda_x = float(kw.get("lambda_x", 0.01))
    cg_tol, cg_max_iter = float(kw.get("cg_tol", 1e-5)), int(kw.get("cg_max_iter", 50))

    demos_all = load_demo_images(resolution=256,
                                 demo_dir=os.path.join(base.IP_PACKAGE, "demo"))
    by_short = {d["short_name"]: d for d in demos_all}
    demos = [by_short[s] for s in args.images]
    shorts = [d["short_name"] for d in demos]
    classes = [int(d["class_idx"]) for d in demos]
    gt = torch.stack([d["gt"] for d in demos], dim=0).to(device)
    B = gt.shape[0]

    model_dir = os.path.join(base.IP_PACKAGE, "pretrained_models", "c2img")
    config = OmegaConf.load(os.path.join(model_dir, "config.yaml"))
    num_stages = int(config.scheduler.num_stages)
    model = config_utils.instantiate_from_config(config.model).to(device)
    ckpt = torch.load(os.path.join(model_dir, "model.pt"),
                      map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt, strict=True)
    model.eval()
    num_classes = int(model.num_classes)
    print(f"[setup] model loaded, B={B}", flush=True)

    scheduler = PixelFlowScheduler(config.scheduler.num_train_timesteps,
                                   num_stages=num_stages, gamma=-1 / 3)
    pyr = base.gt_stage_pyramid(gt, num_stages)

    frame_idx = 0
    t0 = time.time()
    for si in range(num_stages):
        sc = copy.deepcopy(scheduler)
        steps_si = int(base.per_stage(kw["ode_steps_per_stage"], si, num_stages))
        sc.set_timesteps(steps_si, si, device=device, shift=shift)
        sk, ek = float(sc.start_t[si]), float(sc.end_t[si])
        eff_si = si if g_bypass else None
        rho_s = float(base.per_stage(kw.get("rho_s", 1.0), si, num_stages))
        rho_e = float(base.per_stage(kw.get("rho_e", 1.0), si, num_stages))

        x1_gt = pyr[si]
        h, w = x1_gt.shape[-2:]
        eps = base.eps_for(shorts, si, x1_gt.shape).to(device)
        size_tensor, rope_pos = base.rope_for(model, h, w, device)

        for step_idx in range(len(sc.Timesteps)):
            T = sc.Timesteps[step_idx]
            tau = float(sc.t[step_idx])
            sigma_t = compute_sigma_tau(tau, sk, ek)
            x_tau = apply_H_tau(x1_gt, tau, sk, ek, stage_idx=eff_si) + sigma_t * eps

            mus = []
            for lo in range(0, B, args.chunk):
                hi_ = min(lo + args.chunk, B)
                pe = torch.tensor(classes[lo:hi_], dtype=torch.int32, device=device)
                if do_cfg:
                    pe = torch.cat([num_classes * torch.ones_like(pe), pe], dim=0)
                vfn = make_velocity_fn(model, T, pe, size_tensor, rope_pos,
                                       do_cfg, guidance_scale, si)
                with torch.no_grad():
                    mus.append(vfn(x_tau[lo:hi_]))
            mu = torch.cat(mus, dim=0)

            xs_hat = x_tau - tau * mu
            xe_hat = x_tau + (1.0 - tau) * mu
            x1_model = direct_estimate_x1(xs_hat, xe_hat, sk, ek)
            x1_wls = wls_estimate_x1(xs_hat, xe_hat, sk, ek, rho_s, rho_e,
                                     lambda_x, cg_tol, cg_max_iter, stage_idx=eff_si)
            mse_m = (x1_model - x1_gt).pow(2).mean(dim=(1, 2, 3))
            mse_w = (x1_wls - x1_gt).pow(2).mean(dim=(1, 2, 3))

            # ── render one frame: 4 rows x B cols ──
            rows_t = [x1_gt, x_tau, x1_wls, x1_model]
            fig, axes = plt.subplots(4, B, figsize=(1.9 * B, 1.9 * 4 + 0.5))
            for ri in range(4):
                for bi in range(B):
                    ax = axes[ri, bi]
                    ax.imshow(to_img(rows_t[ri][bi]))
                    ax.set_xticks([]); ax.set_yticks([])
                    if ri == 0:
                        ax.set_title(shorts[bi].replace("_", "\n"), fontsize=7)
                    if bi == 0:
                        ax.set_ylabel(ROW_LABELS[ri], fontsize=9)
                    if ri == 2:
                        ax.set_xlabel(f"{mse_w[bi]:.3g}", fontsize=7)
                    if ri == 3:
                        ax.set_xlabel(f"{mse_m[bi]:.3g}", fontsize=7)
            fig.suptitle(
                f"one-step $\\hat{{x}}_1$ · stage {si} ({h}px) · "
                f"step {step_idx + 1}/{steps_si} · t={tau:.3f} · $\\sigma_t$={float(sigma_t):.3f}"
                f"   (task-independent; x-labels = per-image MSE)", fontsize=11)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            fig.savefig(os.path.join(frames_dir, f"frame_{frame_idx:03d}.png"), dpi=110)
            plt.close(fig)
            frame_idx += 1
        print(f"[stage {si}] {steps_si} frames done [{time.time()-t0:.0f}s]", flush=True)

    print(f"[done] {frame_idx} frames -> {frames_dir}", flush=True)


if __name__ == "__main__":
    main()
