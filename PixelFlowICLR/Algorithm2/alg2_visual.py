#!/usr/bin/env python
"""Per-t prediction frames incl. the Algorithm 2 estimator (experiment 2b).

Same frame format as onestep_visual.py (exp 1b) extended with Alg2 rows.
Alg2 is task-dependent (line 14 uses A/y), so ONE video shows everything:

One mp4 PER TASK, rows in the user-specified order:

    [GT x1 | WLS | Model | Alg2(task) | x_t]  x  7 playground images

GT/WLS/Model/x_t rows are task-independent (identical across the five videos);
only the Alg2 row differs. Frames land in frames_tmp/<task>/frame_%03d.png.

40 frames ordered by (stage, step); Alg2 panels are blanked where
sigma_tau < SIGMA_MIN (same skip rule as the experiment). Frames go to
<out>/frames_tmp/ for ffmpeg encoding on the login node (see run_alg2_visual.sbatch).
Implementation verified line-by-line against Algorithm 2 of the ICLR draft
(see .research/experiments/2026-08-16-alg2-onestep-mse-vs-t.md).
"""

import argparse
import copy
import json
import os
import time

ORIG_CWD = os.getcwd()

import algorithm2 as alg  # noqa: E402  (imports base, sets IP_package + DAPS paths)
import onestep_mse_vs_t as base  # noqa: E402
from onestep_visual import to_img  # noqa: E402

import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ms_posterior_sampling_article_version_final_utils import (  # noqa: E402
    apply_H_tau, compute_sigma_tau, direct_estimate_x1, wls_estimate_x1,
    make_velocity_fn,
)
from pixelflow.scheduling_pixelflow import PixelFlowScheduler  # noqa: E402
from pixelflow.utils import config as config_utils             # noqa: E402
from demo_runner import load_demo_images                       # noqa: E402

TASK_SHORT = {"box_inpainting": "box", "random_inpainting": "random",
              "gaussian_blur": "gaussian", "motion_blur": "motion",
              "superresolution": "SR"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(alg.HERE, "results"))
    ap.add_argument("--images", nargs="*", default=base.PLAYGROUND_IMAGES)
    ap.add_argument("--chunk", type=int, default=4)
    args = ap.parse_args()
    if not os.path.isabs(args.out):
        args.out = os.path.join(ORIG_CWD, args.out)
    frames_root = os.path.join(args.out, "frames_tmp")
    for t in alg.TASKS:
        os.makedirs(os.path.join(frames_root, t), exist_ok=True)

    device = "cuda:0"
    demos_all = load_demo_images(resolution=256,
                                 demo_dir=os.path.join(base.IP_PACKAGE, "demo"))
    by_short = {d["short_name"]: d for d in demos_all}
    demos = [by_short[s] for s in args.images]
    shorts = [d["short_name"] for d in demos]
    classes = [int(d["class_idx"]) for d in demos]
    gt = torch.stack([d["gt"] for d in demos], dim=0).to(device)
    B = gt.shape[0]

    task_setups = {t: alg.build_task_setups(t, demos, device) for t in alg.TASKS}
    kw0 = json.load(open(base.find_lpips_king_config("box_inpainting")))["kw"]
    shift = float(kw0.get("shift", 1.0))
    guidance_scale = float(kw0.get("guidance_scale", 0.0))
    do_cfg = guidance_scale > 0
    g_bypass = bool(kw0.get("g_bypass_stage3", True))
    cg_tol = float(kw0.get("cg_tol", 1e-5))
    cg_max_iter = int(kw0.get("cg_max_iter", 50))
    rho_s_all, rho_e_all = kw0.get("rho_s", 1.0), kw0.get("rho_e", 1.0)
    lambda_x = float(kw0.get("lambda_x", 0.01))

    model_dir = os.path.join(base.IP_PACKAGE, "pretrained_models", "c2img")
    config = OmegaConf.load(os.path.join(model_dir, "config.yaml"))
    num_stages = int(config.scheduler.num_stages)
    model = config_utils.instantiate_from_config(config.model).to(device)
    ckpt = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu",
                      weights_only=False)
    model.load_state_dict(ckpt, strict=True)
    model.eval()
    num_classes = int(model.num_classes)
    print(f"[setup] model loaded, B={B}", flush=True)

    scheduler = PixelFlowScheduler(config.scheduler.num_train_timesteps,
                                   num_stages=num_stages, gamma=-1 / 3)
    pyr = base.gt_stage_pyramid(gt, num_stages)
    row_labels = [r"GT $x_1^k$", r"WLS $\hat{x}_1$", r"Model $\hat{x}_1$",
                  r"Alg2 $\hat{x}_1$", r"$x_t^k$"]

    frame_idx = 0
    t0 = time.time()
    for si in range(num_stages):
        sc = copy.deepcopy(scheduler)
        steps_si = int(base.per_stage(kw0["ode_steps_per_stage"], si, num_stages))
        sc.set_timesteps(steps_si, si, device=device, shift=shift)
        sk, ek = float(sc.start_t[si]), float(sc.end_t[si])
        eff_si = si if g_bypass else None
        rho_s = float(base.per_stage(rho_s_all, si, num_stages))
        rho_e = float(base.per_stage(rho_e_all, si, num_stages))
        x1_gt = pyr[si]
        h, w = x1_gt.shape[-2:]
        x0 = base.eps_for(shorts, si, x1_gt.shape).to(device)
        size_tensor, rope_pos = base.rope_for(model, h, w, device)
        stage_A = {t: [s["mkA"](s["op"], s["y"], x1_gt[:1].shape, device)
                       for s in task_setups[t][2]] for t in alg.TASKS}

        for step_idx in range(len(sc.Timesteps)):
            T = sc.Timesteps[step_idx]
            tau = float(sc.t[step_idx])
            sigma_tau = compute_sigma_tau(tau, sk, ek)
            x_tau = apply_H_tau(x1_gt, tau, sk, ek, eff_si) + sigma_tau * x0

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
            v = torch.cat(mus, dim=0)
            xs_hat, xe_hat = x_tau - tau * v, x_tau + (1.0 - tau) * v
            x1_model = direct_estimate_x1(xs_hat, xe_hat, sk, ek)
            x1_wls = wls_estimate_x1(xs_hat, xe_hat, sk, ek, rho_s, rho_e,
                                     lambda_x, cg_tol, cg_max_iter, stage_idx=eff_si)
            skip = sigma_tau < alg.SIGMA_MIN

            alg2 = {}
            if not skip:
                for task in alg.TASKS:
                    _, eta, setups = task_setups[task]
                    outs = []
                    for bi in range(B):
                        A_fn, AT_fn = stage_A[task][bi]
                        outs.append(alg.clean_image_solve(
                            x_tau[bi:bi + 1], A_fn, AT_fn, setups[bi]["y"], eta,
                            sigma_tau, tau, sk, ek, eff_si,
                            x1_model[bi:bi + 1].clone(), cg_tol))
                    alg2[task] = torch.cat(outs, dim=0)

            for task in alg.TASKS:
                rows_t = [x1_gt, x1_wls, x1_model, alg2.get(task), x_tau]
                nrow = len(rows_t)
                fig, axes = plt.subplots(nrow, B, figsize=(1.9 * B, 1.9 * nrow + 0.5))
                for ri in range(nrow):
                    for bi in range(B):
                        ax = axes[ri, bi]
                        if rows_t[ri] is None:
                            ax.text(0.5, 0.5, r"skipped ($\sigma_tau<0.01$)",
                                    ha="center", va="center", fontsize=6)
                            ax.set_facecolor("0.9")
                        else:
                            ax.imshow(to_img(rows_t[ri][bi]))
                            if 1 <= ri <= 3:      # estimator rows get MSE labels
                                mse = float(((rows_t[ri][bi] - x1_gt[bi]) ** 2).mean())
                                ax.set_xlabel(f"{mse:.3g}", fontsize=6)
                        ax.set_xticks([]); ax.set_yticks([])
                        if ri == 0:
                            ax.set_title(shorts[bi].replace("_", "\n"), fontsize=7)
                        if bi == 0:
                            ax.set_ylabel(row_labels[ri], fontsize=8)
                fig.suptitle(
                    f"{task} · one-step $\\hat{{x}}_1$ · stage {si} ({h}px) · "
                    f"step {step_idx + 1}/{steps_si} · t={tau:.3f} · "
                    f"$\\sigma_tau$={float(sigma_tau):.3f}   (x-labels = per-image MSE)",
                    fontsize=11)
                fig.tight_layout(rect=[0, 0, 1, 0.97])
                fig.savefig(os.path.join(frames_root, task,
                                         f"frame_{frame_idx:03d}.png"), dpi=110)
                plt.close(fig)
            frame_idx += 1
        print(f"[stage {si}] {steps_si} frames [{time.time()-t0:.0f}s]", flush=True)

    print(f"[done] {frame_idx} frames x {len(alg.TASKS)} tasks -> {frames_root}",
          flush=True)


if __name__ == "__main__":
    main()
