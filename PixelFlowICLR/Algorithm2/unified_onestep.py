#!/usr/bin/env python
"""Unified one-step comparison: the ONLY variable is the algorithm.

Per user spec (2026-08-16): all three estimators see the identical setup —
  x1^k   = chain-of-bilinear-halving downsample of the same 7 playground GTs
  x0^k   = the same fixed-seed noise (eps_for contract from onestep_mse_vs_t)
  x_t^k  = H_t(x1) + sigma_tau * x0
and produce x1-hat, compared to x1^k by MSE:
  WLS    = wls_estimate_x1      (one velocity call)
  Model  = direct_estimate_x1   (same call)
  Alg2   = line-14 clean solve, with the measurement TEMPORARILY set to
           y = x1^k_gt, A = I at stage resolution ("y 现在暂时用 gt"),
           eta = 0.05 (the configs' sigma_n); line-11 score solve unchanged.

No task dimension. Outputs (overwrite results/): unified_mse.csv,
mse_vs_t_unified.png (global-t axis, 3 curves), frames for ONE
unified_predictions.mp4 with rows [GT | WLS | Model | Alg2 | x_t].
"""

import argparse
import copy
import json
import os
import time

ORIG_CWD = os.getcwd()

import algorithm2 as alg  # noqa: E402
import onestep_mse_vs_t as base  # noqa: E402
from onestep_visual import to_img  # noqa: E402

import numpy as np  # noqa: E402
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

ETA = 0.05          # configs' sigma_n; y itself is noiseless GT (temporary probe)
ROW_LABELS = [r"GT $x_1^k$", r"WLS $\hat{x}_1$", r"Model $\hat{x}_1$",
              r"Alg2 $\hat{x}_1$", r"$x_t^k$"]
identity = lambda x: x  # noqa: E731  A = A^T = I at stage resolution


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(alg.HERE, "results"))
    ap.add_argument("--images", nargs="*", default=base.PLAYGROUND_IMAGES)
    ap.add_argument("--chunk", type=int, default=4)
    args = ap.parse_args()
    if not os.path.isabs(args.out):
        args.out = os.path.join(ORIG_CWD, args.out)
    frames_dir = os.path.join(args.out, "frames_tmp")
    os.makedirs(frames_dir, exist_ok=True)

    device = "cuda:0"
    demos_all = load_demo_images(resolution=256,
                                 demo_dir=os.path.join(base.IP_PACKAGE, "demo"))
    by_short = {d["short_name"]: d for d in demos_all}
    demos = [by_short[s] for s in args.images]
    shorts = [d["short_name"] for d in demos]
    classes = [int(d["class_idx"]) for d in demos]
    gt = torch.stack([d["gt"] for d in demos], dim=0).to(device)
    B = gt.shape[0]

    kw0 = json.load(open(base.find_lpips_king_config("box_inpainting")))["kw"]
    shift = float(kw0.get("shift", 1.0))
    guidance_scale = float(kw0.get("guidance_scale", 0.0))
    do_cfg = guidance_scale > 0
    g_bypass = bool(kw0.get("g_bypass_stage3", True))
    cg_tol = float(kw0.get("cg_tol", 1e-5))
    cg_max_iter = int(kw0.get("cg_max_iter", 50))
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
    print(f"[setup] model loaded, B={B}, eta={ETA}, y=GT (A=I)", flush=True)

    scheduler = PixelFlowScheduler(config.scheduler.num_train_timesteps,
                                   num_stages=num_stages, gamma=-1 / 3)
    pyr = base.gt_stage_pyramid(gt, num_stages)

    rows_out, frame_idx, t0 = [], 0, time.time()
    for si in range(num_stages):
        sc = copy.deepcopy(scheduler)
        steps_si = int(base.per_stage(kw0["ode_steps_per_stage"], si, num_stages))
        sc.set_timesteps(steps_si, si, device=device, shift=shift)
        sk, ek = float(sc.start_t[si]), float(sc.end_t[si])
        eff_si = si if g_bypass else None
        rho_s = float(base.per_stage(kw0.get("rho_s", 1.0), si, num_stages))
        rho_e = float(base.per_stage(kw0.get("rho_e", 1.0), si, num_stages))
        x1_gt = pyr[si]
        h, w = x1_gt.shape[-2:]
        x0 = base.eps_for(shorts, si, x1_gt.shape).to(device)
        size_tensor, rope_pos = base.rope_for(model, h, w, device)

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
            x1_a2 = None
            if not skip:
                x1_a2 = alg.clean_image_solve(x_tau, identity, identity, x1_gt, ETA,
                                          sigma_tau, tau, sk, ek, eff_si,
                                          x1_model.clone(), cg_tol)

            for bi, name in enumerate(shorts):
                rows_out.append(dict(
                    image=name, stage=si, step=step_idx, tau=tau,
                    sigma_tau=float(sigma_tau), resolution=h,
                    mse_wls=float(((x1_wls[bi] - x1_gt[bi]) ** 2).mean()),
                    mse_model=float(((x1_model[bi] - x1_gt[bi]) ** 2).mean()),
                    mse_alg2=float(((x1_a2[bi] - x1_gt[bi]) ** 2).mean())
                    if x1_a2 is not None else float("nan")))

            rows_t = [x1_gt, x1_wls, x1_model, x1_a2, x_tau]
            fig, axes = plt.subplots(5, B, figsize=(1.9 * B, 1.9 * 5 + 0.5))
            for ri in range(5):
                for bi in range(B):
                    ax = axes[ri, bi]
                    if rows_t[ri] is None:
                        ax.text(0.5, 0.5, r"skipped ($\sigma_tau<0.01$)",
                                ha="center", va="center", fontsize=6)
                        ax.set_facecolor("0.9")
                    else:
                        ax.imshow(to_img(rows_t[ri][bi]))
                        if 1 <= ri <= 3:
                            mse = float(((rows_t[ri][bi] - x1_gt[bi]) ** 2).mean())
                            ax.set_xlabel(f"{mse:.3g}", fontsize=6)
                    ax.set_xticks([]); ax.set_yticks([])
                    if ri == 0:
                        ax.set_title(shorts[bi].replace("_", "\n"), fontsize=7)
                    if bi == 0:
                        ax.set_ylabel(ROW_LABELS[ri], fontsize=8)
            fig.suptitle(
                f"unified one-step $\\hat{{x}}_1$ (y=GT, A=I, $\\eta$={ETA}) · "
                f"stage {si} ({h}px) · step {step_idx + 1}/{steps_si} · "
                f"t={tau:.3f} · $\\sigma_tau$={float(sigma_tau):.3f}", fontsize=11)
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            fig.savefig(os.path.join(frames_dir, f"frame_{frame_idx:03d}.png"), dpi=110)
            plt.close(fig)
            frame_idx += 1
        print(f"[stage {si}] done ({h}px) [{time.time()-t0:.0f}s]", flush=True)

    import csv
    with open(os.path.join(args.out, "unified_mse.csv"), "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        wcsv.writeheader(); wcsv.writerows(rows_out)

    def gx(r):
        return r["stage"] + r["tau"]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for est, col in (("wls", "tab:blue"), ("model", "tab:orange"), ("alg2", "tab:green")):
        for name in shorts:
            rr = sorted((r for r in rows_out if r["image"] == name), key=gx)
            ax.semilogy([gx(r) for r in rr], [r[f"mse_{est}"] for r in rr],
                        color=col, alpha=0.18, lw=0.7)
        xs = sorted({round(gx(r), 6) for r in rows_out})
        mean = [np.nanmean([r[f"mse_{est}"] for r in rows_out
                            if round(gx(r), 6) == x]) for x in xs]
        ax.semilogy(xs, mean, "-", color=col, lw=2,
                    label={"wls": "WLS", "model": "Model", "alg2": "Alg2 (y=GT)"}[est])
    for bdy in (1, 2, 3):
        ax.axvline(bdy, color="k", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("global $t$ (stage + within-stage $t$)")
    ax.set_ylabel(r"MSE vs GT $x_1^k$")
    ax.set_title(f"Unified one-step comparison — only variable = algorithm "
                 f"(y=GT, A=I, $\\eta$={ETA}; bold=mean, thin=per image)")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "mse_vs_t_unified.png"), dpi=160)
    plt.close(fig)
    print(f"[done] {len(rows_out)} rows, {frame_idx} frames", flush=True)


if __name__ == "__main__":
    main()
