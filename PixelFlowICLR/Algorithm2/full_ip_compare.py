#!/usr/bin/env python
"""Full inverse-problem comparison (experiment 3).

5 tasks x 3 methods, unified setup (same GT/y/seed/grid):
  WLS   : existing run_posterior_sampling, configs_best_out kw, x1_init_mode='wls'
  Model : same, x1_init_mode='model'
  Alg2  : Algorithm 2 of the ICLR draft, implemented line-by-line (p.13):
          score solve (Cor. 8, measured gamma2(k,tau)) -> stochastic exact draw
          (line 13-14, xi_y/xi_h) -> x0 <- (x_tau - H x1)/sigma (line 15) ->
          x0 Langevin step (line 17, h0) -> ancestral transition U1=nearest x2
          + fresh x0 (line 20). eta = config sigma_n. Ridge at tau=0 (Prop. 4).

Outputs (results/full_ip/): full_ip_metrics.csv (per task/method/image/step
MSE(x1-hat, x1_gt^k)), nfe.json (measured via a model.forward counter),
loss_curves.png (5 panels x 3 methods, global-t axis), frames for ONE
full_ip_trajectories.mp4 (junco; 5 task segments x 40 steps; rows=methods,
cols = x_tau | x1-hat | x0-hat).
"""

import argparse
import copy
import json
import math
import os
import time

ORIG_CWD = os.getcwd()

import algorithm2 as alg  # noqa: E402
import onestep_mse_vs_t as base  # noqa: E402
from onestep_visual import to_img  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ms_posterior_sampling_article_version_final import (  # noqa: E402
    run_posterior_sampling, _per_stage,
)
from ms_posterior_sampling_article_version_final_utils import (  # noqa: E402
    apply_H_tau, compute_sigma_tau, cg_solve, make_velocity_fn, make_Ak_fns,
)
from pixelflow.scheduling_pixelflow import PixelFlowScheduler  # noqa: E402
from pixelflow.utils import config as config_utils             # noqa: E402
from demo_runner import build_setup_and_measurement, load_demo_images  # noqa: E402

BEST_DIR = os.path.join(base.IP_PACKAGE, "rerun_imageNet", "configs_best_out")
H0 = 0.1              # Block-2 step (paper SS7: raise toward 1e-1; single value, unswept)
TRAJ_IMAGE = "junco"  # trajectory/mp4 image
METHODS = ["wls", "model", "alg2"]

NFE = {"n": 0}


def count_nfe_hook(module, args, kwargs=None):
    NFE["n"] += 1


def best_cfg(task):
    c = json.load(open(os.path.join(BEST_DIR, f"{task}_best.json")))
    op = dict(c["operator"])
    if task == "box_inpainting" and "box_len" in op:      # demo_runner key mapping
        bl = int(op.pop("box_len"))
        op["mask_len_range"] = [bl, bl + 1]
    return c, op


def run_posterior_sampling_alg2(
    model, config, gt, y, operator, eta, device,
    *,
    # Per-stage schedules — same names/positions as run_posterior_sampling
    num_langevin=10,                 # feeds S = paper's INNER iterations (l.8);
                                     # NOT the x0-Langevin count (see x0_langevin_steps)
    ode_steps_per_stage=10, shift=1.0,
    # Structural / physical
    guidance_scale=0.0, class_label=10,
    g_bypass_stage3=True,
    cg_tol=1e-5, cg_max_iter=50,     # paper: CG iterations L
    make_Ak_fns_fn=None,
    seed=42,
    record_trajectory=False,
    # Algorithm-2 specific
    gamma2_tab=None,                 # measured gamma^2(k, tau) table (Cor. 8)
    h0=H0,                           # paper: Block-2 step size h_0
    x0_langevin_steps=1,             # K Langevin steps of l.16-17 per inner iter
                                     # (l.15 once; paper = 1)
    gamma2_scale=1.0,                # multiplies the measured gamma2 table (0 => gamma2=0)
    anchor=0.0,                      # EXPERIMENTAL Tweedie anchor, see below
    **unused_kw,                     # PRINCIPLE-only kw (h_x, lambda_reg, ...) ignored
):
    """Algorithm 2 (draft p.13) with the SAME section layout as
    ``run_posterior_sampling`` (ms_posterior_sampling_article_version_final.py)
    so the two samplers can be compared side by side. Differences are the
    paper-line comments: x1 init to 0 (l.1), exact Block-1 draw (l.12-14, no
    h1), x0 recompute (l.15), Block-2 x0 Langevin (l.16-17), and the ancestral
    transition U^(1) + fresh x0 (l.20) instead of the renoise transition.

    anchor > 0 (EXPERIMENTAL, task debug-box-alg2-hole): treats the l.10
    velocity's x1_model = direct_estimate_x1(v) as a Gaussian pseudo-observation
    with precision `anchor` (M_tau += anchor*I, b_tilde += anchor*x1_model +
    sqrt(anchor)*xi_a — Lem.-5 exact-draw semantics kept, zero extra NFE).
    anchor=0.0 reproduces the paper's Algorithm 2 unchanged. Rationale: in
    ker(A) the conditional (62) carries no image prior on x1; the hole chain is
    a non-contracting random walk (oracle sim: std ~24x prior).

    Returns (x1, rows, traj): rows = per-(stage, step) MSE vs the GT stage
    pyramid (instrumentation only), traj = (x_tau, x1, x0_hat) per step when
    ``record_trajectory``.
    """
    if make_Ak_fns_fn is None:
        make_Ak_fns_fn = make_Ak_fns

    B = gt.shape[0]
    S = int(float(num_langevin))         # paper symbol
    L = int(cg_max_iter)                 # paper symbol

    num_stages = int(config.scheduler.num_stages)
    scheduler = PixelFlowScheduler(
        config.scheduler.num_train_timesteps,
        num_stages=num_stages, gamma=-1 / 3,
    )

    # CFG / class-label setup — identical to run_posterior_sampling
    pe_labels = torch.tensor([int(class_label)] * B, dtype=torch.int32, device=device)
    do_cfg = guidance_scale > 0
    if do_cfg:
        uncond_label = int(model.num_classes)
        prompt_embeds = torch.cat([uncond_label * torch.ones_like(pe_labels),
                                   pe_labels], dim=0)
    else:
        prompt_embeds = pe_labels

    # Initial spatial size derives from gt — identical to run_posterior_sampling
    target_h, target_w = int(gt.shape[-2]), int(gt.shape[-1])
    init_factor = 2 ** (num_stages - 1)
    h, w = target_h // init_factor, target_w // init_factor

    # Deterministic CPU noise stream (all xi draws) + GT stage pyramid (metrics)
    g = torch.Generator(device="cpu").manual_seed(int(seed))

    def randn_like_cpu(x):
        return torch.randn(x.shape, generator=g).to(x.device)

    pyr = base.gt_stage_pyramid(gt, num_stages)
    x1 = torch.zeros((B, 3, h, w), device=device)                    # l.1
    rows, traj = [], []

    # ── main loop ──────────────────────────────────────────────────────
    for si in range(num_stages):
        sc = copy.deepcopy(scheduler)
        ode_steps_si = int(float(_per_stage(ode_steps_per_stage, si, num_stages)))
        sc.set_timesteps(ode_steps_si, si, device=device, shift=shift)
        s_k = float(sc.start_t[si])
        e_k = float(sc.end_t[si])
        eff_si = si if g_bypass_stage3 else None

        if si > 0:                                                   # l.20: U^(1)
            h *= 2
            w *= 2
            x1 = F.interpolate(x1, size=(h, w), mode="nearest")
        x0 = randn_like_cpu(pyr[si])                                 # l.3 / l.20 fresh x0

        Ak, ATk = make_Ak_fns_fn(operator, y, (B, 3, h, w), device)

        size_tensor, rope_pos = base.rope_for(model, h, w, device)
        gamma2_stage = gamma2_tab[str(si)]

        for step_idx, T in enumerate(sc.Timesteps):
            tau = float(sc.t[step_idx])
            sigma_tau = compute_sigma_tau(tau, s_k, e_k)             # l.5
            velocity_fn = make_velocity_fn(
                model, T, prompt_embeds, size_tensor, rope_pos,
                do_cfg, guidance_scale, si,
            )
            gamma2 = float(gamma2_scale) * float(gamma2_stage.get(
                f"{round(tau, 6)}", list(gamma2_stage.values())[step_idx]))
            x0_hat = None
            if sigma_tau >= alg.SIGMA_MIN:
                inv_e2, inv_s2 = 1.0 / eta ** 2, 1.0 / float(sigma_tau) ** 2

                def M0(x):                                           # l.7 (pre-ridge)
                    return inv_e2 * ATk(Ak(x)) + inv_s2 * apply_H_tau(
                        apply_H_tau(x, tau, s_k, e_k, eff_si), tau, s_k, e_k, eff_si)
                epsilon = 0.0
                if tau == 0.0:                                       # Prop. 4
                    epsilon = alg.power_iter_norm(M0, x1.shape, device) * 1e-6
                # anchor path disabled by user (see commented block below);
                # keep M consistent with b_tilde: ridge only.
                M_tau = (lambda x: M0(x) + epsilon * x) if epsilon else M0

                for s in range(S):                                   # l.8
                    x_tau = apply_H_tau(x1, tau, s_k, e_k, eff_si) + sigma_tau * x0  # l.9
                    with torch.no_grad():
                        v = velocity_fn(x_tau)                       # l.10
                    x0_hat = alg.score_solve(x_tau, v, s_k, e_k, tau, gamma2,
                                             eff_si, cg_tol, L)      # l.11
                    xi_y = randn_like_cpu(y)                         # l.12
                    xi_h = randn_like_cpu(x1)
                    b_tilde = inv_e2 * ATk(y) + \
                        inv_s2 * apply_H_tau(x_tau, tau, s_k, e_k, eff_si) + \
                        (1.0 / eta) * ATk(xi_y) + \
                        (1.0 / float(sigma_tau)) * apply_H_tau(xi_h, tau, s_k, e_k, eff_si)  # l.13
                    # if anchor > 0:                                   # EXPERIMENTAL anchor
                    #     x1_model = alg.direct_estimate_x1(
                    #         x_tau - tau * v, x_tau + (1.0 - tau) * v, s_k, e_k)
                    #     b_tilde = b_tilde + anchor * x1_model + \
                    #         math.sqrt(anchor) * randn_like_cpu(x1)
                    x1 = cg_solve(M_tau, b_tilde, x0=x1.clone(), tol=cg_tol,
                                  max_iter=200 if tau == 0.0 else L)  # l.14
                    x0 = (x_tau - apply_H_tau(x1, tau, s_k, e_k, eff_si)) / float(sigma_tau)  # l.15
                    for _ in range(int(x0_langevin_steps)):          # user: try K in {1,3,5}
                        xi_0 = randn_like_cpu(x0)                    # l.16
                        x0 = x0 - (h0 / 2.0) * (x0 + x0_hat) + math.sqrt(h0) * xi_0  # l.17

            rows.append(dict(stage=si, step=step_idx, tau=tau,
                             sigma_tau=float(sigma_tau),
                             mse_x1=float(((x1 - pyr[si]) ** 2).mean())))
            if record_trajectory:
                x_tau_rec = apply_H_tau(x1, tau, s_k, e_k, eff_si) + float(sigma_tau) * x0
                traj.append((x_tau_rec[0].cpu(), x1[0].cpu(),
                             (x0_hat if x0_hat is not None else x0)[0].cpu()))
    return x1, rows, traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(alg.HERE, "results", "full_ip"))
    ap.add_argument("--images", nargs="*", default=base.PLAYGROUND_IMAGES)
    ap.add_argument("--tasks", nargs="*", default=alg.TASKS)
    ap.add_argument("--smoke", action="store_true", help="1 task, 1 image, S=2")
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
    if args.smoke:
        args.tasks, demos = args.tasks[:1], [by_short[TRAJ_IMAGE]]

    model_dir = os.path.join(base.IP_PACKAGE, "pretrained_models", "c2img")
    config = OmegaConf.load(os.path.join(model_dir, "config.yaml"))
    model = config_utils.instantiate_from_config(config.model).to(device)
    ckpt = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu",
                      weights_only=False)
    model.load_state_dict(ckpt, strict=True)
    model.eval()
    model.register_forward_pre_hook(count_nfe_hook)
    print("[setup] model loaded", flush=True)

    gamma2_tab = json.load(open(os.path.join(alg.HERE, "gamma2_meas.json")))["table"]
    num_stages = int(config.scheduler.num_stages)

    all_rows, nfe_stats, trajs = [], {}, {}
    t0 = time.time()
    for task in args.tasks:
        cfg, op_cfg = best_cfg(task)
        kw = dict(cfg["kw"])
        if args.smoke:
            kw["num_langevin"] = 2
        sigma_n = float(cfg.get("sigma_n", 0.05))
        for d in demos:
            name = d["short_name"]
            op, mask, y, _, _, make_Ak_fns_fn, term_fn = build_setup_and_measurement(
                task, op_cfg, d, sigma_n, 256, device)
            if task in ("gaussian_blur", "motion_blur"):
                inner = make_Ak_fns_fn

                def make_Ak_fns_fn(operator, y_, ss, dev, _inner=inner):
                    Af, _ = _inner(operator, y_, ss, dev)
                    return Af, alg.make_exact_AT(Af, tuple(ss))
            gt = d["gt"].unsqueeze(0).to(device)
            pyr = base.gt_stage_pyramid(gt, num_stages)
            record = (name == TRAJ_IMAGE)

            for method in METHODS:
                NFE["n"] = 0
                ts = time.time()
                if method == "alg2":
                    _, rows, traj = run_posterior_sampling_alg2(
                        model, config, gt, y, op, sigma_n, device,
                        gamma2_tab=gamma2_tab, make_Ak_fns_fn=make_Ak_fns_fn,
                        seed=int(kw.get("seed", 42)), record_trajectory=record,
                        **{k_: v for k_, v in kw.items()
                           if k_ not in ("class_label", "seed")},
                        class_label=int(d["class_idx"]))
                else:
                    kw_run = {k_: v for k_, v in kw.items()}
                    kw_run["x1_init_mode"] = method          # 'wls' | 'model'
                    _, tj = run_posterior_sampling(
                        model, config, gt, y, op, sigma_n, device,
                        record_trajectory=True,
                        terminal_replacement_fn=term_fn, make_Ak_fns_fn=make_Ak_fns_fn,
                        **{k_: v for k_, v in kw_run.items()
                           if k_ not in ("class_label",)},
                        class_label=int(d["class_idx"]))
                    # per ODE step: x_tau, x1(after), x0(=eps)
                    traj = [(a[0].cpu(), b_[0].cpu(), c[0].cpu()) for a, b_, c in
                            zip(tj["xts_traj"], tj["x1_traj_after"], tj["eps_traj"])] \
                        if record else []
                    # Stage is derived from each trajectory tensor's OWN resolution
                    # (self-aligning; step-counting misreads the recording convention).
                    sch = PixelFlowScheduler(config.scheduler.num_train_timesteps,
                                             num_stages=num_stages, gamma=-1 / 3)
                    taus_k = {}
                    for k in range(num_stages):
                        scc = copy.deepcopy(sch)
                        scc.set_timesteps(int(float(kw.get("ode_steps_per_stage", 10))),
                                          k, device=device, shift=float(kw.get("shift", 1.0)))
                        taus_k[k] = [float(t) for t in scc.t]
                    base_res = pyr[0].shape[-1]
                    rows, seen = [], {}
                    for x1s_cpu in tj["x1_traj_after"]:
                        k = int(round(math.log2(x1s_cpu.shape[-1] / base_res)))
                        i = seen.get(k, 0); seen[k] = i + 1
                        tau = taus_k[k][min(i, len(taus_k[k]) - 1)]
                        x1s = x1s_cpu.to(device)
                        rows.append(dict(stage=k, step=i, tau=tau,
                                         sigma_tau=float("nan"),
                                         mse_x1=float(((x1s - pyr[k]) ** 2).mean())))
                    del tj
                for r in rows:
                    r.update(task=task, image=name, method=method)
                all_rows += rows
                nfe_stats[f"{task}/{name}/{method}"] = NFE["n"]
                if record:
                    trajs[(task, method)] = traj
                print(f"[{task}/{name}/{method}] NFE={NFE['n']} "
                      f"[{time.time()-ts:.0f}s tot {time.time()-t0:.0f}s]", flush=True)

    json.dump(nfe_stats, open(os.path.join(args.out, "nfe.json"), "w"), indent=1)
    import csv
    if all_rows:
        with open(os.path.join(args.out, "full_ip_metrics.csv"), "w", newline="") as f:
            wcsv = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            wcsv.writeheader(); wcsv.writerows(all_rows)
    print(f"[done-sampling] rows={len(all_rows)}", flush=True)

    # ── loss curves: per task panel, mean over images with traj rows ──
    if all_rows and not args.smoke:
        fig, axes = plt.subplots(1, len(args.tasks), figsize=(4.0 * len(args.tasks), 3.6))
        axes = np.atleast_1d(axes)
        for ti, task in enumerate(args.tasks):
            ax = axes[ti]
            for method, col in zip(METHODS, ("tab:blue", "tab:orange", "tab:green")):
                rs = [r for r in all_rows if r["task"] == task and r["method"] == method]
                if not rs:
                    continue
                xs = sorted({r["stage"] + r["tau"] for r in rs})
                mean = [np.mean([r["mse_x1"] for r in rs
                                 if r["stage"] + r["tau"] == x]) for x in xs]
                ax.semilogy(xs, mean, "-", color=col, lw=1.8,
                            label={"wls": "Alg1-WLS", "model": "Alg1-Model",
                                   "alg2": "Alg2"}[method])
            for bdy in (1, 2, 3):
                ax.axvline(bdy, color="k", lw=0.7, ls="--", alpha=0.5)
            ax.set_title(task, fontsize=10)
            ax.set_xlabel("global $t$")
            ax.grid(alpha=0.3)
            if ti == 0:
                ax.set_ylabel(r"MSE$(\hat{x}_1^k, x_1^k)$")
                ax.legend(fontsize=8)
        fig.suptitle("Full inverse-problem sampling — loss vs time "
                     "(best configs; Alg2 per PDF, $h_0$=0.1, measured $\\gamma^2$)",
                     fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(os.path.join(args.out, "loss_curves.png"), dpi=160)
        plt.close(fig)

    # ── ONE mp4: 5 task segments x 40 steps; rows=methods, cols=x_tau|x1|x0 ──
    col_labels = [r"$x_\tau^k$", r"$\hat{x}_1^k$", r"$\hat{x}_0^k$"]
    frame_idx = 0
    for task in args.tasks:
        if (task, "alg2") not in trajs:
            continue
        n_steps = min(len(trajs[(task, m)]) for m in METHODS if (task, m) in trajs)
        for stp in range(n_steps):
            fig, axes = plt.subplots(3, 3, figsize=(3 * 2.2, 3 * 2.2 + 0.5))
            for mi, method in enumerate(METHODS):
                tr = trajs[(task, method)][stp]
                for ci in range(3):
                    ax = axes[mi, ci]
                    ax.imshow(to_img(tr[ci].to(device)))
                    ax.set_xticks([]); ax.set_yticks([])
                    if mi == 0:
                        ax.set_title(col_labels[ci], fontsize=10)
                    if ci == 0:
                        ax.set_ylabel({"wls": "Alg1-WLS", "model": "Alg1-Model",
                                       "alg2": "Alg2"}[method], fontsize=9)
            fig.suptitle(f"{task} · {TRAJ_IMAGE} · outer step {stp + 1}/{n_steps}",
                         fontsize=11)
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            fig.savefig(os.path.join(frames_dir, f"frame_{frame_idx:04d}.png"), dpi=110)
            plt.close(fig)
            frame_idx += 1
    print(f"[done] {frame_idx} frames -> {frames_dir}", flush=True)


if __name__ == "__main__":
    main()
