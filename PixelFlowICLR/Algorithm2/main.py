#!/usr/bin/env python
"""Algorithm 2 experiments — single entry point.

    python main.py [--config config.json] [--mode <mode>]

ONE config file (config.json) is the ONLY source of configuration: paths
(model/demo/gamma2 table), "algorithm" constants (sigma_min, h0, ridge_rel,
seed, ...), "sampler_kw" (shared grid/velocity/CG params), "tasks_setup"
(per-task operator + sigma_n + kw overrides — INLINED from the former
LPIPS_king / configs_best_out files, values verified identical), shared
tasks/images, and one section per mode. Nothing is defaulted in code and no
external task-config files are read. Everything is Algorithm 2 only. Modes:
  onestep   : one-step Alg2 x1-hat MSE vs t (5 tasks x 7 images x 4 stages;
              sanity S1/S2 + adjoint tests first; "self_test": true = CPU sanity only)
  full_ip   : full Algorithm-2 sampling per task/image — metrics + loss curves + frames
  debug_box : box-hole harness (h0 or trw variants; obs/hole split MSE)
  verify    : CPU component audit V1-V7 (dense float64): self-adjointness of
              G/H/B/N (transpose equivalence), sigma_tau, CFG semantics,
              Ak/ATk adjoints, M0 vs dense, power_iter_norm, score_solve vs
              explicit N^T N

GPU submission (no sbatch file needed):
  sbatch -A cbig-ece -p gpu --gres=gpu:a100:1 -c 8 --mem=64G -t 3:00:00 \\
    -o ../logs/%x-%j.out --export=ALL,PYTHONHASHSEED=0 \\
    --wrap "python main.py --mode full_ip"
  (PYTHONHASHSEED=0 is required: demo_runner mask seeds still use hash().)

All math lives in utils.py; results land under results/.
"""

import os

ORIG_CWD = os.getcwd()

import argparse   # noqa: E402
import copy       # noqa: E402
import json       # noqa: E402
import math       # noqa: E402
import time       # noqa: E402

from utils import (  # noqa: E402  (import first: sets IP_package sys.path + chdir)
    HERE, NFE, count_nfe_hook,
    apply_B, apply_N, power_iter_norm, make_exact_AT, adjoint_test, mse_masked,
    score_solve, clean_image_solve, build_task_setups, apply_H_tau_inv,
    run_posterior_sampling_alg2,
)
import onestep_mse_vs_t as base  # noqa: E402

import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
import torch.nn.functional as F                                # noqa: E402
from omegaconf import OmegaConf                                # noqa: E402
import matplotlib                                              # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402

from ms_posterior_sampling_article_version_final_utils import (  # noqa: E402
    apply_G, apply_H_tau, compute_sigma_tau,
    direct_estimate_x1, make_velocity_fn, make_Ak_fns,
)
from ms_posterior_sampling_utils import class_guidance_scale   # noqa: E402
from pixelflow.scheduling_pixelflow import PixelFlowScheduler  # noqa: E402
from pixelflow.utils import config as config_utils             # noqa: E402
from demo_runner import load_demo_images                        # noqa: E402
import measurement                                              # noqa: E402
from measurement import build_setup_and_measurement             # noqa: E402
from onestep_visual import to_img                              # noqa: E402

# Populated from config.json by main() before dispatch — no defaults in code.
PATHS = {}         # resolved "paths" section ({IP_PACKAGE}/{HERE} substituted)
ALG = {}           # "algorithm" section
SAMPLER_KW = {}    # shared "sampler_kw" section
TASKS_SETUP = {}   # per-task {"sigma_n", "operator", optional "kw" overrides}
TRAJ_IMAGE = None  # "traj_image": which image gets trajectory frames (full_ip)

# The same ceph share is mounted at different roots depending on the machine
# (cluster login/compute nodes vs the mounted GPU box). Placeholder paths
# ({IP_PACKAGE}/{HERE}) are machine-independent; an absolute path written into
# config.json on one machine is re-rooted here so the config runs everywhere.
SHARE_ROOTS = [
    "/sfs/ceph/standard/CBIG-Standard-ECE",
    "/standard/CBIG-Standard-ECE",
    "/CBIG-Standard-ECE",
]


def resolve_path(p):
    """Return an existing form of p, retrying it under every known share
    root; raise with the full candidate list if none is accessible."""
    if os.path.exists(p):
        return p
    tried = [p]
    for old in SHARE_ROOTS:
        if p.startswith(old + "/"):
            for new in SHARE_ROOTS:
                cand = new + p[len(old):]
                if cand == p:
                    continue
                if os.path.exists(cand):
                    return cand
                tried.append(cand)
    raise FileNotFoundError(
        "config path not found on this machine; tried: " + ", ".join(tried))


def task_kw(task):
    return {**SAMPLER_KW, **TASKS_SETUP[task].get("kw", {})}


def _resolve_out(out):
    # Anchored to this file's directory, not the caller's CWD: running from the
    # repo root used to scatter a top-level results/ tree.
    return out if os.path.isabs(out) else os.path.join(HERE, out)


def _load_model(config, device):
    model = config_utils.instantiate_from_config(config.model).to(device)
    model_dir = PATHS["model_dir"]
    ckpt = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu",
                      weights_only=False)
    model.load_state_dict(ckpt, strict=True)
    model.eval()
    return model


# ═════════════════════════════ mode: onestep ═══════════════════════════════
def run_onestep(args):
    out = _resolve_out(args.out)
    os.makedirs(out, exist_ok=True)
    device = "cuda:0" if (torch.cuda.is_available() and not args.self_test) else "cpu"

    demos_all = load_demo_images(resolution=256, demo_dir=PATHS["demo_dir"])
    by_short = {d["short_name"]: d for d in demos_all}
    demos = [by_short[s] for s in args.images]
    shorts = [d["short_name"] for d in demos]
    classes = [int(d["class_idx"]) for d in demos]
    gt = torch.stack([d["gt"] for d in demos], dim=0).to(device)
    B = gt.shape[0]

    config = OmegaConf.load(os.path.join(PATHS["model_dir"], "config.yaml"))
    num_stages = int(config.scheduler.num_stages)
    scheduler = PixelFlowScheduler(config.scheduler.num_train_timesteps,
                                   num_stages=num_stages, gamma=-1 / 3)
    pyr = base.gt_stage_pyramid(gt, num_stages)

    # kw consumed for the shared grid/velocity (identical across the 5 tasks —
    # verified bitwise in experiment 1); per-task eta/operator re-read anyway.
    kw0 = dict(SAMPLER_KW)          # shared grid/velocity/cg params (config.json)
    shift = float(kw0.get("shift", 1.0))
    guidance_scale = float(kw0.get("guidance_scale", 0.0))
    do_cfg = guidance_scale > 0
    g_bypass = bool(kw0.get("g_bypass_stage3", True))
    cg_tol = float(kw0.get("cg_tol", 1e-5))
    cg_max_iter = int(kw0.get("cg_max_iter", 50))

    # ── per-task measurement setups (playground-identical) ──
    task_setups = {}
    for task in args.tasks:
        sigma_n_t, setups = build_task_setups(task, demos, device,
                                              TASKS_SETUP[task])
        task_setups[task] = (sigma_n_t, setups)
        print(f"[setup] {task}: eta={sigma_n_t}  y[0] shape={tuple(setups[0]['y'].shape)}",
              flush=True)

    # ── sanity + adjoint tests ──────────────────────────────────────────────
    sanity = {"adjoint": {}, "S1": {}, "S2": {}}
    for task in args.tasks:
        _, setups = task_setups[task]
        for si in range(num_stages):
            x1s = pyr[si][:1]
            A_fn, AT_fn = setups[0]["mkA"](setups[0]["op"], setups[0]["y"],
                                           x1s.shape, device)
            err = adjoint_test(A_fn, AT_fn, x1s.shape, A_fn(x1s), device)
            sanity["adjoint"][f"{task}/stage{si}"] = err
    worst = max(sanity["adjoint"].items(), key=lambda kv: kv[1])
    print(f"[adjoint] worst {worst[0]}: {worst[1]:.2e}", flush=True)
    assert worst[1] < 1e-4, f"adjoint test failed: {worst}"

    # S1: v := d_exact, gamma2=0 -> x0 recovered (exact identity)
    for si in range(num_stages):
        sc = copy.deepcopy(scheduler)
        sc.set_timesteps(int(base.per_stage(kw0["ode_steps_per_stage"], si, num_stages)),
                         si, device=device, shift=shift)
        sk, ek = float(sc.start_t[si]), float(sc.end_t[si])
        eff_si = si if g_bypass else None
        x1_gt = pyr[si][:2]
        x0 = base.eps_for(shorts[:2], si, x1_gt.shape).to(device)
        tau = float(sc.t[len(sc.t) // 2])
        sig = compute_sigma_tau(tau, sk, ek)
        if sig < ALG["sigma_min"]:
            continue
        x_tau = apply_H_tau(x1_gt, tau, sk, ek, eff_si) + sig * x0
        d_exact = apply_B(x1_gt, sk, ek, eff_si) - (ek - sk) * x0
        x0_hat = score_solve(x_tau, d_exact, sk, ek, tau, 0.0, eff_si, cg_tol, cg_max_iter)
        rel = float((x0_hat - x0).norm() / x0.norm())
        sanity["S1"][f"stage{si}(tau={tau:.3f})"] = rel
    s1_worst = max(sanity["S1"].values())
    print(f"[S1] worst rel err {s1_worst:.2e}  {sanity['S1']}", flush=True)
    assert s1_worst < 1e-4, "S1 identity check failed — wiring bug"

    # S2: GT x_t + NOISELESS y, line 14 -> observed-region error ~ eta
    # Wiring assert at stage 3 (A = mask directly, no resize): obs err must be
    # << eta. At resized stages the nearest-mask "observed" label under-states
    # the true constraint (bilinear upsample x sparse mask conditioning), so
    # stage-1 numbers are logged as informational only (verified: stage-3
    # random obsRMSE 0.011 vs eta 0.05; stage-1 ~0.09-0.11 with obs ~= miss).
    for task in [t for t in args.tasks if "inpainting" in t]:
        sigma_n_t, setups = task_setups[task]
        st = setups[0]
        for si in [1, 3]:
            sc = copy.deepcopy(scheduler)
            sc.set_timesteps(int(base.per_stage(kw0["ode_steps_per_stage"], si, num_stages)),
                             si, device=device, shift=shift)
            sk, ek = float(sc.start_t[si]), float(sc.end_t[si])
            eff_si = si if g_bypass else None
            x1_gt = pyr[si][:1]
            x0 = base.eps_for(shorts[:1], si, x1_gt.shape).to(device)
            tau = float(sc.t[len(sc.t) // 2])
            sig = compute_sigma_tau(tau, sk, ek)
            if sig < ALG["sigma_min"]:
                tau = float(sc.t[2]); sig = compute_sigma_tau(tau, sk, ek)
            x_tau = apply_H_tau(x1_gt, tau, sk, ek, eff_si) + sig * x0
            y_clean = st["op"](gt[:1]).detach()          # xi = 0
            A_fn, AT_fn = st["mkA"](st["op"], st["y"], x1_gt.shape, device)
            # warm start from ZERO, not the GT: warm-starting at the answer
            # would let a solver that returns its input unchanged pass S2.
            x1_hat = clean_image_solve(x_tau, A_fn, AT_fn, y_clean, sigma_n_t, sig,
                                       tau, sk, ek, eff_si,
                                       torch.zeros_like(x1_gt), cg_tol,
                                       ridge_rel=ALG["ridge_rel"],
                                       max_iter=ALG["cg_max_iter_l14"])
            m_st = F.interpolate(st["mask"], size=x1_gt.shape[-2:], mode="nearest")
            obs_rmse = math.sqrt(mse_masked(x1_hat, x1_gt, m_st))
            sanity["S2"][f"{task}/stage{si}"] = obs_rmse
            print(f"[S2] {task} stage{si}: observed RMSE {obs_rmse:.4f} (eta={sigma_n_t})",
                  flush=True)
            if si == 3:
                assert obs_rmse < 2 * sigma_n_t, "S2 failed — line 14 wiring bug"

    json.dump(sanity, open(os.path.join(out, "sanity.json"), "w"), indent=1)
    if args.self_test:
        print("[self-test] all checks passed", flush=True)
        return

    # ── model ───────────────────────────────────────────────────────────────
    model = _load_model(config, device)
    num_classes = int(model.num_classes)
    print("[setup] model loaded", flush=True)

    # ── main loop: velocity/score/gamma2 shared; line 14 per task ───────────
    rows, gamma2_tab = [], {}
    t0 = time.time()
    for si in range(num_stages):
        sc = copy.deepcopy(scheduler)
        steps_si = int(base.per_stage(kw0["ode_steps_per_stage"], si, num_stages))
        sc.set_timesteps(steps_si, si, device=device, shift=shift)
        sk, ek = float(sc.start_t[si]), float(sc.end_t[si])
        eff_si = si if g_bypass else None
        x1_gt = pyr[si]
        h, w = x1_gt.shape[-2:]
        x0 = base.eps_for(shorts, si, x1_gt.shape).to(device)     # kept: score/gamma2
        size_tensor, rope_pos = base.rope_for(model, h, w, device)
        d_exact = apply_B(x1_gt, sk, ek, eff_si) - (ek - sk) * x0

        # per-(task,image) stage-shape A/A^T closures
        stage_A = {t: [s["mkA"](s["op"], s["y"], x1_gt[:1].shape, device)
                       for s in task_setups[t][1]] for t in args.tasks}
        stage_mask = {t: [F.interpolate(s["mask"], size=(h, w), mode="nearest")
                          for s in task_setups[t][1]] for t in args.tasks}

        for step_idx in range(len(sc.Timesteps)):
            T = sc.Timesteps[step_idx]
            tau = float(sc.t[step_idx])
            sigma_tau = compute_sigma_tau(tau, sk, ek)
            x_tau = apply_H_tau(x1_gt, tau, sk, ek, eff_si) + sigma_tau * x0

            # ONE velocity call (shared by all tasks)
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

            xs_hat = x_tau - tau * v
            xe_hat = x_tau + (1.0 - tau) * v
            x1_model = direct_estimate_x1(xs_hat, xe_hat, sk, ek)

            # gamma2 measured (task-independent; note eq. 56)
            g2_per_img = ((v - d_exact) ** 2).mean(dim=(1, 2, 3))
            g2_meas = float(g2_per_img.mean())
            gamma2_tab.setdefault(si, {})[round(tau, 6)] = g2_meas

            skip_alg2 = sigma_tau < ALG["sigma_min"]
            if not skip_alg2:
                x0_hat = score_solve(x_tau, v, sk, ek, tau, 0.0, eff_si, cg_tol, cg_max_iter)
                x0_hat_g2 = score_solve(x_tau, v, sk, ek, tau, g2_meas, eff_si,
                                        cg_tol, cg_max_iter)
                ##Predict x_1_hat^k:
                # paper p.20 (7.2): x1_hat = (H_tau)^-1 (x_tau - sigma_tau*x0_hat).
                # With the exact displacement it is the ground truth exactly;
                # with v_theta it should read as a plausible denoising of x_tau.
                x1_hat = apply_H_tau_inv(x_tau - sigma_tau * x0_hat,
                                         tau, sk, ek, eff_si)
                x1_hat_g2 = apply_H_tau_inv(x_tau - sigma_tau * x0_hat_g2,
                                            tau, sk, ek, eff_si)

                sc_err = ((x0_hat - x0) ** 2).mean(dim=(1, 2, 3))
                sc_err_g2 = ((x0_hat_g2 - x0) ** 2).mean(dim=(1, 2, 3))
                x1h_err = ((x1_hat - x1_gt) ** 2).mean(dim=(1, 2, 3))
                x1h_err_g2 = ((x1_hat_g2 - x1_gt) ** 2).mean(dim=(1, 2, 3))

            for task in args.tasks:
                eta, setups = task_setups[task]
                inp = "inpainting" in task
                for bi, name in enumerate(shorts):
                    sl = slice(bi, bi + 1)
                    if skip_alg2:
                        x1_a2 = None
                    else:
                        A_fn, AT_fn = stage_A[task][bi]
                        x1_a2 = clean_image_solve(x_tau[sl], A_fn, AT_fn,
                                                  setups[bi]["y"], eta, sigma_tau, tau,
                                                  sk, ek, eff_si, x1_model[sl].clone(), cg_tol,
                                                  ridge_rel=ALG["ridge_rel"],
                                                  max_iter=ALG["cg_max_iter_l14"])
                    row = dict(task=task, image=name, stage=si, step=step_idx,
                               tau=tau, T=float(T), sigma_tau=float(sigma_tau),
                               s_k=sk, e_k=ek, resolution=h,
                               mse_alg2=float(((x1_a2 - x1_gt[sl]) ** 2).mean())
                               if x1_a2 is not None else float("nan"),
                               score_err=float(sc_err[bi]) if not skip_alg2 else float("nan"),
                               score_err_g2=float(sc_err_g2[bi]) if not skip_alg2 else float("nan"),
                               x1hat_err=float(x1h_err[bi]) if not skip_alg2 else float("nan"),
                               x1hat_err_g2=float(x1h_err_g2[bi]) if not skip_alg2 else float("nan"),
                               gamma2_meas=g2_meas)
                    if inp:
                        m = stage_mask[task][bi]
                        for tag, mm in (("obs", m), ("miss", 1.0 - m)):
                            row[f"mse_alg2_{tag}"] = (mse_masked(x1_a2, x1_gt[sl], mm)
                                                      if x1_a2 is not None else float("nan"))
                    rows.append(row)
        print(f"[stage {si}] done ({h}px, {steps_si} steps) [{time.time()-t0:.0f}s]",
              flush=True)

    # ── save raw ────────────────────────────────────────────────────────────
    import csv
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in (
        "task", "image", "stage", "step", "tau"), k))
    with open(os.path.join(out, "alg2_mse.csv"), "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=fields)
        wcsv.writeheader()
        wcsv.writerows(rows)
    json.dump({"note": "gamma2_meas is task-independent (no A/y in v or d_exact); "
                       "keyed stage -> tau -> mean_i ||v - d_exact||^2 / n",
               "table": gamma2_tab},
              open(os.path.join(out, "gamma2_meas.json"), "w"), indent=1)
    print(f"[done] {len(rows)} rows", flush=True)

    # ── figures: per-task global-t axis (stages concatenated) ───────────────
    def gx(r):
        return r["stage"] + r["tau"]

    for task in args.tasks:
        tr = [r for r in rows if r["task"] == task]
        inp = "inpainting" in task
        nrow = 2 if inp else 1
        fig, axes = plt.subplots(nrow, 1, figsize=(10, 3.6 * nrow), squeeze=False)
        ax = axes[0][0]
        for est, col in (("alg2", "tab:green"),):
            for name in shorts:
                rr = sorted((r for r in tr if r["image"] == name), key=gx)
                ax.semilogy([gx(r) for r in rr], [r[f"mse_{est}"] for r in rr],
                            color=col, alpha=0.18, lw=0.7)
            xs = sorted({round(gx(r), 6) for r in tr})
            mean = [np.nanmean([r[f"mse_{est}"] for r in tr if round(gx(r), 6) == x])
                    for x in xs]
            ax.semilogy(xs, mean, "-", color=col, lw=2, label=r"Alg2 $\hat{x}_1$")
        for bdy in (1, 2, 3):
            ax.axvline(bdy, color="k", lw=0.8, ls="--", alpha=0.5)
        ax.set_xlabel("global $t$ (stage + within-stage $t$)")
        ax.set_ylabel(r"MSE vs GT $x_1^k$")
        ax.set_title(f"{task} — one-step $\\hat{{x}}_1$ MSE (LPIPS_king; bold=mean, thin=per image)")
        ax.grid(alpha=0.3); ax.legend(fontsize=9)
        if inp:
            ax2 = axes[1][0]
            for est, col in (("alg2", "tab:green"),):
                xs = sorted({round(gx(r), 6) for r in tr})
                for tag, ls in (("obs", "-"), ("miss", ":")):
                    mean = [np.nanmean([r[f"mse_{est}_{tag}"] for r in tr
                                        if round(gx(r), 6) == x]) for x in xs]
                    ax2.semilogy(xs, mean, ls, color=col, lw=1.8,
                                 label=f"{est} {tag}")
            for bdy in (1, 2, 3):
                ax2.axvline(bdy, color="k", lw=0.8, ls="--", alpha=0.5)
            ax2.set_xlabel("global $t$"); ax2.set_ylabel("region MSE (mean)")
            ax2.set_title("observed (solid) vs missing (dotted)")
            ax2.grid(alpha=0.3); ax2.legend(fontsize=8, ncol=3)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"alg2_{task}.png"), dpi=150)
        plt.close(fig)

    # score error + gamma2 (task-independent -> one figure)
    tr = [r for r in rows if r["task"] == args.tasks[0]]
    xs = sorted({round(gx(r), 6) for r in tr})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.4))
    for key, lab, col in (("score_err", r"$\gamma^2$=0", "tab:green"),
                          ("score_err_g2", r"$\gamma^2=\gamma^2_{meas}$", "tab:red")):
        mean = [np.nanmean([r[key] for r in tr if round(gx(r), 6) == x]) for x in xs]
        a1.semilogy(xs, mean, "-", color=col, lw=1.8, label=lab)
    a1.set_title(r"score solve error $\|\hat{x}_0-x_0\|^2/n$"); a1.legend(fontsize=9)
    g2m = [np.nanmean([r["gamma2_meas"] for r in tr if round(gx(r), 6) == x]) for x in xs]
    a2.semilogy(xs, g2m, "-", color="tab:purple", lw=1.8)
    a2.set_title(r"$\gamma^2_{meas}$ (eq. 56, task-independent)")
    for ax in (a1, a2):
        for bdy in (1, 2, 3):
            ax.axvline(bdy, color="k", lw=0.8, ls="--", alpha=0.5)
        ax.set_xlabel("global $t$"); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "score_gamma2.png"), dpi=150)
    plt.close(fig)
    print("[done] figures saved ->", out, flush=True)


# ═════════════════════════════ mode: full_ip ═══════════════════════════════
def run_full_ip(args):
    out = _resolve_out(args.out)
    frames_dir = os.path.join(out, "frames_tmp")
    os.makedirs(frames_dir, exist_ok=True)
    device = "cuda:0"

    demos_all = load_demo_images(resolution=256, demo_dir=PATHS["demo_dir"])
    by_short = {d["short_name"]: d for d in demos_all}
    demos = [by_short[s] for s in args.images]
    tasks = list(args.tasks)
    if args.smoke:
        tasks, demos = tasks[:1], [by_short[TRAJ_IMAGE]]

    config = OmegaConf.load(os.path.join(PATHS["model_dir"], "config.yaml"))
    model = _load_model(config, device)
    model.register_forward_pre_hook(count_nfe_hook)
    print("[setup] model loaded", flush=True)

    gamma2_tab = json.load(open(PATHS["gamma2_table"]))["table"]

    all_rows, final_rows, nfe_stats, trajs = [], [], {}, {}
    t0 = time.time()
    for task in tasks:
        op_cfg = TASKS_SETUP[task]["operator"]
        kw = task_kw(task)
        if args.smoke:
            kw["num_langevin"] = 2
        sigma_n = float(TASKS_SETUP[task]["sigma_n"])
        for d in demos:
            name = d["short_name"]
            op, mask, y, _, _, make_Ak_fns_fn, _ = build_setup_and_measurement(
                task, op_cfg, d, sigma_n, 256, device)
            if task in ("gaussian_blur", "motion_blur"):
                inner = make_Ak_fns_fn

                def make_Ak_fns_fn(operator, y_, ss, dev, _inner=inner):
                    Af, _ = _inner(operator, y_, ss, dev)
                    return Af, make_exact_AT(Af, tuple(ss))
            gt = d["gt"].unsqueeze(0).to(device)
            record = (name == TRAJ_IMAGE)

            NFE["n"] = 0
            ts = time.time()
            x1_final, rows, traj = run_posterior_sampling_alg2(
                model, config, gt, y, op, sigma_n, device,
                gamma2_tab=gamma2_tab, make_Ak_fns_fn=make_Ak_fns_fn,
                seed=int(kw.get("seed", ALG["seed"])), record_trajectory=record,
                h0=ALG["h0"],
                sigma_min=ALG["sigma_min"], ridge_rel=ALG["ridge_rel"],
                cg_max_iter_l14=ALG["cg_max_iter_l14"],
                terminal_replace_weight=TASKS_SETUP[task]["terminal_replace_weight"],
                **{k_: v for k_, v in kw.items()
                   if k_ not in ("class_label", "seed")},
                class_label=int(d["class_idx"]))
            for r in rows:
                r.update(task=task, image=name)
            all_rows += rows
            # rows/traj are recorded pre-projection inside the sampler, so the
            # terminal projection is invisible there; record the returned
            # (post-projection) reconstruction separately.
            trw = float(TASKS_SETUP[task]["terminal_replace_weight"])
            err_f = (x1_final - gt) ** 2
            fin = dict(task=task, image=name, trw=trw,
                       pre_mse=rows[-1]["mse_x1"], post_mse=float(err_f.mean()))
            if trw > 0:                       # inpainting: split by the mask
                hole = (1.0 - mask).to(device)
                fin["post_hole"] = float((err_f * hole).sum() / (hole.sum() * 3))
                fin["post_obs"] = float((err_f * (1 - hole)).sum()
                                        / (((1 - hole)).sum() * 3))
            final_rows.append(fin)
            nfe_stats[f"{task}/{name}"] = NFE["n"]
            if record:
                trajs[task] = traj
            print(f"[{task}/{name}] NFE={NFE['n']} "
                  f"[{time.time()-ts:.0f}s tot {time.time()-t0:.0f}s]", flush=True)

    json.dump(nfe_stats, open(os.path.join(out, "nfe.json"), "w"), indent=1)
    import csv
    if all_rows:
        with open(os.path.join(out, "full_ip_metrics.csv"), "w", newline="") as f:
            wcsv = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            wcsv.writeheader(); wcsv.writerows(all_rows)
    if final_rows:
        keys = sorted({k for r in final_rows for k in r})
        with open(os.path.join(out, "full_ip_final.csv"), "w", newline="") as f:
            wcsv = csv.DictWriter(f, fieldnames=keys, restval="")
            wcsv.writeheader(); wcsv.writerows(final_rows)
    print(f"[done-sampling] rows={len(all_rows)} final={len(final_rows)}", flush=True)

    # ── loss curves: per task panel, mean over images with traj rows ──
    if all_rows and not args.smoke:
        fig, axes = plt.subplots(1, len(tasks), figsize=(4.0 * len(tasks), 3.6))
        axes = np.atleast_1d(axes)
        for ti, task in enumerate(tasks):
            ax = axes[ti]
            rs = [r for r in all_rows if r["task"] == task]
            if rs:
                xs = sorted({r["stage"] + r["tau"] for r in rs})
                mean = [np.mean([r["mse_x1"] for r in rs
                                 if r["stage"] + r["tau"] == x]) for x in xs]
                ax.semilogy(xs, mean, "-", color="tab:green", lw=1.8, label="Alg2")
            for bdy in (1, 2, 3):
                ax.axvline(bdy, color="k", lw=0.7, ls="--", alpha=0.5)
            ax.set_title(task, fontsize=10)
            ax.set_xlabel("global $t$")
            ax.grid(alpha=0.3)
            if ti == 0:
                ax.set_ylabel(r"MSE$(\hat{x}_1^k, x_1^k)$")
                ax.legend(fontsize=8)
        fig.suptitle("Algorithm 2 full sampling — loss vs time "
                     "(best configs; per PDF, $h_0$=0.1, measured $\\gamma^2$)",
                     fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(os.path.join(out, "loss_curves.png"), dpi=160)
        plt.close(fig)

    # ── frames: per task x step, one Alg2 row, cols = x_tau | x1 | x0_hat ──
    col_labels = [r"$x_\tau^k$", r"$\hat{x}_1^k$", r"$\hat{x}_0^k$"]
    frame_idx = 0
    for task in tasks:
        if task not in trajs:
            continue
        for stp, tr in enumerate(trajs[task]):
            fig, axes = plt.subplots(1, 3, figsize=(3 * 2.2, 2.7))
            for ci in range(3):
                ax = axes[ci]
                ax.imshow(to_img(tr[ci].to(device)))
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(col_labels[ci], fontsize=10)
            fig.suptitle(f"Alg2 · {task} · {TRAJ_IMAGE} · step {stp + 1}/{len(trajs[task])}",
                         fontsize=11)
            fig.tight_layout(rect=[0, 0, 1, 0.93])
            fig.savefig(os.path.join(frames_dir, f"frame_{frame_idx:04d}.png"), dpi=110)
            plt.close(fig)
            frame_idx += 1
    print(f"[done] {frame_idx} frames -> {frames_dir}", flush=True)


# ═════════════════════════════ mode: debug_box ═════════════════════════════
def run_debug_box(args):
    out = _resolve_out(args.out)
    os.makedirs(out, exist_ok=True)
    device = "cuda:0"
    task = "box_inpainting"

    demos_all = load_demo_images(resolution=256, demo_dir=PATHS["demo_dir"])
    demo = {d["short_name"]: d for d in demos_all}[args.image]
    gt = demo["gt"].unsqueeze(0).to(device)

    config = OmegaConf.load(os.path.join(PATHS["model_dir"], "config.yaml"))
    model = _load_model(config, device)
    print("[setup] model loaded", flush=True)

    op_cfg = TASKS_SETUP[task]["operator"]
    kw = task_kw(task)
    kw["class_label"] = int(demo["class_idx"])
    sigma_n = float(TASKS_SETUP[task]["sigma_n"])
    op, mask, y, _, _, make_Ak_fns_fn, _ = build_setup_and_measurement(
        task, op_cfg, demo, sigma_n, 256, device)
    gamma2_tab = json.load(open(PATHS["gamma2_table"]))["table"]
    num_stages = int(config.scheduler.num_stages)
    pyr = base.gt_stage_pyramid(gt, num_stages)
    hole = (1.0 - mask).to(device)          # [1,1,256,256]; 1 inside the box

    trw_task = float(TASKS_SETUP[task]["terminal_replace_weight"])
    if args.trw_values is not None:
        variants = [(f"trw={w:g}", ALG["h0"], float(w)) for w in args.trw_values]
    else:
        variants = [(f"h0={h}", h, trw_task) for h in args.h0]
    all_rows = []
    finals = {}
    final_rows = []
    for label, h0, trw in variants:
        x1, rows, traj = run_posterior_sampling_alg2(
            model, config, gt, y, op, sigma_n, device,
            gamma2_tab=gamma2_tab, make_Ak_fns_fn=make_Ak_fns_fn,
            seed=ALG["seed"], record_trajectory=True, h0=h0,
            sigma_min=ALG["sigma_min"], ridge_rel=ALG["ridge_rel"],
            cg_max_iter_l14=ALG["cg_max_iter_l14"],
            terminal_replace_weight=trw,
            **{k: v for k, v in kw.items() if k not in ("class_label", "seed")},
            class_label=int(demo["class_idx"]))
        # traj entries follow the same (stage, step) order as rows
        for r, (x_tau_rec, x1_rec, _) in zip(rows, traj):
            k = r["stage"]
            x1k = x1_rec.unsqueeze(0).to(device)
            h, w = x1k.shape[-2:]
            m_k = F.interpolate(hole, size=(h, w), mode="nearest")
            gt_k = pyr[k]
            err = (x1k - gt_k) ** 2
            r2 = dict(r)
            r2["variant"] = label
            r2["h0"] = h0
            r2["trw"] = trw
            r2["mse_hole"] = float((err * m_k).sum() / (m_k.sum() * 3))
            r2["mse_obs"] = float((err * (1 - m_k)).sum() / ((1 - m_k).sum() * 3))
            all_rows.append(r2)
        finals[label] = x1.detach().cpu()
        hm = [r for r in all_rows if r["variant"] == label]
        # returned x1 is post-projection; rows above are pre-projection
        err_f = (x1 - gt) ** 2
        post_hole = float((err_f * hole).sum() / (hole.sum() * 3))
        post_obs = float((err_f * (1 - hole)).sum() / ((1 - hole).sum() * 3))
        final_rows.append(dict(variant=label, trw=trw,
                               pre_hole=hm[-1]["mse_hole"], pre_obs=hm[-1]["mse_obs"],
                               post_hole=post_hole, post_obs=post_obs))
        print(f"[{label}] final-stage last-step hole MSE = {hm[-1]['mse_hole']:.4f} "
              f"obs = {hm[-1]['mse_obs']:.4f}", flush=True)
        print(f"[{label}] post-projection final hole MSE = {post_hole:.4f} "
              f"obs = {post_obs:.4f}", flush=True)

    # ── raw CSV ──
    import csv
    with open(os.path.join(out, "h0_sweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader(); w.writerows(all_rows)
    with open(os.path.join(out, "final_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(final_rows[0].keys()))
        w.writeheader(); w.writerows(final_rows)

    # ── hole/obs MSE curves (global step axis, one line per variant) ──
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for label, *_ in variants:
        rs = [r for r in all_rows if r["variant"] == label]
        xs = list(range(len(rs)))
        axes[0].semilogy(xs, [r["mse_hole"] for r in rs], label=label)
        axes[1].semilogy(xs, [r["mse_obs"] for r in rs], label=label)
    for ax, ttl in zip(axes, ["HOLE region MSE", "OBSERVED region MSE"]):
        for b in (10, 20, 30):
            ax.axvline(b, color="gray", lw=0.6, alpha=0.5)
        ax.set_title(f"box·{args.image} · {ttl}"); ax.set_xlabel("global step (4 stages x 10)")
        ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "h0_sweep_mse.png"), dpi=150)
    plt.close(fig)

    # ── final x1 comparison panel ──
    n = len(variants)
    fig, axes = plt.subplots(1, n + 2, figsize=(3.2 * (n + 2), 3.4))
    axes[0].imshow(((gt[0].cpu().permute(1, 2, 0) + 1) / 2).clamp(0, 1))
    axes[0].set_title("GT"); axes[0].axis("off")
    axes[1].imshow(((y[0].cpu().permute(1, 2, 0) + 1) / 2).clamp(0, 1))
    axes[1].set_title("y (observed)"); axes[1].axis("off")
    for i, (label, *_) in enumerate(variants):
        axes[i + 2].imshow(((finals[label][0].permute(1, 2, 0) + 1) / 2).clamp(0, 1))
        fr = final_rows[i]
        axes[i + 2].set_title(f"{label}\npost hole {fr['post_hole']:.3f} "
                              f"obs {fr['post_obs']:.4f}", fontsize=9)
        axes[i + 2].axis("off")
    fig.suptitle(f"Alg2 final x1 — box_inpainting · {args.image} · sweep", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "h0_sweep_final_x1.png"), dpi=150)
    plt.close(fig)
    print(f"[done] -> {out}", flush=True)


# ═════════════════════════════ mode: verify ════════════════════════════════
def run_verify(args):
    """CPU component audit V1-V7 (see module docstring). Dense float64."""
    torch.set_default_dtype(torch.float64)
    RES = int(args.res)
    N_PIX = RES * RES
    STAGES = [(0, 0.0, 0.25), (1, 0.142857, 0.5), (2, 0.333333, 0.75), (3, 0.6, 1.0)]

    def dense(op, n=N_PIX, shape=(1, 1, RES, RES)):
        M = np.zeros((n, n))
        for j in range(n):
            e = torch.zeros(shape)
            e.view(-1)[j] = 1.0
            M[:, j] = op(e).reshape(-1).numpy()
        return M

    checks = {"ok": True}

    def report(name, val, tol, unit="max|Δ|"):
        ok = val < tol
        checks["ok"] &= ok
        print(f"  [{name:<46}] {unit} = {val:.3e}  {'PASS' if ok else '** FAIL **'} (tol {tol:g})")

    print("== V1: self-adjointness of G / H_tau / B / N (dense, float64) ==")
    for si, s_k, e_k in STAGES:
        G = dense(lambda x: apply_G(x, stage_idx=si))
        report(f"G symmetry (stage {si})", np.abs(G - G.T).max(), 1e-12)
    si, s_k, e_k, tau = 1, 0.142857, 0.5, 0.4
    H = dense(lambda x: apply_H_tau(x, tau, s_k, e_k, 1))
    Bm = dense(lambda x: apply_B(x, s_k, e_k, 1))
    Nm = dense(lambda x: apply_N(x, s_k, e_k, 1))
    report("H_tau symmetry", np.abs(H - H.T).max(), 1e-12)
    report("B symmetry", np.abs(Bm - Bm.T).max(), 1e-12)
    report("N symmetry", np.abs(Nm - Nm.T).max(), 1e-12)
    HH = dense(lambda x: apply_H_tau(apply_H_tau(x, tau, s_k, e_k, 1), tau, s_k, e_k, 1))
    NN = dense(lambda x: apply_N(apply_N(x, s_k, e_k, 1), s_k, e_k, 1))
    report("nested H∘H  vs  H^T H", np.abs(HH - H.T @ H).max(), 1e-12)
    report("nested N∘N  vs  N^T N", np.abs(NN - Nm.T @ Nm).max(), 1e-12)

    print("== V2: compute_sigma_tau vs paper l.5 ==")
    err = max(abs(compute_sigma_tau(t, s, e) - ((1 - t) * (1 - s) + t * (1 - e)))
              for _, s, e in STAGES for t in np.linspace(0, 1, 11))
    report("sigma_tau formula", err, 1e-15)

    print("== V3: make_velocity_fn CFG semantics (stub model) ==")

    class Stub(torch.nn.Module):
        def forward(self, x, timestep=None, class_labels=None,
                    latent_size=None, pos_embed=None):
            half = x.shape[0] // 2 if x.shape[0] > 1 else x.shape[0]
            outv = x.clone()
            outv[:half] = 1.0
            outv[half:] = 3.0
            return outv

    for stage in range(4):
        base_scale = 2.0
        vfn = make_velocity_fn(Stub(), torch.tensor(500.0), None, None, None,
                               True, base_scale, stage)
        got = float(vfn(torch.zeros(1, 1, 4, 4))[0, 0, 0, 0])
        want = 1.0 + class_guidance_scale(base_scale, stage) * (3.0 - 1.0)
        report(f"CFG stage {stage}: u + scale*(c-u)", abs(got - want), 1e-12)

    print("== V4: adjoint tests Ak/ATk ==")
    g = torch.Generator().manual_seed(0)

    class MaskOp:
        def __init__(self, mask):
            self.mask = mask

        def get_mask(self, x=None):
            return self.mask

    mask = (torch.rand(1, 1, RES, RES, generator=g) > 0.3).double()
    y_probe = torch.randn(1, 1, RES, RES, generator=g)
    for stage_res in (RES // 4, RES // 2, RES):
        Ak, ATk = make_Ak_fns(MaskOp(mask), y_probe, (1, 1, stage_res, stage_res), "cpu")
        u = torch.randn(1, 1, stage_res, stage_res, generator=g)
        w = torch.randn(1, 1, RES, RES, generator=g)
        gap = abs(float((Ak(u) * w).sum()) - float((u * ATk(w)).sum()))
        report(f"make_Ak_fns adjoint (stage res {stage_res})", gap, 1e-10)

    print("== V5: M0 closure vs dense (1/eta^2)A^T A + (1/s^2)H^T H ==")
    eta, sigma = 0.05, 0.6
    Ak, ATk = make_Ak_fns(MaskOp(mask), y_probe, (1, 1, RES, RES), "cpu")

    def M0(x):
        return (1 / eta ** 2) * ATk(Ak(x)) + (1 / sigma ** 2) * apply_H_tau(
            apply_H_tau(x, tau, s_k, e_k, 1), tau, s_k, e_k, 1)
    Mdense = dense(M0)
    A = dense(Ak)
    Mref = (1 / eta ** 2) * A.T @ A + (1 / sigma ** 2) * H.T @ H
    report("M0 vs explicit-transpose reference",
           np.abs(Mdense - Mref).max() / np.abs(Mref).max(), 1e-12, "rel")

    print("== V6: power_iter_norm vs exact lambda_max ==")
    lam_true = float(np.linalg.eigvalsh(Mdense).max())
    lam_est = power_iter_norm(M0, (1, 1, RES, RES), "cpu", iters=20)
    rel = abs(lam_est - lam_true) / lam_true
    print(f"  lambda_max exact {lam_true:.4f}  power-iter(20) {lam_est:.4f}  rel err {rel:.2e}")
    report("power_iter_norm(20) within 5%", rel, 5e-2, "rel")

    print("== V7: score_solve — N∘N form vs explicit N^T N dense solve; S1 ==")
    x1_true = torch.randn(1, 1, RES, RES, generator=g)
    x0_true = torch.randn(1, 1, RES, RES, generator=g)
    sigma_tau = compute_sigma_tau(tau, s_k, e_k)   # NOT V5's test constant
    x_tau = apply_H_tau(x1_true, tau, s_k, e_k, 1) + sigma_tau * x0_true
    d_exact = apply_B(x1_true, s_k, e_k, 1) - (e_k - s_k) * x0_true
    for g2 in (0.0, 0.02):
        x0_cg = score_solve(x_tau, d_exact, s_k, e_k, tau, g2, 1, 1e-12, 4000)
        rhs = (Nm.T @ (Bm @ x_tau.reshape(-1).numpy()
                       - H @ d_exact.reshape(-1).numpy()))
        lhs = Nm.T @ Nm + g2 * (H.T @ H)
        x0_dense = np.linalg.solve(lhs, rhs)
        gap = np.abs(x0_cg.reshape(-1).numpy() - x0_dense).max()
        report(f"CG(N∘N) vs dense(N^T N), gamma2={g2}", gap, 1e-6)
    x0_s1 = score_solve(x_tau, d_exact, s_k, e_k, tau, 0.0, 1, 1e-12, 4000)
    report("S1 identity: x0_hat == x0 (v=d_exact, g2=0)",
           float((x0_s1 - x0_true).abs().max()), 1e-6)

    print("\n" + ("ALL CHECKS PASSED" if checks["ok"] else "** SOME CHECKS FAILED **"))
    return 0 if checks["ok"] else 1


# ═════════════════════════════ dispatcher ══════════════════════════════════
RUNNERS = {"onestep": run_onestep, "full_ip": run_full_ip,
           "debug_box": run_debug_box, "verify": run_verify}

# Exact key contract for config.json. Everything main() consumes must be listed
# here, so a typo or a dropped key fails loudly instead of being swallowed by a
# .get() default or by the sampler's **unused_kw.
CONFIG_SCHEMA = {
    "mode": None,
    "paths": {"model_dir", "demo_dir", "gamma2_table"},
    "algorithm": {"sigma_min", "h0", "seed", "measurement_seed",
                  "ridge_rel", "cg_max_iter_l14"},
    "sampler_kw": {"num_langevin", "ode_steps_per_stage", "shift", "guidance_scale",
                   "g_bypass_stage3", "cg_tol", "cg_max_iter"},
    "tasks_setup": None, "tasks": None, "images": None, "traj_image": None,
    "onestep": {"out", "chunk", "self_test"},
    "full_ip": {"out", "smoke"},
    "debug_box": {"out", "image", "h0", "trw_values"},
    "verify": {"res"},
}
TASK_KEYS = {"sigma_n", "operator", "terminal_replace_weight", "measurement_mode",
             "kw"}
# Operator kwargs are consumed by build_setup_and_measurement per task, which
# reads them with .get() — an unchecked typo there silently reverts to that
# function's own default, which is exactly what the config contract must stop.
TASK_OPERATOR_KEYS = {
    "box_inpainting": {"mask_len_range", "center"},
    "random_inpainting": {"mask_prob"},
    "gaussian_blur": {"kernel_size", "kernel_std"},
    "motion_blur": {"kernel_size", "kernel_intensity", "kernel_seed",
                    "use_daps_kernel"},
    "superresolution": {"scale_factor", "antialias"},
}


def _check_config_keys(cfg):
    missing = set(CONFIG_SCHEMA) - set(cfg)
    extra = set(cfg) - set(CONFIG_SCHEMA)
    if missing or extra:
        raise KeyError(f"config.json top level: missing={sorted(missing)} "
                       f"unknown={sorted(extra)}")
    for sect, allowed in CONFIG_SCHEMA.items():
        if allowed is None:
            continue
        got = set(cfg[sect])
        if got != allowed:
            raise KeyError(f'config.json "{sect}": missing={sorted(allowed - got)} '
                           f"unknown={sorted(got - allowed)}")
    required = {"sigma_n", "operator", "terminal_replace_weight",
                "measurement_mode"}
    for task, spec in cfg["tasks_setup"].items():
        got = set(spec)
        if not got <= TASK_KEYS or not required <= got:
            raise KeyError(f'config.json "tasks_setup"."{task}": '
                           f"missing={sorted(TASK_KEYS - got - {'kw'})} "
                           f"unknown={sorted(got - TASK_KEYS)}")
        if task not in TASK_OPERATOR_KEYS:
            raise KeyError(f'config.json "tasks_setup": unknown task {task!r} '
                           f"(known: {sorted(TASK_OPERATOR_KEYS)})")
        op_got = set(spec["operator"])
        if op_got != TASK_OPERATOR_KEYS[task]:
            raise KeyError(
                f'config.json "tasks_setup"."{task}"."operator": '
                f"missing={sorted(TASK_OPERATOR_KEYS[task] - op_got)} "
                f"unknown={sorted(op_got - TASK_OPERATOR_KEYS[task])}")
        kw_got = set(spec.get("kw", {}))          # merged over sampler_kw
        if not kw_got <= CONFIG_SCHEMA["sampler_kw"]:
            raise KeyError(f'config.json "tasks_setup"."{task}"."kw": '
                           f"unknown={sorted(kw_got - CONFIG_SCHEMA['sampler_kw'])}")
    for name in cfg["tasks"]:
        if name not in cfg["tasks_setup"]:
            raise KeyError(f'config.json "tasks": {name!r} has no tasks_setup entry')
    if cfg["traj_image"] not in cfg["images"]:
        raise KeyError(f'config.json "traj_image" {cfg["traj_image"]!r} '
                       "is not in images")


def _check_hash_seed(mode):
    """demo_runner's mask seeds go through Python's hash(), so the masks are
    only reproducible under PYTHONHASHSEED=0 (the documented contract). verify
    builds no masks and is exempt."""
    if mode != "verify" and os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError(
            "PYTHONHASHSEED=0 is required: demo_runner derives mask seeds from "
            f"hash(), so mode {mode!r} would draw a different mask per process. "
            "Re-run as: PYTHONHASHSEED=0 python main.py ...")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None,
                    help='single config: {"mode": ..., "<mode>": {overrides}} '
                         '(default: <Algorithm2>/config.json)')
    ap.add_argument("--mode", default=None, help="override the config's mode")
    cli = ap.parse_args()
    # Omitting --config uses this directory's config.json; ANY value passed on
    # the command line resolves against the caller's CWD and must exist there.
    # (Comparing the value to the default cannot tell "--config config.json"
    # apart from the default, so such a run silently loaded HERE/config.json.)
    if cli.config is None:
        cfg_path = os.path.join(HERE, "config.json")
    elif os.path.isabs(cli.config):
        cfg_path = cli.config
    else:
        cfg_path = os.path.join(ORIG_CWD, cli.config)
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"--config {cli.config!r} -> {cfg_path} not found")
    cfg = json.load(open(cfg_path))
    _check_config_keys(cfg)
    mode = cli.mode or cfg["mode"]

    if mode not in RUNNERS:
        raise KeyError(f"mode {mode!r} not in {sorted(RUNNERS)}")
    _check_hash_seed(mode)

    global PATHS, ALG, SAMPLER_KW, TASKS_SETUP, TRAJ_IMAGE
    TRAJ_IMAGE = cfg["traj_image"]
    subst = {"{IP_PACKAGE}": base.IP_PACKAGE, "{HERE}": HERE}
    PATHS = {k: v for k, v in cfg["paths"].items()}
    for k, v in PATHS.items():
        for token, real in subst.items():
            v = v.replace(token, real)
        PATHS[k] = resolve_path(v)
    ALG = dict(cfg["algorithm"])
    SAMPLER_KW = dict(cfg["sampler_kw"])
    TASKS_SETUP = dict(cfg["tasks_setup"])
    measurement.configure(TASKS_SETUP, ALG["measurement_seed"])

    params = dict(cfg[mode])                       # strict: no code defaults
    if mode in ("onestep", "full_ip"):
        params.setdefault("tasks", cfg["tasks"])
        params.setdefault("images", cfg["images"])
    print(f"[main] mode={mode}  config={cfg_path}\n[main] paths={PATHS}\n"
          f"[main] algorithm={ALG}\n[main] params={params}", flush=True)
    return RUNNERS[mode](argparse.Namespace(**params))


if __name__ == "__main__":
    raise SystemExit(main())
