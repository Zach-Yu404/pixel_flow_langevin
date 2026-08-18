#!/usr/bin/env python
"""Algorithm 2 experiments — single entry point.

    python main.py --config <mode>.json

Modes (JSON "mode" field; remaining keys override the mode's defaults below):
  onestep   : experiment 2 — one-step Alg2/WLS/Model x1-hat MSE vs t
              (5 tasks x 7 images x 4 stages; sanity S1/S2 + adjoint tests first;
              {"self_test": true} runs only the CPU sanity block)
  full_ip   : experiment 3 — full inverse-problem sampling, 3 methods
              (Alg1-WLS / Alg1-Model / Alg2), metrics + loss curves + frames
  debug_box : box-hole sweep harness (h0 / anchors / x0_steps x gamma2_scales;
              obs/hole split MSE) — task debug-box-alg2-hole
  verify    : CPU component audit V1-V8 (dense float64): self-adjointness of
              G/H/B/N (transpose equivalence), sigma_tau, CFG semantics,
              Ak/ATk adjoints, M0 vs dense, power_iter_norm, score_solve vs
              explicit N^T N, gamma2 sensitivity probe

GPU submission (no sbatch file needed):
  sbatch -A cbig-ece -p gpu --gres=gpu:a100:1 -c 8 --mem=64G -t 3:00:00 \\
    -o ../logs/%x-%j.out --export=ALL,PYTHONHASHSEED=0 \\
    --wrap "python main.py --config full_ip.json"
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
    HERE, SIGMA_MIN, TASKS, TRAJ_IMAGE, METHODS, NFE, count_nfe_hook,
    apply_B, apply_N, power_iter_norm, make_exact_AT, adjoint_test, mse_masked,
    score_solve, clean_image_solve, build_task_setups, best_cfg,
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

from ms_posterior_sampling_article_version_final import run_posterior_sampling  # noqa: E402
from ms_posterior_sampling_article_version_final_utils import (  # noqa: E402
    apply_G, apply_H_tau, compute_sigma_tau,
    direct_estimate_x1, wls_estimate_x1, make_velocity_fn, make_Ak_fns,
)
from ms_posterior_sampling_utils import class_guidance_scale   # noqa: E402
from pixelflow.scheduling_pixelflow import PixelFlowScheduler  # noqa: E402
from pixelflow.utils import config as config_utils             # noqa: E402
from demo_runner import build_setup_and_measurement, load_demo_images  # noqa: E402
from onestep_visual import to_img                              # noqa: E402

MODE_DEFAULTS = {
    "onestep": dict(out="results/onestep", tasks=TASKS,
                    images=base.PLAYGROUND_IMAGES, chunk=4, self_test=False),
    "full_ip": dict(out="results/full_ip", tasks=TASKS,
                    images=base.PLAYGROUND_IMAGES, smoke=False),
    "debug_box": dict(out="results/debug_box", image="junco",
                      h0=[0.1, 0.5, 1.0], anchors=None, x0_steps=None,
                      gamma2_scales=[1.0]),
    "verify": dict(),
}


def _resolve_out(out):
    return out if os.path.isabs(out) else os.path.join(ORIG_CWD, out)


def _load_model(config, device):
    model_dir = os.path.join(base.IP_PACKAGE, "pretrained_models", "c2img")
    model = config_utils.instantiate_from_config(config.model).to(device)
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

    demos_all = load_demo_images(resolution=256, demo_dir=os.path.join(base.IP_PACKAGE, "demo"))
    by_short = {d["short_name"]: d for d in demos_all}
    demos = [by_short[s] for s in args.images]
    shorts = [d["short_name"] for d in demos]
    classes = [int(d["class_idx"]) for d in demos]
    gt = torch.stack([d["gt"] for d in demos], dim=0).to(device)
    B = gt.shape[0]

    model_dir = os.path.join(base.IP_PACKAGE, "pretrained_models", "c2img")
    config = OmegaConf.load(os.path.join(model_dir, "config.yaml"))
    num_stages = int(config.scheduler.num_stages)
    scheduler = PixelFlowScheduler(config.scheduler.num_train_timesteps,
                                   num_stages=num_stages, gamma=-1 / 3)
    pyr = base.gt_stage_pyramid(gt, num_stages)

    # kw consumed for the shared grid/velocity (identical across the 5 tasks —
    # verified bitwise in experiment 1); per-task kw re-read for WLS params anyway.
    kw0 = json.load(open(base.find_lpips_king_config("box_inpainting")))["kw"]
    shift = float(kw0.get("shift", 1.0))
    guidance_scale = float(kw0.get("guidance_scale", 0.0))
    do_cfg = guidance_scale > 0
    g_bypass = bool(kw0.get("g_bypass_stage3", True))
    cg_tol = float(kw0.get("cg_tol", 1e-5))
    cg_max_iter = int(kw0.get("cg_max_iter", 50))

    # ── per-task measurement setups (playground-identical) ──
    task_setups = {}
    for task in args.tasks:
        kw_t, sigma_n_t, setups = build_task_setups(task, demos, device)
        task_setups[task] = (kw_t, sigma_n_t, setups)
        print(f"[setup] {task}: eta={sigma_n_t}  y[0] shape={tuple(setups[0]['y'].shape)}",
              flush=True)

    # ── sanity + adjoint tests ──────────────────────────────────────────────
    sanity = {"adjoint": {}, "S1": {}, "S2": {}}
    for task in args.tasks:
        _, _, setups = task_setups[task]
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
        if sig < SIGMA_MIN:
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
        _, sigma_n_t, setups = task_setups[task]
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
            if sig < SIGMA_MIN:
                tau = float(sc.t[2]); sig = compute_sigma_tau(tau, sk, ek)
            x_tau = apply_H_tau(x1_gt, tau, sk, ek, eff_si) + sig * x0
            y_clean = st["op"](gt[:1]).detach()          # xi = 0
            A_fn, AT_fn = st["mkA"](st["op"], st["y"], x1_gt.shape, device)
            x1_hat = clean_image_solve(x_tau, A_fn, AT_fn, y_clean, sigma_n_t, sig,
                                       tau, sk, ek, eff_si, x1_gt.clone(), cg_tol)
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
                       for s in task_setups[t][2]] for t in args.tasks}
        stage_mask = {t: [F.interpolate(s["mask"], size=(h, w), mode="nearest")
                          for s in task_setups[t][2]] for t in args.tasks}

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

            skip_alg2 = sigma_tau < SIGMA_MIN
            if not skip_alg2:
                x0_hat = score_solve(x_tau, v, sk, ek, tau, 0.0, eff_si, cg_tol, cg_max_iter)
                x0_hat_g2 = score_solve(x_tau, v, sk, ek, tau, g2_meas, eff_si,
                                        cg_tol, cg_max_iter)
                sc_err = ((x0_hat - x0) ** 2).mean(dim=(1, 2, 3))
                sc_err_g2 = ((x0_hat_g2 - x0) ** 2).mean(dim=(1, 2, 3))

            for task in args.tasks:
                kw_t, eta, setups = task_setups[task]
                rho_s = float(base.per_stage(kw_t.get("rho_s", 1.0), si, num_stages))
                rho_e = float(base.per_stage(kw_t.get("rho_e", 1.0), si, num_stages))
                lambda_x = float(kw_t.get("lambda_x", 0.01))
                x1_wls = wls_estimate_x1(xs_hat, xe_hat, sk, ek, rho_s, rho_e,
                                         lambda_x, cg_tol, cg_max_iter, stage_idx=eff_si)
                inp = "inpainting" in task
                for bi, name in enumerate(shorts):
                    sl = slice(bi, bi + 1)
                    if skip_alg2:
                        x1_a2 = None
                    else:
                        A_fn, AT_fn = stage_A[task][bi]
                        x1_a2 = clean_image_solve(x_tau[sl], A_fn, AT_fn,
                                                  setups[bi]["y"], eta, sigma_tau, tau,
                                                  sk, ek, eff_si, x1_model[sl].clone(), cg_tol)
                    row = dict(task=task, image=name, stage=si, step=step_idx,
                               tau=tau, T=float(T), sigma_tau=float(sigma_tau),
                               s_k=sk, e_k=ek, resolution=h,
                               mse_alg2=float(((x1_a2 - x1_gt[sl]) ** 2).mean())
                               if x1_a2 is not None else float("nan"),
                               mse_wls=float(((x1_wls[sl] - x1_gt[sl]) ** 2).mean()),
                               mse_model=float(((x1_model[sl] - x1_gt[sl]) ** 2).mean()),
                               score_err=float(sc_err[bi]) if not skip_alg2 else float("nan"),
                               score_err_g2=float(sc_err_g2[bi]) if not skip_alg2 else float("nan"),
                               gamma2_meas=g2_meas)
                    if inp:
                        m = stage_mask[task][bi]
                        for tag, mm in (("obs", m), ("miss", 1.0 - m)):
                            row[f"mse_alg2_{tag}"] = (mse_masked(x1_a2, x1_gt[sl], mm)
                                                      if x1_a2 is not None else float("nan"))
                            row[f"mse_wls_{tag}"] = mse_masked(x1_wls[sl], x1_gt[sl], mm)
                            row[f"mse_model_{tag}"] = mse_masked(x1_model[sl], x1_gt[sl], mm)
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
        for est, col in (("alg2", "tab:green"), ("wls", "tab:blue"), ("model", "tab:orange")):
            for name in shorts:
                rr = sorted((r for r in tr if r["image"] == name), key=gx)
                ax.semilogy([gx(r) for r in rr], [r[f"mse_{est}"] for r in rr],
                            color=col, alpha=0.18, lw=0.7)
            xs = sorted({round(gx(r), 6) for r in tr})
            mean = [np.nanmean([r[f"mse_{est}"] for r in tr if round(gx(r), 6) == x])
                    for x in xs]
            ax.semilogy(xs, mean, "-", color=col, lw=2,
                        label={"alg2": r"Alg2 $\hat{x}_1$", "wls": r"WLS $\hat{x}_1$",
                               "model": r"Model $\hat{x}_1$"}[est])
        for bdy in (1, 2, 3):
            ax.axvline(bdy, color="k", lw=0.8, ls="--", alpha=0.5)
        ax.set_xlabel("global $t$ (stage + within-stage $t$)")
        ax.set_ylabel(r"MSE vs GT $x_1^k$")
        ax.set_title(f"{task} — one-step $\\hat{{x}}_1$ MSE (LPIPS_king; bold=mean, thin=per image)")
        ax.grid(alpha=0.3); ax.legend(fontsize=9)
        if inp:
            ax2 = axes[1][0]
            for est, col in (("alg2", "tab:green"), ("wls", "tab:blue"), ("model", "tab:orange")):
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

    demos_all = load_demo_images(resolution=256,
                                 demo_dir=os.path.join(base.IP_PACKAGE, "demo"))
    by_short = {d["short_name"]: d for d in demos_all}
    demos = [by_short[s] for s in args.images]
    tasks = list(args.tasks)
    if args.smoke:
        tasks, demos = tasks[:1], [by_short[TRAJ_IMAGE]]

    model_dir = os.path.join(base.IP_PACKAGE, "pretrained_models", "c2img")
    config = OmegaConf.load(os.path.join(model_dir, "config.yaml"))
    model = _load_model(config, device)
    model.register_forward_pre_hook(count_nfe_hook)
    print("[setup] model loaded", flush=True)

    gamma2_tab = json.load(open(os.path.join(HERE, "gamma2_meas.json")))["table"]
    num_stages = int(config.scheduler.num_stages)

    all_rows, nfe_stats, trajs = [], {}, {}
    t0 = time.time()
    for task in tasks:
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
                    return Af, make_exact_AT(Af, tuple(ss))
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

    json.dump(nfe_stats, open(os.path.join(out, "nfe.json"), "w"), indent=1)
    import csv
    if all_rows:
        with open(os.path.join(out, "full_ip_metrics.csv"), "w", newline="") as f:
            wcsv = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            wcsv.writeheader(); wcsv.writerows(all_rows)
    print(f"[done-sampling] rows={len(all_rows)}", flush=True)

    # ── loss curves: per task panel, mean over images with traj rows ──
    if all_rows and not args.smoke:
        fig, axes = plt.subplots(1, len(tasks), figsize=(4.0 * len(tasks), 3.6))
        axes = np.atleast_1d(axes)
        for ti, task in enumerate(tasks):
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
        fig.savefig(os.path.join(out, "loss_curves.png"), dpi=160)
        plt.close(fig)

    # ── ONE mp4: task segments x steps; rows=methods, cols=x_tau|x1|x0 ──
    col_labels = [r"$x_\tau^k$", r"$\hat{x}_1^k$", r"$\hat{x}_0^k$"]
    frame_idx = 0
    for task in tasks:
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


# ═════════════════════════════ mode: debug_box ═════════════════════════════
def run_debug_box(args):
    out = _resolve_out(args.out)
    os.makedirs(out, exist_ok=True)
    device = "cuda:0"
    task = "box_inpainting"

    demos_all = load_demo_images(resolution=256,
                                 demo_dir=os.path.join(base.IP_PACKAGE, "demo"))
    demo = {d["short_name"]: d for d in demos_all}[args.image]
    gt = demo["gt"].unsqueeze(0).to(device)

    model_dir = os.path.join(base.IP_PACKAGE, "pretrained_models", "c2img")
    config = OmegaConf.load(os.path.join(model_dir, "config.yaml"))
    model = _load_model(config, device)
    print("[setup] model loaded", flush=True)

    cfg, op_cfg = best_cfg(task)
    kw = dict(cfg["kw"])
    kw["class_label"] = int(demo["class_idx"])
    sigma_n = float(cfg.get("sigma_n", 0.05))
    op, mask, y, _, _, make_Ak_fns_fn, _ = build_setup_and_measurement(
        task, op_cfg, demo, sigma_n, 256, device)
    gamma2_tab = json.load(open(os.path.join(HERE, "gamma2_meas.json")))["table"]
    num_stages = int(config.scheduler.num_stages)
    pyr = base.gt_stage_pyramid(gt, num_stages)
    hole = (1.0 - mask).to(device)          # [1,1,256,256]; 1 inside the box

    if args.x0_steps is not None:
        variants = [(f"K={k},g2x{gs:g}", 0.1, 0.0, k, gs)
                    for k in args.x0_steps for gs in args.gamma2_scales]
    elif args.anchors is not None:
        variants = [(f"anchor={a}", 0.1, a, 1, 1.0) for a in args.anchors]
    else:
        variants = [(f"h0={h}", h, 0.0, 1, 1.0) for h in args.h0]
    all_rows = []
    finals = {}
    for label, h0, anchor_val, x0_steps, g2_scale in variants:
        x1, rows, traj = run_posterior_sampling_alg2(
            model, config, gt, y, op, sigma_n, device,
            gamma2_tab=gamma2_tab, make_Ak_fns_fn=make_Ak_fns_fn,
            seed=42, record_trajectory=True, h0=h0, anchor=anchor_val,
            x0_langevin_steps=x0_steps, gamma2_scale=g2_scale,
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
            r2["anchor"] = anchor_val
            r2["x0_steps"] = x0_steps
            r2["gamma2_scale"] = g2_scale
            r2["mse_hole"] = float((err * m_k).sum() / (m_k.sum() * 3))
            r2["mse_obs"] = float((err * (1 - m_k)).sum() / ((1 - m_k).sum() * 3))
            all_rows.append(r2)
        finals[label] = x1.detach().cpu()
        hm = [r for r in all_rows if r["variant"] == label]
        print(f"[{label}] final-stage last-step hole MSE = {hm[-1]['mse_hole']:.4f} "
              f"obs = {hm[-1]['mse_obs']:.4f}", flush=True)

    # ── raw CSV ──
    import csv
    with open(os.path.join(out, "h0_sweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader(); w.writerows(all_rows)

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
    fig, axes = plt.subplots(1, n + 1, figsize=(3.2 * (n + 1), 3.4))
    axes[0].imshow(((gt[0].cpu().permute(1, 2, 0) + 1) / 2).clamp(0, 1))
    axes[0].set_title("GT"); axes[0].axis("off")
    for i, (label, *_) in enumerate(variants):
        axes[i + 1].imshow(((finals[label][0].permute(1, 2, 0) + 1) / 2).clamp(0, 1))
        last = [r for r in all_rows if r["variant"] == label][-1]
        axes[i + 1].set_title(f"{label}\nhole MSE {last['mse_hole']:.3f}", fontsize=9)
        axes[i + 1].axis("off")
    fig.suptitle(f"Alg2 final x1 — box_inpainting · {args.image} · sweep", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "h0_sweep_final_x1.png"), dpi=150)
    plt.close(fig)
    print(f"[done] -> {out}", flush=True)


# ═════════════════════════════ mode: verify ════════════════════════════════
def run_verify(args):
    """CPU component audit V1-V8 (see module docstring). Dense float64."""
    torch.set_default_dtype(torch.float64)
    RES = 16
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

    print("== V8: gamma2 probe (white noise on v; real v-error is structured) ==")
    for gamma_true in (0.05, 0.15):
        noise = torch.randn(1, 1, RES, RES, generator=g)
        v_noisy = d_exact + gamma_true * noise
        errs = {}
        for g2 in (0.0, 1e-4, 1e-3, 1e-2, gamma_true ** 2, 4 * gamma_true ** 2, 1.0):
            x0_hat = score_solve(x_tau, v_noisy, s_k, e_k, tau, g2, 1, 1e-12, 4000)
            errs[g2] = float(((x0_hat - x0_true) ** 2).mean())
        best = min(errs, key=errs.get)
        line = "  ".join(f"g2={kk:.1e}:{vv:.4f}" for kk, vv in errs.items())
        print(f"  gamma_true={gamma_true}: {line}")
        print(f"    best gamma2 = {best:.1e}")

    print("\n" + ("ALL CHECKS PASSED" if checks["ok"] else "** SOME CHECKS FAILED **"))
    return 0 if checks["ok"] else 1


# ═════════════════════════════ dispatcher ══════════════════════════════════
RUNNERS = {"onestep": run_onestep, "full_ip": run_full_ip,
           "debug_box": run_debug_box, "verify": run_verify}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="mode JSON (see MODE_DEFAULTS)")
    cli = ap.parse_args()
    cfg_path = cli.config if os.path.isabs(cli.config) else os.path.join(ORIG_CWD, cli.config)
    if not os.path.isfile(cfg_path):
        cfg_path = os.path.join(HERE, os.path.basename(cli.config))
    cfg = json.load(open(cfg_path))
    mode = cfg.pop("mode")
    params = dict(MODE_DEFAULTS[mode])
    params.update(cfg)
    print(f"[main] mode={mode}  config={cfg_path}  params={params}", flush=True)
    return RUNNERS[mode](argparse.Namespace(**params))


if __name__ == "__main__":
    raise SystemExit(main())
