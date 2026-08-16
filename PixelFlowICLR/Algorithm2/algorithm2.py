#!/usr/bin/env python
"""Algorithm 2 one-step estimator vs WLS / Model  (PixelFlowICLR experiment 2).

Extends onestep_mse_vs_t: same tasks/images/stage grid/x_t construction (helpers
imported from it verbatim), plus per-task measurement y built by the SAME code
path as playground_runs (demo_runner.build_setup_and_measurement, per-task seed
contract; run with PYTHONHASHSEED=0 — the mask seeding still uses hash()).

Per (task, image, stage k, grid t) with sigma_tau >= 0.01:
  v = velocity_fn(x_t)                               (ONE net call; cached across
                                                      tasks — v, score solve and
                                                      gamma2 contain no A/y)
  line 11:  [N^2 + g2*H^2] x0_hat = N(B x_t - H v)   (CG; g2=0 main, g2=g2_meas aux)
  line 14:  M x1_hat = b,  M = A^T A/eta^2 + H^2/sigma^2 (+ridge at tau=0),
            b = A^T y/eta^2 + H x_t/sigma^2, warm start = direct_estimate_x1
  baselines: WLS + Model on the same x_t (same calls as onestep_mse_vs_t).

Identity behind sanity S1 (verified analytically):
  sigma_tau*B + (e-s)*H_tau == N   =>   B x_t - H d_exact = N x0,
  d_exact = B x1 - (e-s) x0, so with v:=d_exact, gamma2=0 CG must return x0.

No Langevin / iteration / rollout. playground_runs is never written to.
"""

import argparse
import copy
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # PixelFlowICLR/ for base import
ORIG_CWD = os.getcwd()

import onestep_mse_vs_t as base  # noqa: E402  (sets IP_package sys.path, chdir)

# pipeline._daps_motion_kernel hardcodes the OLD box's DAPS path (contradictions
# registry #16); pre-inserting the live path makes its dead insert harmless.
_DAPS = os.path.join(base.IP_PACKAGE, "baselines", "DAPS")
if _DAPS not in sys.path:
    sys.path.insert(0, _DAPS)

import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402
import torch.nn.functional as F                               # noqa: E402
from omegaconf import OmegaConf                               # noqa: E402
import matplotlib                                             # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                               # noqa: E402

from ms_posterior_sampling_article_version_final_utils import (  # noqa: E402
    apply_G, apply_H_tau, compute_sigma_tau, cg_solve,
    direct_estimate_x1, wls_estimate_x1, make_velocity_fn,
)
from pixelflow.scheduling_pixelflow import PixelFlowScheduler  # noqa: E402
from pixelflow.utils import config as config_utils             # noqa: E402
from demo_runner import build_setup_and_measurement, load_demo_images  # noqa: E402

SIGMA_MIN = 0.01          # skip Alg2 below this (matches sampler's Langevin skip)
TASKS = base.TASKS


# ── new operators (reuse apply_G; stage_idx = eff_si keeps g_bypass semantics) ──
def apply_B(x, s_k, e_k, stage_idx):
    """B_k x = e_k*x - s_k*G(x)  (= dH_tau/dtau)."""
    return e_k * x - s_k * apply_G(x, stage_idx=stage_idx)


def apply_N(x, s_k, e_k, stage_idx):
    """N_k x = e_k*(1-s_k)*x - s_k*(1-e_k)*G(x)."""
    return e_k * (1.0 - s_k) * x - s_k * (1.0 - e_k) * apply_G(x, stage_idx=stage_idx)


def power_iter_norm(M_fn, shape, device, iters=20, seed=0):
    """Matrix-free largest-eigenvalue estimate of SPD M (no ridge term)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    u = torch.randn(shape, generator=g).to(device)
    u = u / u.norm().clamp(min=1e-12)
    lam = torch.tensor(1.0, device=device)
    for _ in range(iters):
        Mu = M_fn(u)
        lam = Mu.norm()
        u = Mu / lam.clamp(min=1e-12)
    return float(lam)


def make_exact_AT(A_fn, x_shape):
    """Exact adjoint of a LINEAR A_fn via autograd (same pattern as
    _interpolate_adjoint). Needed for blur/motion: their analytic flip(K)
    adjoint ignores that reflection-padding's adjoint is not reflection-padding
    (boundary error ~1e-3 at 32x32)."""
    def AT(r):
        with torch.enable_grad():
            xp = torch.zeros(x_shape, device=r.device, dtype=r.dtype,
                             requires_grad=True)
            Ax = A_fn(xp)
            (grad,) = torch.autograd.grad(Ax, xp, grad_outputs=r.detach())
        return grad.detach()
    return AT


def adjoint_test(A_fn, AT_fn, x_shape, y_probe, device, seed=1):
    """|<A v, w> - <v, A^T w>| with unit-norm v,w."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    v = torch.randn(x_shape, generator=g).to(device)
    w = torch.randn_like(y_probe)
    v = v / v.norm(); w = w / w.norm()
    lhs = (A_fn(v) * w).sum()
    rhs = (v * AT_fn(w)).sum()
    return float((lhs - rhs).abs())


def mse_masked(a, b, m):
    """Mean squared error over mask m (broadcast [1,1,h,w]); NaN if empty."""
    n = m.sum() * a.shape[1]
    if float(n) == 0:
        return float("nan")
    return float(((a - b) ** 2 * m).sum() / n)


def score_solve(x_tau, v, s_k, e_k, tau, gamma2, eff_si, cg_tol, cg_max_iter):
    """line 11: [N^2 + gamma2*H_tau^2] x0_hat = N(B x_tau - H_tau v)."""
    def op(u):
        Nu = apply_N(apply_N(u, s_k, e_k, eff_si), s_k, e_k, eff_si)
        if gamma2 > 0:
            Hu = apply_H_tau(apply_H_tau(u, tau, s_k, e_k, eff_si), tau, s_k, e_k, eff_si)
            Nu = Nu + gamma2 * Hu
        return Nu
    rhs = apply_N(apply_B(x_tau, s_k, e_k, eff_si) - apply_H_tau(v, tau, s_k, e_k, eff_si),
                  s_k, e_k, eff_si)
    return cg_solve(op, rhs, tol=cg_tol, max_iter=cg_max_iter)


def alg2_x1_solve(x_tau, A_fn, AT_fn, y, eta, sigma_t, tau, s_k, e_k, eff_si,
                  x1_warm, cg_tol):
    """line 14 (deterministic): M x = b, warm-started CG, max_iter=200."""
    inv_e2, inv_s2 = 1.0 / eta ** 2, 1.0 / float(sigma_t) ** 2

    def M0(x):
        return inv_e2 * AT_fn(A_fn(x)) + \
            inv_s2 * apply_H_tau(apply_H_tau(x, tau, s_k, e_k, eff_si), tau, s_k, e_k, eff_si)

    ridge = 0.0
    if tau == 0.0:
        ridge = 1e-6 * power_iter_norm(M0, x_tau.shape, x_tau.device)
    M_fn = (lambda x: M0(x) + ridge * x) if ridge else M0
    b = inv_e2 * AT_fn(y) + inv_s2 * apply_H_tau(x_tau, tau, s_k, e_k, eff_si)
    return cg_solve(M_fn, b, x0=x1_warm, tol=cg_tol, max_iter=200)


# ── measurement setup per (task, image): playground-identical code path ──────
def build_task_setups(task, demos, device):
    cfg = json.load(open(base.find_lpips_king_config(task)))
    sigma_n = float(cfg["sigma_n"])
    setups = []
    for d in demos:
        op, mask, y, _, _, mkA, _ = build_setup_and_measurement(
            task, cfg["operator"], d, sigma_n, 256, device)
        if task in ("gaussian_blur", "motion_blur"):
            # replace the analytic flip(K) adjoint with the exact autograd one
            inner = mkA

            def mkA(operator, y_, stage_shape, device_, _inner=inner):
                A_fn, _ = _inner(operator, y_, stage_shape, device_)
                return A_fn, make_exact_AT(A_fn, tuple(stage_shape))
        setups.append(dict(op=op, mask=mask.to(device).float(), y=y, mkA=mkA))
    return cfg["kw"], sigma_n, setups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    ap.add_argument("--images", nargs="*", default=base.PLAYGROUND_IMAGES)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--self_test", action="store_true",
                    help="CPU S1/S2 + adjoint tests, no model, then exit")
    args = ap.parse_args()
    if not os.path.isabs(args.out):
        args.out = os.path.join(ORIG_CWD, args.out)
    os.makedirs(args.out, exist_ok=True)

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
            x1_hat = alg2_x1_solve(x_tau, A_fn, AT_fn, y_clean, sigma_n_t, sig,
                                   tau, sk, ek, eff_si, x1_gt.clone(), cg_tol)
            m_st = F.interpolate(st["mask"], size=x1_gt.shape[-2:], mode="nearest")
            obs_rmse = math.sqrt(mse_masked(x1_hat, x1_gt, m_st))
            sanity["S2"][f"{task}/stage{si}"] = obs_rmse
            print(f"[S2] {task} stage{si}: observed RMSE {obs_rmse:.4f} (eta={sigma_n_t})",
                  flush=True)
            if si == 3:
                assert obs_rmse < 2 * sigma_n_t, "S2 failed — line 14 wiring bug"

    json.dump(sanity, open(os.path.join(args.out, "sanity.json"), "w"), indent=1)
    if args.self_test:
        print("[self-test] all checks passed", flush=True)
        return

    # ── model ───────────────────────────────────────────────────────────────
    model = config_utils.instantiate_from_config(config.model).to(device)
    ckpt = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu",
                      weights_only=False)
    model.load_state_dict(ckpt, strict=True)
    model.eval()
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
        n_px = 3 * h * w
        d_exact = apply_B(x1_gt, sk, ek, eff_si) - (ek - sk) * x0

        # per-(task,image) stage-shape A/A^T closures
        stage_A = {t: [s["mkA"](s["op"], s["y"], x1_gt[:1].shape, device)
                       for s in task_setups[t][2]] for t in args.tasks}
        stage_mask = {t: [F.interpolate(s["mask"], size=(h, w), mode="nearest")
                          for s in task_setups[t][2]] for t in args.tasks}

        for step_idx in range(len(sc.Timesteps)):
            T = sc.Timesteps[step_idx]
            tau = float(sc.t[step_idx])
            sigma_t = compute_sigma_tau(tau, sk, ek)
            x_tau = apply_H_tau(x1_gt, tau, sk, ek, eff_si) + sigma_t * x0

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

            skip_alg2 = sigma_t < SIGMA_MIN
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
                        x1_a2 = alg2_x1_solve(x_tau[sl], A_fn, AT_fn,
                                              setups[bi]["y"], eta, sigma_t, tau,
                                              sk, ek, eff_si, x1_model[sl].clone(), cg_tol)
                    row = dict(task=task, image=name, stage=si, step=step_idx,
                               tau=tau, T=float(T), sigma_tau=float(sigma_t),
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
    with open(os.path.join(args.out, "alg2_mse.csv"), "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=fields)
        wcsv.writeheader()
        wcsv.writerows(rows)
    json.dump({"note": "gamma2_meas is task-independent (no A/y in v or d_exact); "
                       "keyed stage -> tau -> mean_i ||v - d_exact||^2 / n",
               "table": gamma2_tab},
              open(os.path.join(args.out, "gamma2_meas.json"), "w"), indent=1)
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
        fig.savefig(os.path.join(args.out, f"alg2_{task}.png"), dpi=150)
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
    fig.savefig(os.path.join(args.out, "score_gamma2.png"), dpi=150)
    plt.close(fig)
    print("[done] figures saved ->", args.out, flush=True)


if __name__ == "__main__":
    main()
