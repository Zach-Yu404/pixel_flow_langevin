#!/usr/bin/env python
"""Algorithm 2 utilities (ICLR draft p.13) — the single library for main.py.

Paper-line map (variables follow the draft's symbols; anything the paper does
not name follows the legacy inference code ms_posterior_sampling_article_
version_final.py):
  l.5   H_tau / sigma_tau            -> final_utils.apply_H_tau / compute_sigma_tau
  l.6   B_k, N_k                     -> apply_B / apply_N
  l.7   M_tau (+ eps ridge, Prop. 4) -> M0 closures + power_iter_norm
  l.11  score solve (Cor. 8)         -> score_solve
  l.14  clean-image solve (Lem. 5)   -> clean_image_solve (one-step, deterministic b)
                                        / the exact draw inside run_posterior_sampling_alg2
  l.8-20 full sampler                -> run_posterior_sampling_alg2 (section layout
                                        mirrors run_posterior_sampling for side-by-side diff)

Verified (main.py --config verify.json, dense float64): G is exactly
self-adjoint (bilinear-half == 2x2 avg-pool, adjoint == nearest-up/4, G^2=G),
so nested H(H(x)) == H^T H x and N(N(x)) == N^T N x to ~1e-17 — the
un-transposed implementations are exact. Any non-self-adjoint future G would
break this; the verify mode is the guard.

History: split from algorithm2.py / full_ip_compare.py / debug_box_h0.py /
verify_components.py (git history keeps the originals; see
.research/tasks/debug-box-alg2-hole.md for the box-hole debugging record).
"""

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # PixelFlowICLR/ for base import
ORIG_CWD = os.getcwd()

import onestep_mse_vs_t as base  # noqa: E402  (sets IP_package sys.path, chdir)

# pipeline._daps_motion_kernel hardcodes the OLD box's DAPS path (contradictions
# registry #16); pre-inserting the live path makes its dead insert harmless.
_DAPS = os.path.join(base.IP_PACKAGE, "baselines", "DAPS")
if _DAPS not in sys.path:
    sys.path.insert(0, _DAPS)

import copy                                                    # noqa: E402
import torch                                                   # noqa: E402
import torch.nn.functional as F                                # noqa: E402

from ms_posterior_sampling_article_version_final import _per_stage  # noqa: E402
from ms_posterior_sampling_article_version_final_utils import (  # noqa: E402
    apply_G, apply_H_tau, compute_sigma_tau, cg_solve,
    make_velocity_fn, make_Ak_fns,
)
from pixelflow.scheduling_pixelflow import PixelFlowScheduler  # noqa: E402
from demo_runner import build_setup_and_measurement            # noqa: E402

SIGMA_MIN = 0.01          # skip Alg2 below this (matches sampler's Langevin skip)
TASKS = base.TASKS
H0 = 0.1                  # Block-2 step (paper SS7); h0/K sweeps: both hurt the box hole
TRAJ_IMAGE = "junco"      # trajectory/mp4 image (full_ip mode)
METHODS = ["wls", "model", "alg2"]
BEST_DIR = os.path.join(base.IP_PACKAGE, "rerun_imageNet", "configs_best_out")

NFE = {"n": 0}


def count_nfe_hook(module, args, kwargs=None):
    NFE["n"] += 1


# ── paper l.6 operators (reuse apply_G; stage_idx = eff_si keeps g_bypass) ────
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
        # reflection_pad2d_backward has no deterministic CUDA impl; the repo's
        # standing convention for blur/SR adjoints is warn_only (see DESIGN §10).
        det = torch.are_deterministic_algorithms_enabled()
        wo = torch.is_deterministic_algorithms_warn_only_enabled()
        if det and not wo:
            torch.use_deterministic_algorithms(True, warn_only=True)
        try:
            with torch.enable_grad():
                xp = torch.zeros(x_shape, device=r.device, dtype=r.dtype,
                                 requires_grad=True)
                Ax = A_fn(xp)
                (grad,) = torch.autograd.grad(Ax, xp, grad_outputs=r.detach())
        finally:
            if det and not wo:
                torch.use_deterministic_algorithms(True, warn_only=False)
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


def score_solve(x_tau, v, s_k, e_k, tau, gamma2, eff_si, cg_tol, L):
    """line 11: [N^2 + gamma2*H_tau^2] x0_hat = N(B x_tau - H_tau v)."""
    def op(u):
        Nu = apply_N(apply_N(u, s_k, e_k, eff_si), s_k, e_k, eff_si)
        if gamma2 > 0:
            Hu = apply_H_tau(apply_H_tau(u, tau, s_k, e_k, eff_si), tau, s_k, e_k, eff_si)
            Nu = Nu + gamma2 * Hu
        return Nu
    rhs = apply_N(apply_B(x_tau, s_k, e_k, eff_si) - apply_H_tau(v, tau, s_k, e_k, eff_si),
                  s_k, e_k, eff_si)
    return cg_solve(op, rhs, tol=cg_tol, max_iter=L)


def clean_image_solve(x_tau, A_fn, AT_fn, y, eta, sigma_tau, tau, s_k, e_k, eff_si,
                      x1_warm, cg_tol):
    """line 14 (one-step, deterministic b): M x = b, warm-started CG, max_iter=200."""
    inv_e2, inv_s2 = 1.0 / eta ** 2, 1.0 / float(sigma_tau) ** 2

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

    anchor > 0 (EXPERIMENTAL, task debug-box-alg2-hole): Tweedie-anchored
    Block 1 — CURRENTLY DISABLED by user edit (the b_tilde block below is
    commented out and M carries no anchor term); the parameter is kept for the
    pending adoption decision. Fixes the box hole 7/7 (pooled 6.2x) when on.

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
            if sigma_tau >= SIGMA_MIN:
                inv_e2, inv_s2 = 1.0 / eta ** 2, 1.0 / float(sigma_tau) ** 2

                def M0(x):                                           # l.7 (pre-ridge)
                    return inv_e2 * ATk(Ak(x)) + inv_s2 * apply_H_tau(
                        apply_H_tau(x, tau, s_k, e_k, eff_si), tau, s_k, e_k, eff_si)
                epsilon = 0.0
                if tau == 0.0:                                       # Prop. 4
                    epsilon = power_iter_norm(M0, x1.shape, device) * 1e-6
                # anchor path disabled by user (see commented block below);
                # keep M consistent with b_tilde: ridge only.
                M_tau = (lambda x: M0(x) + epsilon * x) if epsilon else M0

                for s in range(S):                                   # l.8
                    x_tau = apply_H_tau(x1, tau, s_k, e_k, eff_si) + sigma_tau * x0  # l.9
                    with torch.no_grad():
                        v = velocity_fn(x_tau)                       # l.10
                    x0_hat = score_solve(x_tau, v, s_k, e_k, tau, gamma2,
                                         eff_si, cg_tol, L)          # l.11
                    xi_y = randn_like_cpu(y)                         # l.12
                    xi_h = randn_like_cpu(x1)
                    b_tilde = inv_e2 * ATk(y) + \
                        inv_s2 * apply_H_tau(x_tau, tau, s_k, e_k, eff_si) + \
                        (1.0 / eta) * ATk(xi_y) + \
                        (1.0 / float(sigma_tau)) * apply_H_tau(xi_h, tau, s_k, e_k, eff_si)  # l.13
                    # if anchor > 0:                                   # EXPERIMENTAL anchor
                    #     x1_model = direct_estimate_x1(
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
