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

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # PixelFlowICLR/ for base import
sys.path.insert(0, HERE)                           # Algorithm2/ for measurement import
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
from measurement import build_setup_and_measurement           # noqa: E402

SIGMA_MIN = 0.01          # skip Alg2 below this (matches sampler's Langevin skip)
TASKS = base.TASKS
H0 = 0.1                  # Block-2 step (paper SS7); h0/K sweeps: both hurt the box hole

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


def apply_H_tau_inv(x, tau, s_k, e_k, stage_idx=None):
    """(H_tau^k)^-1 in closed form — paper p.20 (7.2) needs it for x1_hat.

    G is an orthogonal projection (G^2 = G, G^T = G; V1 verifies both), so
    H_tau = (1-tau) s_k G + tau e_k I has exactly two eigenvalues:
        (1-tau) s_k + tau e_k   on range(G)
        tau e_k                 on ker(G)
    and the inverse is the same split, no solve required.

    Refused where H_tau is singular: at tau=0 it is s_k G, and at stage 0 that
    is 0 outright. The guard has to come before the divides and cannot look at
    x — testing the residual instead let r = 0 through, and 0/0 returned NaN
    silently, which is exactly the case the sampler hits at stage 0, tau=0.
    """
    lam_range = (1.0 - tau) * s_k + tau * e_k
    lam_ker = tau * e_k
    if lam_range <= 0 or lam_ker <= 0:
        raise ValueError(
            f"H_tau is singular at tau={tau}, s_k={s_k}: x1_hat is undefined. "
            "Start the grid at the first tau > 0.")
    Gx = apply_G(x, stage_idx=stage_idx)
    return Gx / lam_range + (x - Gx) / lam_ker


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


def make_M_tau(A_fn, AT_fn, eta, sigma_tau, tau, s_k, e_k, eff_si, shape, device,
               ridge_rel):
    """line 7: M_tau = (1/eta^2) A^T A + (1/sigma_tau^2) H_tau^T H_tau, plus the
    Prop.-4 ridge eps*I at tau=0 (where H_0 = s_k G is rank deficient).
    Returns (M_fn, inv_e2, inv_s2, epsilon) — the scalars and epsilon are needed
    by the callers to build their right-hand sides."""
    inv_e2, inv_s2 = 1.0 / eta ** 2, 1.0 / float(sigma_tau) ** 2

    def M0(x):
        return inv_e2 * AT_fn(A_fn(x)) + inv_s2 * apply_H_tau(
            apply_H_tau(x, tau, s_k, e_k, eff_si), tau, s_k, e_k, eff_si)

    epsilon = power_iter_norm(M0, shape, device) * ridge_rel if tau == 0.0 else 0.0
    M_fn = (lambda x: M0(x) + epsilon * x) if epsilon else M0
    return M_fn, inv_e2, inv_s2, epsilon


def data_rhs(AT_fn, y, x_tau, inv_e2, inv_s2, tau, s_k, e_k, eff_si):
    """The deterministic part of b_tau (33): (1/eta^2) A^T y + (1/sigma^2) H^T x_tau.
    Line 14's one-step solve uses it as is; the sampler adds Lemma 5's noise."""
    return inv_e2 * AT_fn(y) + inv_s2 * apply_H_tau(x_tau, tau, s_k, e_k, eff_si)


def make_M_tau_wv(A_fn, AT_fn, eta, sigma_tau, tau, s_k, e_k, eff_si, shape, device,
                  ridge_rel, gamma2):
    """make_M_tau plus the (x0, x1)-coupling precision N^T N / (sigma_tau^2 gamma^2).

    x_tau = H_tau x1 + sigma_tau x0 and v_theta = B_k x1 - (e_k - s_k) x0 + eps_v
    with eps_v ~ N(0, gamma^2 I). Eliminating x0 between them leaves
        r_tau := (e_k - s_k) x_tau + sigma_tau v_theta = N_k x1 + sigma_tau eps_v
    -- the identity (e_k - s_k) H_tau + sigma_tau B_k == N_k is exact, and
    test/test_wv_alg2.py --check verifies it numerically. So r_tau is a second
    Gaussian observation of x1 with precision N^T N / (sigma_tau^2 gamma^2).
    gamma^2 is the same measured Cor.-8 table score_solve uses, so no hand-tuned
    weight enters.

    Returns (M_fn, inv_e2, inv_s2, inv_v2, epsilon) -- one scalar more than
    make_M_tau: the coupling weight inv_v2 = 1/(sigma_tau^2 gamma^2)."""
    inv_e2, inv_s2 = 1.0 / eta ** 2, 1.0 / float(sigma_tau) ** 2
    inv_v2 = inv_s2 / float(gamma2)

    def M0(x):
        return (inv_e2 * AT_fn(A_fn(x))
                + inv_s2 * apply_H_tau(
                    apply_H_tau(x, tau, s_k, e_k, eff_si), tau, s_k, e_k, eff_si)
                + inv_v2 * apply_N(
                    apply_N(x, s_k, e_k, eff_si), s_k, e_k, eff_si))

    epsilon = power_iter_norm(M0, shape, device) * ridge_rel if tau == 0.0 else 0.0
    M_fn = (lambda x: M0(x) + epsilon * x) if epsilon else M0
    return M_fn, inv_e2, inv_s2, inv_v2, epsilon


def data_rhs_wv(AT_fn, y, x_tau, v, inv_e2, inv_s2, inv_v2, sigma_tau,
                tau, s_k, e_k, eff_si):
    """data_rhs plus the coupling term N^T r_tau / (sigma_tau^2 gamma^2), with
    r_tau = (e_k - s_k) x_tau + sigma_tau v. The x_tau conditional term is kept
    unchanged -- the coupling is added information, not a replacement. x_tau
    must be the state v was evaluated at, or r_tau's identity does not hold."""
    r_tau = (e_k - s_k) * x_tau + float(sigma_tau) * v
    return (data_rhs(AT_fn, y, x_tau, inv_e2, inv_s2, tau, s_k, e_k, eff_si)
            + inv_v2 * apply_N(r_tau, s_k, e_k, eff_si))


def _x1_hat_diag(method, x_tau, v, x0_hat, sigma_tau, tau, s_k, e_k, eff_si):
    """Reported clean endpoint estimate (diagnostic only; see
    run_posterior_wv_sampling_alg2). None where (51) is not identifiable."""
    if method == "direct":
        return direct_estimate_x1(x_tau - tau * v, x_tau + (1.0 - tau) * v, s_k, e_k)
    if method != "inverse":
        raise ValueError(f"x1_hat_method {method!r} not in ('direct', 'inverse')")
    try:
        return apply_H_tau_inv(x_tau - float(sigma_tau) * x0_hat,
                               tau, s_k, e_k, eff_si)
    except ValueError:
        return None                     # H_tau singular: (51) is undefined here


def clean_image_solve(x_tau, A_fn, AT_fn, y, eta, sigma_tau, tau, s_k, e_k, eff_si,
                      x1_warm, cg_tol, ridge_rel=1e-6, max_iter=200):
    """line 14 (one-step, deterministic b): M x = b, warm-started CG.
    ridge_rel / max_iter come from config.json "algorithm"."""
    M_fn, inv_e2, inv_s2, _ = make_M_tau(A_fn, AT_fn, eta, sigma_tau, tau, s_k, e_k,
                                         eff_si, x_tau.shape, x_tau.device, ridge_rel)
    b = data_rhs(AT_fn, y, x_tau, inv_e2, inv_s2, tau, s_k, e_k, eff_si)
    return cg_solve(M_fn, b, x0=x1_warm, tol=cg_tol, max_iter=max_iter)


# ── measurement setup per (task, image): playground-identical code path ──────
def build_task_setups(task, demos, device, task_cfg):
    """task_cfg = config.json "tasks_setup"[task]: {"sigma_n", "operator"}.
    All operator/sigma_n values are inlined in config.json (provenance:
    IP_package LPIPS_king / configs_best_out, values verified identical)."""
    sigma_n = float(task_cfg["sigma_n"])
    setups = []
    for d in demos:
        op, mask, y, _, _, mkA, _ = build_setup_and_measurement(
            task, task_cfg["operator"], d, sigma_n, 256, device)
        if task in ("gaussian_blur", "motion_blur"):
            # replace the analytic flip(K) adjoint with the exact autograd one
            inner = mkA

            def mkA(operator, y_, stage_shape, device_, _inner=inner):
                A_fn, _ = _inner(operator, y_, stage_shape, device_)
                return A_fn, make_exact_AT(A_fn, tuple(stage_shape))
        setups.append(dict(op=op, mask=mask.to(device).float(), y=y, mkA=mkA))
    return sigma_n, setups

def direct_estimate_x1(x_start_hat, x_end_hat, s_k, e_k):
    """Direct x1 estimate: x1 = ((1-s)*xe - (1-e)*xs) / (e-s).
    Simpler than WLS, sometimes more accurate (see debug_IP TEST 8)."""
    denom = max(e_k - s_k, 1e-8)
    return ((1.0 - s_k) * x_end_hat - (1.0 - e_k) * x_start_hat) / denom


def run_posterior_sampling_alg2(
    model, config, gt, y, operator, eta, device,
    *,
    # Per-stage schedules — same names/positions as run_posterior_sampling
    num_langevin=10,                 # feeds S = paper's INNER iterations (l.8),
                                     # not a count of Langevin steps
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
    sigma_min=SIGMA_MIN,             # skip threshold (config.json "algorithm")
    ridge_rel=1e-6,                  # Prop.-4 ridge = ridge_rel * ||M|| at tau=0
    cg_max_iter_l14=200,             # line-14 CG cap at tau=0 (config "algorithm")
    terminal_replace_weight=0.0,     # POST-sampling projection (pipeline convention,
                                     # NOT a paper line): x1 <- w*(m*y+(1-m)*x1)+(1-w)*x1;
                                     # inpainting tasks use 1.0, blur/SR 0.0 (config)
    **unused_kw,                     # PRINCIPLE-only kw (h_x, lambda_reg, ...) ignored
):
    """Algorithm 2 (draft p.13) with the SAME section layout as
    ``run_posterior_sampling`` (ms_posterior_sampling_article_version_final.py)
    so the two samplers can be compared side by side. Differences are the
    paper-line comments: x1 init to 0 (l.1), exact Block-1 draw (l.12-14, no
    h1), x0 recompute (l.15), Block-2 x0 Langevin (l.16-17), and the ancestral
    transition U^(1) + fresh x0 (l.20) instead of the renoise transition.

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
            gamma2 = float(gamma2_stage.get(
                f"{round(tau, 6)}", list(gamma2_stage.values())[step_idx]))
            x0_hat = None
            if sigma_tau >= sigma_min:
                # l.7 (+ Prop.-4 ridge at tau=0); shared with line 14's one-step
                # solve and the GT diagnostic, so the operator is defined once.
                M_tau, inv_e2, inv_s2, epsilon = make_M_tau(
                    Ak, ATk, eta, sigma_tau, tau, s_k, e_k, eff_si,
                    x1.shape, device, ridge_rel)

                for s in range(S):                                   # l.8
                    x_tau = apply_H_tau(x1, tau, s_k, e_k, eff_si) + sigma_tau * x0  # l.9
                    with torch.no_grad():
                        v = velocity_fn(x_tau)                       # l.10
                    x0_hat = score_solve(x_tau, v, s_k, e_k, tau, gamma2,
                                         eff_si, cg_tol, L)          # l.11
                    xi_y = randn_like_cpu(y)                         # l.12
                    xi_h = randn_like_cpu(x1)
                    b_tilde = data_rhs(ATk, y, x_tau, inv_e2, inv_s2,
                                       tau, s_k, e_k, eff_si) + \
                        (1.0 / eta) * ATk(xi_y) + \
                        (1.0 / float(sigma_tau)) * apply_H_tau(xi_h, tau, s_k, e_k, eff_si)  # l.13
                    if epsilon:
                        # xi_y/xi_h give the RHS covariance M0; the tau=0 ridge
                        # (Prop. 4) solves against M0 + epsilon*I, so without this
                        # term the draw has covariance M^-1 M0 M^-1, not M^-1.
                        b_tilde = b_tilde + math.sqrt(epsilon) * randn_like_cpu(x1)
                    x1 = cg_solve(M_tau, b_tilde, x0=x1.clone(), tol=cg_tol,
                                  max_iter=cg_max_iter_l14 if tau == 0.0 else L)  # l.14
                    x0 = (x_tau - apply_H_tau(x1, tau, s_k, e_k, eff_si)) / float(sigma_tau)  # l.15
                    xi_0 = randn_like_cpu(x0)                    # l.16
                    x0 = x0 - (h0 / 2.0) * (x0 + x0_hat) + math.sqrt(h0) * xi_0  # l.17

            rows.append(dict(stage=si, step=step_idx, tau=tau,
                             sigma_tau=float(sigma_tau),
                             mse_x1=float(((x1 - pyr[si]) ** 2).mean())))
            if record_trajectory:
                x_tau_rec = apply_H_tau(x1, tau, s_k, e_k, eff_si) + float(sigma_tau) * x0
                traj.append((x_tau_rec[0].cpu(), x1[0].cpu(),
                             (x0_hat if x0_hat is not None else x0)[0].cpu()))


    # POST-sampling terminal projection (inpainting: snap observed pixels to y).
    # Pipeline convention (CONSTRAINTS: inpainting tr=1, blur/SR tr=0); applied
    # AFTER the loop, so per-step rows/traj metrics above are unaffected.
    if terminal_replace_weight > 0:
        m = operator.get_mask(x=y).float().to(x1.device)
        x1 = terminal_replace_weight * (m * y + (1.0 - m) * x1) + \
            (1.0 - terminal_replace_weight) * x1
    return x1, rows, traj


def run_posterior_wv_sampling_alg2(
    model, config, gt, y, operator, eta, device,
    *,
    # Per-stage schedules — same names/positions as run_posterior_sampling
    num_langevin=10,                 # feeds S = paper's INNER iterations (l.8),
                                     # not a count of Langevin steps
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
    sigma_min=SIGMA_MIN,             # skip threshold (config.json "algorithm")
    ridge_rel=1e-6,                  # Prop.-4 ridge = ridge_rel * ||M|| at tau=0
    cg_max_iter_l14=200,             # line-14 CG cap at tau=0 (config "algorithm")
    x1_hat_method="direct",          # DIAGNOSTIC ONLY: "direct" | "inverse"
    block1_noise=True,               # False -> l.14 returns the mean, not a draw
    terminal_replace_weight=0.0,     # POST-sampling projection (pipeline convention,
                                     # NOT a paper line): x1 <- w*(m*y+(1-m)*x1)+(1-w)*x1;
                                     # inpainting tasks use 1.0, blur/SR 0.0 (config)
    **unused_kw,                     # PRINCIPLE-only kw (h_x, lambda_reg, ...) ignored
):
    """Algorithm 2 with the (x0, x1)-coupling / velocity-uncertainty term.

    Block 1's M_tau and b_tau are the only algorithmic change (make_M_tau_wv /
    data_rhs_wv). Stage/time schedule, score solve, line-15 restore, Block-2
    Langevin, stage transition, CG, terminal projection and trajectory
    recording all run the same code as run_posterior_sampling_alg2:

        M_tau^wv = A^T A / eta^2 + H^T H / sigma^2 + N^T N / (sigma^2 gamma^2)
        b_tau^wv = A^T y / eta^2 + H^T x_tau / sigma^2
                   + N^T [(e-s) x_tau + sigma v] / (sigma^2 gamma^2)

    Lemma 5's draw needs noise matching the new precision, so it gains a third
    term (1/(sigma_tau*gamma)) N^T xi_v beside (1/eta) A^T xi_y and
    (1/sigma_tau) H^T xi_h.

    This is built on the UNMODIFIED Algorithm 2 body: it does not rebuild x_tau
    from a clean estimate before the draw. r_tau's identity needs the x_tau that
    v was evaluated at, so the l.9 state is the one both terms use.

    RNG: xi_v is drawn from a SECOND, independent CPU generator, leaving the
    shared stream in the positional order run_posterior_sampling_alg2 consumes
    (x0 -> xi_y -> xi_h -> ridge -> xi_0). Baseline and WV therefore see the
    same x0/xi_y/xi_h/xi_0 and differ only through M_tau and b_tau.

    ``x1_hat_method`` is DIAGNOSTIC ONLY -- the coupling RHS is built from
    (x_tau, v) and never needs a clean endpoint estimate. It selects which
    x1_hat is REPORTED in traj and changes nothing else: "direct" =
    direct_estimate_x1, "inverse" = (51) H_tau^-1 (x_tau - sigma_tau x0_hat).
    Where H_tau is singular (tau=0 outside stage 3) (51) is not identifiable
    and x1_hat is reported as None rather than regularised.

``block1_noise=False`` drops Lemma 5's noise from b_tau, so l.14 returns
    the conditional MEAN M_tau^-1 b_tau instead of a draw. Block 1 then stops
    being a posterior sample and the chain is no longer a Gibbs sampler for the
    target -- it is an alternating conditional-mean scheme. The xi draws still
    happen so the RNG stream stays positionally aligned with the sampling
    version; they are simply not used.

    Returns (x1, rows, traj). With ``record_trajectory`` the traj entries are
    (x_tau, x1, x0_hat, x1_hat, mu, x_tau_solve, v): the first three are
    Algorithm 2's, then the reported clean estimate, the Block-1 conditional
    mean M_tau^-1 b_tau (noise-free RHS), and the l.9 state and velocity of the
    LAST inner iteration -- the state mu was solved at, so a caller can rebuild
    any competing right-hand side at exactly the same state offline.
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
    # xi_v on its own stream keeps the shared one positionally identical to
    # run_posterior_sampling_alg2's (see docstring, RNG).
    g_v = torch.Generator(device="cpu").manual_seed(int(seed) + 104729)

    def randn_like_cpu(x):
        return torch.randn(x.shape, generator=g).to(x.device)

    def randn_v(x):
        return torch.randn(x.shape, generator=g_v).to(x.device)

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
            gamma2 = float(gamma2_stage.get(
                f"{round(tau, 6)}", list(gamma2_stage.values())[step_idx]))
            x0_hat = None
            mu_rec = x1_hat_rec = xt_rec = v_rec = None
            if sigma_tau >= sigma_min:
                # l.7 (+ Prop.-4 ridge at tau=0); shared with line 14's one-step
                # solve and the GT diagnostic, so the operator is defined once.
                M_tau, inv_e2, inv_s2, inv_v2, epsilon = make_M_tau_wv(
                    Ak, ATk, eta, sigma_tau, tau, s_k, e_k, eff_si,
                    x1.shape, device, ridge_rel, gamma2)

                for s in range(S):                                   # l.8
                    x_tau = apply_H_tau(x1, tau, s_k, e_k, eff_si) + sigma_tau * x0  # l.9
                    with torch.no_grad():
                        v = velocity_fn(x_tau)                       # l.10
                    x0_hat = score_solve(x_tau, v, s_k, e_k, tau, gamma2,
                                         eff_si, cg_tol, L)          # l.11
                    b_det = data_rhs_wv(ATk, y, x_tau, v, inv_e2, inv_s2, inv_v2,
                                        sigma_tau, tau, s_k, e_k, eff_si)
                    if record_trajectory and s == S - 1:
                        mu_rec = cg_solve(M_tau, b_det, x0=x1.clone(), tol=cg_tol,
                                          max_iter=cg_max_iter_l14 if tau == 0.0 else L)
                        x1_hat_rec = _x1_hat_diag(x1_hat_method, x_tau, v, x0_hat,
                                                  sigma_tau, tau, s_k, e_k, eff_si)
                        xt_rec, v_rec = x_tau, v
                    xi_y = randn_like_cpu(y)                         # l.12
                    xi_h = randn_like_cpu(x1)
                    xi_v = randn_v(x1)                               # own stream
                    b_tilde = b_det + \
                        (1.0 / eta) * ATk(xi_y) + \
                        (1.0 / float(sigma_tau)) * apply_H_tau(xi_h, tau, s_k, e_k, eff_si) + \
                        (1.0 / (float(sigma_tau) * math.sqrt(float(gamma2)))) * \
                        apply_N(xi_v, s_k, e_k, eff_si)              # l.13
                    if not block1_noise:
                        b_tilde = b_det
                    if epsilon:
                        # xi_y/xi_h give the RHS covariance M0; the tau=0 ridge
                        # (Prop. 4) solves against M0 + epsilon*I, so without this
                        # term the draw has covariance M^-1 M0 M^-1, not M^-1.
                        xi_eps = randn_like_cpu(x1)          # drawn either way
                        if block1_noise:
                            b_tilde = b_tilde + math.sqrt(epsilon) * xi_eps
                    x1 = cg_solve(M_tau, b_tilde, x0=x1.clone(), tol=cg_tol,
                                  max_iter=cg_max_iter_l14 if tau == 0.0 else L)  # l.14
                    x0 = (x_tau - apply_H_tau(x1, tau, s_k, e_k, eff_si)) / float(sigma_tau)  # l.15
                    xi_0 = randn_like_cpu(x0)                    # l.16
                    x0 = x0 - (h0 / 2.0) * (x0 + x0_hat) + math.sqrt(h0) * xi_0  # l.17

            rows.append(dict(stage=si, step=step_idx, tau=tau,
                             sigma_tau=float(sigma_tau),
                             mse_x1=float(((x1 - pyr[si]) ** 2).mean())))
            if record_trajectory:
                x_tau_rec = apply_H_tau(x1, tau, s_k, e_k, eff_si) + float(sigma_tau) * x0
                traj.append((x_tau_rec[0].cpu(), x1[0].cpu(),
                             (x0_hat if x0_hat is not None else x0)[0].cpu(),
                             None if x1_hat_rec is None else x1_hat_rec[0].cpu(),
                             None if mu_rec is None else mu_rec[0].cpu(),
                             None if xt_rec is None else xt_rec[0].cpu(),
                             None if v_rec is None else v_rec[0].cpu()))


    # POST-sampling terminal projection (inpainting: snap observed pixels to y).
    # Pipeline convention (CONSTRAINTS: inpainting tr=1, blur/SR tr=0); applied
    # AFTER the loop, so per-step rows/traj metrics above are unaffected.
    if terminal_replace_weight > 0:
        m = operator.get_mask(x=y).float().to(x1.device)
        x1 = terminal_replace_weight * (m * y + (1.0 - m) * x1) + \
            (1.0 - terminal_replace_weight) * x1
    return x1, rows, traj


def _make_anchor_P(operator, y, shape, device, mode):
    """P for run_posterior_reg_sampling_alg2's anchor, plus the kind actually
    used. "nullspace" is the hole indicator at the stage resolution (see that
    function's docstring); operators with no hole fall back to identity."""
    if mode == "identity":
        return (lambda x: x), "identity"
    if mode != "nullspace":
        raise ValueError(f"anchor_P {mode!r} not in ('nullspace', 'identity')")
    hole = 1.0 - operator.get_mask(x=y).float().to(device)
    if float(hole.sum()) == 0.0:
        return (lambda x: x), "identity(fallback: operator has no hole mask)"
    m_k = F.interpolate(hole, size=tuple(shape[-2:]), mode="nearest")
    return (lambda x: m_k * x), "nullspace"

def run_posterior_reg_sampling_alg2(
    model, config, gt, y, operator, eta, device,
    *,
    # Per-stage schedules — same names/positions as run_posterior_sampling
    num_langevin=10,                 # feeds S = paper's INNER iterations (l.8),
                                     # not a count of Langevin steps
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
    sigma_min=SIGMA_MIN,             # skip threshold (config.json "algorithm")
    ridge_rel=1e-6,                  # Prop.-4 ridge = ridge_rel * ||M|| at tau=0
    cg_max_iter_l14=200,             # line-14 CG cap at tau=0 (config "algorithm")
    anchor_lambda=0.0,               # Tweedie pseudo-observation weight
    anchor_P="identity",             # "identity" (the original) | "nullspace"
    block1_noise=True,               # ablation: Lemma 5's xi_y / xi_h in b
    block2_noise=True,               # ablation: sqrt(h0) xi_0 in l.17
    terminal_replace_weight=0.0,     # POST-sampling projection (pipeline convention,
                                     # NOT a paper line): x1 <- w*(m*y+(1-m)*x1)+(1-w)*x1;
                                     # inpainting tasks use 1.0, blur/SR 0.0 (config)
    **unused_kw,                     # PRINCIPLE-only kw (h_x, lambda_reg, ...) ignored
):
    """Algorithm 2 with a Tweedie pseudo-observation anchoring Block 1.

    Treat the network's clean estimate as one more Gaussian observation,
    x1_model = x1 + eps_a with eps_a ~ N(0, lambda^-1 P^+):

        M_tau = M_tau^(0) + lambda P
        b_tau = b_tau^(0) + lambda P x1_model

    with M_tau^(0), b_tau^(0) Algorithm 2's own (make_M_tau / data_rhs, the
    H^T x_tau form) and x1_model = direct_estimate_x1 of the l.10 velocity --
    zero extra NFE, direct_estimate_x1 unmodified.

    ``anchor_P="identity"`` (default) is the ORIGINAL formulation, restored from
    commit 74a0c39 (`sample_alg2(anchor=lam)`, dropped again in 8ff8091):
    M += lam*I, b += lam*x1_model + sqrt(lam)*xi_a. On junco it took the hole
    from 0.969 to 0.086 at lam=25.

    ``anchor_P="nullspace"`` is the alternative that anchors only the
    measurement nullspace: for inpainting A_k = mask . U, so P is the hole
    indicator at the stage resolution -- diagonal 0/1, hence P = P^T = P^2, and
    the observed region gets no anchor at all. Because U interpolates, that P is
    exactly inside ker(A_k) only at stage 3; at stages 0-2 it leaks 16-42% (see
    test/test_reg_alg2.py --check). Operators whose get_mask has no hole (blur /
    SR) fall back to P = I, reported in ``rows["anchor_P"]``, never silent.

    Lemma 5's draw needs noise matching the added precision. P being an
    orthogonal projection (identity included), sqrt(lambda) P xi_a has
    covariance lambda P exactly, so the RHS gains that term.

    ``block1_noise=False`` drops (1/eta) A^T xi_y + (1/sigma_tau) H^T xi_h from
    b_tilde, so l.14 returns the conditional MEAN of the anchored Block 1
    instead of a draw (the anchor's own sqrt(lambda) P xi_a stays).
    ``block2_noise=False`` drops sqrt(h0) xi_0 from l.17, making Block 2 a
    deterministic gradient step. Either one makes the chain stop being a Gibbs
    sampler for the target. Both xi are still DRAWN so the RNG stream stays
    positionally aligned with the full-noise version; they are just multiplied
    out.

    Two deliberate departures from 74a0c39, both later fixes rather than part of
    the anchor:
      * the tau=0 ridge keeps its sqrt(epsilon) xi_eps term (74a0c39 put epsilon
        into M without the matching RHS noise, which drew from M^-1 M0 M^-1);
      * xi_a is drawn at EVERY lambda, including 0, and multiplied by
        sqrt(lambda). 74a0c39 drew it only when anchor > 0, which shifted the
        stream so different lambdas were not comparable. Every lambda now
        consumes the same count/shape/order of random numbers, making the sweep
        a paired-noise comparison; the cost is one extra draw per inner
        iteration relative to run_posterior_sampling_alg2, so lambda=0 is the
        in-family baseline rather than a bit-exact reproduction of it.

    Returns (x1, rows, traj). With ``record_trajectory`` traj entries are
    (x_tau, x1, x0_hat, x1_model, mu, x_tau_solve): Algorithm 2's three, then
    the Tweedie estimate, the Block-1 conditional mean M_tau^-1 b_tau
    (noise-free RHS) and the l.9 state mu was solved at.
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
        P_fn, P_kind = _make_anchor_P(operator, y, (B, 3, h, w), device, anchor_P)

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
            gamma2 = float(gamma2_stage.get(
                f"{round(tau, 6)}", list(gamma2_stage.values())[step_idx]))
            x0_hat = None
            mu_rec = x1m_rec = xt_rec = None
            if sigma_tau >= sigma_min:
                # l.7 (+ Prop.-4 ridge at tau=0); shared with line 14's one-step
                # solve and the GT diagnostic, so the operator is defined once.
                M_base, inv_e2, inv_s2, epsilon = make_M_tau(
                    Ak, ATk, eta, sigma_tau, tau, s_k, e_k, eff_si,
                    x1.shape, device, ridge_rel)
                M_tau = ((lambda x: M_base(x) + anchor_lambda * P_fn(x))
                         if anchor_lambda else M_base)

                for s in range(S):                                   # l.8
                    x_tau = apply_H_tau(x1, tau, s_k, e_k, eff_si) + sigma_tau * x0  # l.9
                    with torch.no_grad():
                        v = velocity_fn(x_tau)                       # l.10
                    x0_hat = score_solve(x_tau, v, s_k, e_k, tau, gamma2,
                                         eff_si, cg_tol, L)          # l.11
                    x_start_hat = x_tau - tau * v
                    x_end_hat = x_tau + (1.0 - tau) * v
                    x1_model = direct_estimate_x1(x_start_hat, x_end_hat, s_k, e_k)
                    b_det = data_rhs(ATk, y, x_tau, inv_e2, inv_s2,
                                     tau, s_k, e_k, eff_si)
                    if anchor_lambda:
                        b_det = b_det + anchor_lambda * P_fn(x1_model)
                    if record_trajectory and s == S - 1:
                        mu_rec = cg_solve(M_tau, b_det, x0=x1.clone(), tol=cg_tol,
                                          max_iter=cg_max_iter_l14 if tau == 0.0 else L)
                        x1m_rec, xt_rec = x1_model, x_tau
                    xi_y = randn_like_cpu(y)                         # l.12
                    xi_h = randn_like_cpu(x1)
                    xi_a = randn_like_cpu(x1)   # drawn at EVERY lambda (paired noise)
                    b_tilde = b_det + math.sqrt(anchor_lambda) * P_fn(xi_a)
                    if block1_noise:                                 # l.13
                        b_tilde = b_tilde + (1.0 / eta) * ATk(xi_y) + \
                            (1.0 / float(sigma_tau)) * apply_H_tau(
                                xi_h, tau, s_k, e_k, eff_si)
                    if epsilon:
                        # xi_y/xi_h give the RHS covariance M0; the tau=0 ridge
                        # (Prop. 4) solves against M0 + epsilon*I, so without this
                        # term the draw has covariance M^-1 M0 M^-1, not M^-1.
                        b_tilde = b_tilde + math.sqrt(epsilon) * randn_like_cpu(x1)
                    x1 = cg_solve(M_tau, b_tilde, x0=x1.clone(), tol=cg_tol,
                                  max_iter=cg_max_iter_l14 if tau == 0.0 else L)  # l.14
                    x0 = (x_tau - apply_H_tau(x1, tau, s_k, e_k, eff_si)) / float(sigma_tau)  # l.15
                    xi_0 = randn_like_cpu(x0)                    # l.16
                    x0 = x0 - (h0 / 2.0) * (x0 + x0_hat)             # l.17
                    if block2_noise:
                        x0 = x0 + math.sqrt(h0) * xi_0

            rows.append(dict(stage=si, step=step_idx, tau=tau,
                             sigma_tau=float(sigma_tau), anchor_P=P_kind,
                             mse_x1=float(((x1 - pyr[si]) ** 2).mean())))
            if record_trajectory:
                x_tau_rec = apply_H_tau(x1, tau, s_k, e_k, eff_si) + float(sigma_tau) * x0
                traj.append((x_tau_rec[0].cpu(), x1[0].cpu(),
                             (x0_hat if x0_hat is not None else x0)[0].cpu(),
                             None if x1m_rec is None else x1m_rec[0].cpu(),
                             None if mu_rec is None else mu_rec[0].cpu(),
                             None if xt_rec is None else xt_rec[0].cpu()))


    # POST-sampling terminal projection (inpainting: snap observed pixels to y).
    # Pipeline convention (CONSTRAINTS: inpainting tr=1, blur/SR tr=0); applied
    # AFTER the loop, so per-step rows/traj metrics above are unaffected.
    if terminal_replace_weight > 0:
        m = operator.get_mask(x=y).float().to(x1.device)
        x1 = terminal_replace_weight * (m * y + (1.0 - m) * x1) + \
            (1.0 - terminal_replace_weight) * x1
    return x1, rows, traj




# ═══════════════════════════════════════════════════════════════════════════
# Algorithm 4 — "A clean-endpoint posterior sampler for cascaded flow priors"
# (results/algorithm.4pdf.pdf; the note numbers its own listing "Algorithm 1")
#
# Equation numbers below are that draft's. The reading note is
# .research/references/2026-08-21-algorithm4-clean-endpoint-sampler.md.
#
# What changes relative to Algorithm 2, and why:
#
#   Alg. 2 couples x1 to x_tau through the flat-prior Gaussian
#   N(x_tau; H_tau x1, sigma_tau^2 I). The learned marginal p_tau survives in
#   the x_tau-conditional, so that block needs a Langevin step (h0) and the
#   sampled target carries a prior smoothed by sqrt(2) sigma_tau H_tau^-1.
#
#   Alg. 4 couples through the true denoising conditional (5). p_tau cancels
#   from (7), both conditionals are Gaussian, and both are drawn exactly:
#     - no h0 and no h1 anywhere (Sec. 6.3: no step size reproduces an exact
#       draw -- the drift needs h=2 and the noise needs h=1);
#     - no ridge at tau=0 (Prop. 4(a): S^-1 makes M_tau^den positive definite
#       at every tau, so ridge_rel / power_iter_norm / sqrt(eps) xi_eps and the
#       cg_max_iter_l14 special case all disappear);
#     - the price is the Gaussian surrogate (8) for p(x1|x_tau), whose
#       covariance is fixed by (12) with no tuned quantity.
#
# The one input Algorithm 2 does not have is S, the prior covariance surrogate
# (draft "Require:" line). It is a MEASURED quantity, not a searched one --
# see main4.py --mode measure_s2. Note lambda in
# run_posterior_reg_sampling_alg2 is the same object: lambda*I == S^-1, i.e.
# s2 == 1/lambda.
#
# Symbol collision, kept explicit throughout: the draft's S is the prior
# covariance surrogate; this repo's S (num_langevin) is the inner-iteration
# count. Below they are s2 / S_prior and S_it.
# ═══════════════════════════════════════════════════════════════════════════

def make_endpoint_operator(s_k, e_k, tau, gamma2, eff_si):
    """The operator shared by (18) and (19): u -> [N_k^2 + gamma^2 H_tau^2] u.

    score_solve builds this same closure inline. It is duplicated rather than
    factored out of score_solve so that committed Algorithm-2 results cannot
    move; test_alg4.py --mode verify (A1) asserts the two agree densely, which
    is what keeps them in sync.

    SPD at EVERY tau, tau=0 included: with 0 <= s_k < e_k <= 1 the operator
    N_k = e_k(1-s_k)I - s_k(1-e_k)G has eigenvalue e_k - s_k > 0 on range(G)
    and e_k(1-s_k) > 0 on ker(G), so N_k^2 is positive definite and the
    gamma^2 H_tau^2 term only adds. (19) therefore never needs a ridge.
    """
    def op(u):
        Nu = apply_N(apply_N(u, s_k, e_k, eff_si), s_k, e_k, eff_si)
        if gamma2 > 0:
            Hu = apply_H_tau(apply_H_tau(u, tau, s_k, e_k, eff_si),
                             tau, s_k, e_k, eff_si)
            Nu = Nu + gamma2 * Hu
        return Nu
    return op


def clean_endpoint_solve(x_tau, v, sigma_tau, s_k, e_k, tau, gamma2, eff_si,
                         cg_tol, L, x1_warm=None):
    """draft l.11 / Prop. 7 (19) — the clean endpoint in ONE solve:

        [N^2 + gamma^2 H_tau^2] x1_hat
             = N [ (e_k - s_k) x_tau + sigma_tau v ] + gamma^2 H_tau x_tau

    Same left-hand operator as score_solve's (18), so a caller that wants both
    x0_hat and x1_hat pays for two right-hand sides, not two operators.

    Prop. 7 proves that for tau > 0 this equals H_tau^-1 (x_tau - sigma_tau
    x0_hat) with x0_hat the solution of (18) -- verified in test_alg4.py (A2).
    The proof cancels a factor H_tau from both sides and that factor is
    invertible only for tau > 0; AT tau = 0 THE CANCELLED FORM IS THE
    DEFINITION of x1_hat, not a rewriting of an identity, because
    H_tau^-1 (x_tau - sigma_tau x0_hat) does not exist there. The draft's own
    listing runs line 11 on every tau of the stage grid with no tau=0 case,
    and the operator stays SPD there, so that is what is implemented.

    Limits (Appendix A.3, checked in A3): gamma -> 0 gives (17),
    N x1_hat = (e_k - s_k) x_tau + sigma_tau v; gamma -> infinity gives
    x1_hat = H_tau^-1 x_tau, the flat-prior deconvolution that the
    interpolant-coupled samplers centre on.
    """
    op = make_endpoint_operator(s_k, e_k, tau, gamma2, eff_si)
    rhs = apply_N((e_k - s_k) * x_tau + float(sigma_tau) * v, s_k, e_k, eff_si)
    if gamma2 > 0:
        rhs = rhs + gamma2 * apply_H_tau(x_tau, tau, s_k, e_k, eff_si)
    return cg_solve(op, rhs, x0=x1_warm, tol=cg_tol, max_iter=L)


class SOperator:
    """Prior covariance surrogate S of (12)/(22), reduced to the ONLY two
    applications Block 1 needs. Instances are per-stage (none of the
    implemented constructions depends on tau).

        apply_S_inv(x)      = S^-1 x        (the (12) precision term)
        apply_S_inv_sqrt(x) = S^-1/2 x      (the Lemma-9 R3^T factor)

    S must be symmetric positive definite and apply_S_inv_sqrt must satisfy
    S^-1/2 S^-1/2 = S^-1 (all implementations below are diagonal in some
    orthonormal basis, so this is exact). `scalar_equiv` = Tr(S)/D is what
    the trajectory row records as s2 and what the diagnostics' scalar
    precision columns use; `meta` is a JSON-able description.
    """
    scalar_equiv = float("nan")
    meta = {}

    def apply_S_inv(self, x):
        raise NotImplementedError

    def apply_S_inv_sqrt(self, x):
        raise NotImplementedError


class ScalarSOp(SOperator):
    """S = s2 * I — the isotropic surrogate (Sec. 4.2). Bit-identical to the
    pre-SOperator scalar code path: same 1/s2 and 1/sqrt(s2) constants."""

    def __init__(self, s2):
        if not (s2 > 0):
            raise ValueError(f"S_prior s2 must be > 0 (got {s2!r}); "
                             "S^-1 is what makes M_tau^den positive definite at tau=0")
        self.s2 = float(s2)
        self.scalar_equiv = float(s2)
        self._inv = 1.0 / float(s2)
        self._inv_half = 1.0 / math.sqrt(float(s2))
        self.meta = {"mode": "scalar", "s2": float(s2)}

    def apply_S_inv(self, x):
        return self._inv * x

    def apply_S_inv_sqrt(self, x):
        return self._inv_half * x

    def inv_diag_mean(self):
        # exact constant diagonal of S^-1 for the isotropic case
        return self._inv


class SpectralSOp(SOperator):
    """S diagonal in the (orthonormal) 2-D Fourier basis: S(w) = power[w].

    `power` is the floored mean power spectrum of the centered calibration
    images at this stage's resolution, shape (H, W), real, symmetric under
    w -> -w (automatic for power spectra of real fields), shared across the
    three channels. Then S^-1 x = F^H diag(1/power) F x is real, symmetric
    and positive definite; the .real below only strips float round-off.
    """

    def __init__(self, power, meta=None):
        if not torch.is_tensor(power) or power.dim() != 2:
            raise ValueError("spectral power must be a 2-D tensor (H, W)")
        if not bool((power > 0).all()):
            raise ValueError("spectral power must be strictly positive (floor it first)")
        self.power = power.detach().to(torch.float32)
        self.scalar_equiv = float(power.mean())
        self.meta = dict(meta or {}, mode="spectral",
                         scalar_equiv=self.scalar_equiv)

    def _apply(self, x, p):
        X = torch.fft.fft2(x, norm="ortho")
        return torch.fft.ifft2(X / p.to(x.device), norm="ortho").real

    def apply_S_inv(self, x):
        return self._apply(x, self.power)

    def apply_S_inv_sqrt(self, x):
        return self._apply(x, self.power.sqrt())

    def inv_diag_mean(self):
        # S^-1 = F^H diag(1/P) F has CONSTANT diagonal mean_w 1/P(w). The
        # all-ones Jacobi probe instead sees only the DC response 1/P(0),
        # which under-estimates it by orders of magnitude (image DC power is
        # huge) and cripples the preconditioner exactly where S^-1 dominates
        # M — the 2026-08-26 non-converged-PCG bug. Use this for the probe.
        return float(self.power.reciprocal().mean())


def make_M_tau_den(A_fn, AT_fn, eta, sigma_tau, tau, s_k, e_k, eff_si, s2):
    """draft l.7 / (12) + (21). Returns (M_fn, Cinv_fn, inv_e2, inv_s2, inv_S).

        C^-1     = H_tau^T H_tau / sigma_tau^2 + S^-1          (12)
        M_tau^den = A_k^T A_k / eta^2 + C^-1                    (21)

    `s2` is either the isotropic variance (S = s2 * I, so S^-1 x = x / s2 and
    the Lemma-9 factor S^-1/2 is a division by sqrt(s2) -- no square root of
    a matrix is formed, draft Sec. 6.1) or an SOperator, whose apply_S_inv
    supplies the S^-1 term instead. The mathematical form of (12)/(21) is the
    same either way. The returned inv_S stays a FLOAT (1/scalar-equivalent
    of S) so existing scalar consumers are untouched.

    NO RIDGE and no power iteration, unlike make_M_tau: by Prop. 4(a),
    C^-1 >= S^-1 > 0 for every tau in [0,1] including tau=0, where the first
    term degenerates to s_k^2 G^2 / sigma_tau^2 and vanishes on ker(G). The
    surrogate supplies the entire precision in exactly the directions the
    interpolant does not constrain.
    """
    s_op = s2 if isinstance(s2, SOperator) else ScalarSOp(float(s2))
    inv_e2, inv_s2 = 1.0 / eta ** 2, 1.0 / float(sigma_tau) ** 2
    inv_S = 1.0 / float(s_op.scalar_equiv)

    def Cinv(x):
        return inv_s2 * apply_H_tau(
            apply_H_tau(x, tau, s_k, e_k, eff_si), tau, s_k, e_k, eff_si) \
            + s_op.apply_S_inv(x)

    def M(x):
        return inv_e2 * AT_fn(A_fn(x)) + Cinv(x)

    return M, Cinv, inv_e2, inv_s2, inv_S


def measurement_residual(A_fn, x1, y, eta):
    """draft Sec. 8.6: ||A_k x1 - y|| / (eta sqrt(m)), which "should settle
    near 1". Reported per step rather than tuned. Averaged over the batch."""
    B = y.shape[0]
    m = y[0].numel()
    r = ((A_fn(x1) - y) ** 2).reshape(B, -1).sum(dim=1).sqrt()
    return float((r / (eta * math.sqrt(m))).mean())


def run_posterior_sampling_alg4(
    model, config, gt, y, operator, eta, device,
    *,
    # Per-stage schedules — same names/positions as run_posterior_sampling_alg2
    num_langevin=10,                 # draft S_it: INNER iterations (l.9);
                                     # scalar or per-stage list, like
                                     # ode_steps_per_stage
                                     # The name is kept so task_kw/config keys
                                     # stay shared with the other samplers; it
                                     # is not a count of Langevin steps here
                                     # because there are none.
    ode_steps_per_stage=10, shift=1.0,
    # Structural / physical
    guidance_scale=0.0, class_label=10,
    cg_tol=1e-5, cg_max_iter=50,     # draft: CG iterations L
    make_Ak_fns_fn=None,
    seed=42,
    record_trajectory=False,
    diag=None,                       # optional Alg4Diagnostics recorder;
                                     # READ-ONLY and RNG-free (see alg4_diag.py)
    # Algorithm-4 specific
    gamma2_tab=None,                 # measured gamma^2(k, tau) table, (20)
    s2_fn=None,                      # REQUIRED (stage_idx, sigma_tau) -> s^2
    hole_mask=None,                  # optional [1,1,H,W], 1 = unobserved. When
                                     # given, every row also carries mse_hole /
                                     # mse_obs at that stage's resolution --
                                     # the split every inpainting result in
                                     # this repo is compared on.
    sigma_min=SIGMA_MIN,             # sigma_tau floor (config "algorithm")
    cg_max_iter_endpoint=200,        # CG cap for the (19) solve
    terminal_replace_weight=0.0,     # POST-sampling projection (pipeline
                                     # convention, NOT a draft line)
    diag_noise_off=None,             # DIAGNOSTIC PROBE ONLY: iterable of
                                     # noise names ("xi_y","xi_h","xi_s",
                                     # "xi_0") to ZERO from
                                     # diag_noise_off_from_stage on. Noises
                                     # are still DRAWN first so the RNG
                                     # stream stays aligned across arms.
                                     # Breaks the exact-draw property — never
                                     # a production setting. None = exact
                                     # sampler, bit-identical path.
    diag_noise_off_from_stage=2,
    diag_xi0_use_xih=False,          # DIAGNOSTIC PROBE ONLY: Block 2 uses the
                                     # xi_h REALISATION drawn for (22) this
                                     # inner (pre-zeroing) as xi_0 in (23).
                                     # xi_0 is still drawn (RNG aligned) and
                                     # discarded. False = bit-identical.
    diag_xi0_off_from_frame=None,    # DIAGNOSTIC PROBE ONLY: from this global
                                     # frame index on, use x0 = 0 in l.8
                                     # (x_tau = H x1) and zero Block-2 xi_0
                                     # (drawn first, RNG stays aligned). Breaks
                                     # the exact draw. None = bit-identical.
    diag_block2_langevin=None,       # DIAGNOSTIC PROBE ONLY (user 2026-09-03): if a
                                     # float h0 is given, Block 2 is replaced by
                                     # Algorithm 2's l.11 + l.15-17 on x0:
                                     #   x0_hat = [N^2 + g2 H^2]^-1 N (B x_tau - H v)   (CG)
                                     #   x0 = (x_tau - H x1)/sigma_tau
                                     #   x0 = x0 - h0/2 (x0 + x0_hat) + sqrt(h0) xi_0
                                     #   x_tau = H x1 + sigma_tau x0
                                     # None = the exact draw (23), bit-identical.
    diag_block2_langevin_sign=+1.0,  # probe: sign of x0_hat in the drift
                                     # (+1 = the spec / Alg-2 l.17; -1 = pull toward +x0_hat)
    **unused_kw,
):
    """Algorithm 4 (draft Sec. 7) with the SAME section layout as
    ``run_posterior_sampling_alg2`` so the two can be diffed side by side.

    Line numbers in the comments are the draft listing's own:

       l.1   x_1 <- 0
       l.2   for stage k = 0..K-1:
       l.3     x_0 ~ N(0, I_n)                       [fresh endpoint per stage]
       l.4     for tau on the stage grid:
       l.5       H_tau, sigma_tau
       l.6       N_k                                 [independent of tau]
       l.7       C^-1 <- H^T H/sigma^2 + S^-1;  M^den <- A^T A/eta^2 + C^-1
       l.8       x_tau <- H_tau x_1 + sigma_tau x_0  [ONCE per tau, not per s]
       l.9       for s = 1..S_it:
       l.10        v <- v_theta(x_tau, tau, k)       [one network evaluation]
       l.11        solve [N^2 + gamma^2 H^2] x1_hat = N[(e-s)x_tau + sigma v]
                                                      + gamma^2 H x_tau   (19)
       l.12        xi_y ~ N(0,I_m);  xi_h, xi_s ~ N(0,I_n)
       l.13        solve M^den x_1 = A^T y/eta^2 + C^-1 x1_hat
                        + A^T xi_y/eta + H^T xi_h/sigma + xi_s/sqrt(s2)     (22)
       l.14        xi_0 ~ N(0,I_n);  x_tau <- H_tau x_1 + sigma_tau xi_0    (23)
       l.15      end for
       l.16      x_0 <- (x_tau - H_tau x_1)/sigma_tau  [leave the coordinates]
       l.17    end for
       l.18    x_1 <- U^(1)(x_1);  x_0 ~ N(0,I)      [ancestral transition]
       l.20  return x_1^{K-1}

    THREE STRUCTURAL DIFFERENCES from run_posterior_sampling_alg2, each of
    which is the point of the draft rather than an implementation choice:

    1. **l.8 sits outside the inner loop.** In Algorithm 2, x_tau is rebuilt
       from (x_1, x_0) at the top of every inner iteration. Here x_tau is a
       state variable of the inner loop: Block 2 (l.14) updates it, and l.8
       only enters the coordinates once per tau. Moving l.8 inside would make
       Block 2's draw unobservable and turn the scheme back into Algorithm 2's.
    2. **Block 1 centres on x1_hat, not on x_tau.** Algorithm 2's right-hand
       side carries H^T x_tau / sigma^2; here the whole of C^-1 acts on the
       clean endpoint (22). Draft Sec. 8.4 is explicit that this is the trade:
       x1_hat carries the velocity error at full weight, whereas x_tau is
       exact, so Algorithm 4's advantage shrinks as gamma grows and reverses
       once gamma is large enough. gamma^2 is tabulated per (k, tau) by (20),
       so the crossover can be located empirically before deployment.
    3. **Block 2 is a direct draw (23), not a Langevin step.** There is no h0.
       Note l.16 then returns exactly the xi_0 drawn at l.14 -- it is computed
       from the state anyway, as the listing writes it, and test_alg4.py (A5)
       checks the identity.

    ``s2_fn(stage_idx, sigma_tau) -> float`` supplies S = s2*I. It is required:
    there is no default, because the draft's whole claim is that this quantity
    is measured (Sec. 4.2, Sec. 8.6) and a code-side default would silently
    reintroduce the tuned parameter the construction removes.

    Line 13 uses ``pcg_solve`` with a Jacobi preconditioner rather than
    ``cg_solve``. The system, and therefore the sampled distribution, is
    identical; the preconditioner only changes how many iterations reaching the
    solution takes. It is here because plain CG was NOT reaching it: see
    ``--mode cg_audit``. Every row carries ``blk1_cg_iters`` /
    ``blk1_cg_resid`` / ``blk1_cg_converged`` so a truncated solve is visible
    instead of silent.

    Returns (x1, rows, traj). ``rows`` carries the Algorithm-2 columns plus
    ``meas_resid`` (Sec. 8.6), the ``s2`` actually used at that step, the
    ``x0_rms`` diagnostic the draft asks for in Sec. 7 (||x_0||^2/n ~ 1), and,
    when ``hole_mask`` is supplied, ``mse_hole`` / ``mse_obs``.
    ``traj`` entries are (x_tau, x1, x1_hat) -- x1_hat replaces Algorithm 2's
    x0_hat, since (18) is not solved here.
    """
    if make_Ak_fns_fn is None:
        make_Ak_fns_fn = make_Ak_fns
    if s2_fn is None:
        raise ValueError(
            "run_posterior_sampling_alg4 requires s2_fn: the prior covariance "
            "surrogate S is an input of the algorithm (draft 'Require:'), and "
            "defaulting it in code would put back a tuned parameter.")
    # h0 / h1 / ridge_rel have no meaning here. Swallowing them in **unused_kw
    # would be a silent no-op of exactly the kind the config contract exists to
    # stop, so they are rejected loudly instead.
    dead = sorted(k for k in ("h0", "h1", "ridge_rel", "cg_max_iter_l14",
                              "anchor_lambda", "anchor_P") if k in unused_kw)
    if dead:
        raise TypeError(
            f"run_posterior_sampling_alg4 got {dead}: Algorithm 4 has no step "
            "size (Sec. 6.3) and needs no ridge (Prop. 4(a)). Passing these "
            "would change nothing, so they are refused rather than ignored.")
    if "g_bypass_stage3" in unused_kw:
        raise TypeError(
            "run_posterior_sampling_alg4 got g_bypass_stage3: no such thing "
            "exists any more. The stage-3 identity bypass was removed from "
            "apply_G (2026-08-24); G is the same real projection at every "
            "stage, so there is nothing to toggle.")

    B = gt.shape[0]
    # num_langevin (draft symbol S, inner iterations) is a scalar or a
    # length-num_stages list; resolved per stage below via _per_stage,
    # the same convention ode_steps_per_stage already follows.
    L = int(cg_max_iter)                 # draft symbol L

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
    frame = -1          # global step index across all stages; the "frame id"
                        # the trajectory frames are numbered by
    # ── main loop ──────────────────────────────────────────────────────
    for si in range(num_stages):                                     # l.2
        sc = copy.deepcopy(scheduler)
        ode_steps_si = int(float(_per_stage(ode_steps_per_stage, si, num_stages)))
        # l.9 count: scalar, per-stage entry, or a per-stage LIST of per-frame
        # counts (len == that stage's ode steps) — pure schedule either way.
        S_it_entry = _per_stage(num_langevin, si, num_stages)
        if isinstance(S_it_entry, (list, tuple)) and \
                len(S_it_entry) != ode_steps_si:
            raise ValueError(
                f"per-frame num_langevin for stage {si} has "
                f"{len(S_it_entry)} entries, stage has {ode_steps_si} steps")
        sc.set_timesteps(ode_steps_si, si, device=device, shift=shift)
        s_k = float(sc.start_t[si])
        e_k = float(sc.end_t[si])
        # G is the real projection up(down(x)) at every stage -- the former
        # stage-3 identity bypass was removed from apply_G (2026-08-24), so
        # stage_idx is a no-op and this value is inert; None states the intent.
        # Prop. 4(a) keeps M_tau^den positive definite where H_tau = s_k G is
        # rank deficient. The stage-3 gamma^2 entries were measured under this
        # convention (gamma2_meas_alg4.json).
        eff_si = None

        if si > 0:                                                   # l.18: U^(1)
            h *= 2
            w *= 2
            x1 = F.interpolate(x1, size=(h, w), mode="nearest")
        x0 = randn_like_cpu(pyr[si])                                 # l.3 / l.18

        Ak, ATk = make_Ak_fns_fn(operator, y, (B, 3, h, w), device)

        size_tensor, rope_pos = base.rope_for(model, h, w, device)
        gamma2_stage = gamma2_tab[str(si)]
        # nearest, to keep the hole indicator 0/1 at the stage resolution
        hole_k = None if hole_mask is None else F.interpolate(
            hole_mask.to(device).float(), size=(h, w), mode="nearest")

        for step_idx, T in enumerate(sc.Timesteps):                  # l.4
            frame += 1
            S_it = int(float(S_it_entry[step_idx])) \
                if isinstance(S_it_entry, (list, tuple)) \
                else int(float(S_it_entry))
            tau = float(sc.t[step_idx])
            sigma_tau = compute_sigma_tau(tau, s_k, e_k)             # l.5
            velocity_fn = make_velocity_fn(
                model, T, prompt_embeds, size_tensor, rope_pos,
                do_cfg, guidance_scale, si,
            )
            gamma2 = float(gamma2_stage.get(
                f"{round(tau, 6)}", list(gamma2_stage.values())[step_idx]))
            x1_hat = None
            s2 = float("nan")
            cg_iters, cg_resid = 0, 0.0
            # sigma_tau == 0 exactly at tau=1 of the final stage (the draft
            # notes this in Sec. 2.1), where 1/sigma_tau^2 does not exist. The
            # threshold is the repo's existing one so the schedules stay
            # comparable across samplers; it is NOT a "skip Langevin" test
            # here, since Algorithm 4 has no Langevin step.
            if sigma_tau >= sigma_min:
                # s2_fn may return the isotropic variance (a float — the
                # original path, bit-identical through ScalarSOp) or an
                # SOperator for a structured S. Either way the (12)/(22)
                # math below is unchanged; only who applies S^-1 differs.
                s_ret = s2_fn(si, float(sigma_tau))
                s_op = s_ret if isinstance(s_ret, SOperator) else ScalarSOp(float(s_ret))
                s2 = float(s_op.scalar_equiv)
                # l.6 / l.7 — built once per tau, shared by every inner iteration
                M_den, Cinv, inv_e2, inv_s2, inv_S = make_M_tau_den(
                    Ak, ATk, eta, sigma_tau, tau, s_k, e_k, eff_si, s_op)
                # Line 13 is an exact draw only if its solve converges
                # (Lemma 9). With A_k = A . U^(K-1-k) plain CG needs 57-76
                # iterations at stages 0-2 and the inherited cap is 50, so it
                # was silently truncating. Jacobi preconditioning solves the
                # SAME system -- the draw is unchanged -- in 18-22.
                if isinstance(s_op, SpectralSOp):
                    # The all-ones probe reads S^-1's DC response 1/P(0)
                    # instead of its true constant diagonal mean_w 1/P(w),
                    # which broke the preconditioner wherever S^-1 dominates
                    # M (early frames) and left PCG truncated at the cap.
                    # Rebuild the probe with the exact S^-1 diagonal.
                    _ones = torch.ones_like(x1)
                    _d = (M_den(_ones) - s_op.apply_S_inv(_ones)
                          + s_op.inv_diag_mean() * _ones)
                    if bool(torch.isfinite(_d).all()) and float(_d.min()) > 1e-12:
                        _inv_d = 1.0 / _d
                        M_inv = (lambda r, _inv_d=_inv_d: _inv_d * r)
                    else:
                        M_inv = None
                else:
                    M_inv = make_jacobi_precond(M_den, x1.shape, device)
                if diag is not None:
                    diag.on_frame_setup(
                        frame=frame, stage=si, step=step_idx, tau=tau,
                        s_k=s_k, e_k=e_k, sigma_tau=float(sigma_tau),
                        gamma2=gamma2, s2=s2, eta=eta, eff_si=eff_si,
                        A_fn=Ak, AT_fn=ATk, shape=x1.shape, device=device)

                if diag_xi0_off_from_frame is not None \
                        and frame >= int(diag_xi0_off_from_frame):
                    x0 = torch.zeros_like(x0)          # probe only
                x_tau = apply_H_tau(x1, tau, s_k, e_k, eff_si) + sigma_tau * x0  # l.8

                for s in range(S_it):                                # l.9
                    x1_in = x1 if diag is None else x1.clone()
                    with torch.no_grad():
                        v = velocity_fn(x_tau)                       # l.10
                    # l.11 (19): the clean endpoint, one solve
                    x1_hat = clean_endpoint_solve(
                        x_tau, v, sigma_tau, s_k, e_k, tau, gamma2, eff_si,
                        cg_tol, cg_max_iter_endpoint, x1_warm=x1.clone())

                    # ── Block 1: exact draw of x1 from pi(x1 | x_tau, y) ──
                    xi_y = randn_like_cpu(y)                         # l.12
                    xi_h = randn_like_cpu(x1)
                    xi_s = randn_like_cpu(x1)
                    xi_h_drawn = xi_h if diag_xi0_use_xih else None
                    if diag_noise_off and si >= diag_noise_off_from_stage:
                        if "xi_y" in diag_noise_off:
                            xi_y = torch.zeros_like(xi_y)
                        if "xi_h" in diag_noise_off:
                            xi_h = torch.zeros_like(xi_h)
                        if "xi_s" in diag_noise_off:
                            xi_s = torch.zeros_like(xi_s)
                    # Lemma 9 with R1 = A/eta, R2 = H_tau/sigma_tau, R3 = S^-1/2:
                    # zeta = R1^T xi_y + R2^T xi_h + R3^T xi_s has covariance
                    # exactly M_tau^den, so the solve below IS a draw from
                    # N(M^-1 b, M^-1). No ridge term is needed anywhere.
                    b_tilde = (inv_e2 * ATk(y) + Cinv(x1_hat)
                               + (1.0 / eta) * ATk(xi_y)
                               + (1.0 / float(sigma_tau)) * apply_H_tau(
                                   xi_h, tau, s_k, e_k, eff_si)
                               + s_op.apply_S_inv_sqrt(xi_s))        # l.13 (22)
                    x1, cg_it, cg_rel = pcg_solve(
                        M_den, b_tilde, M_inv, x0=x1.clone(), tol=cg_tol,
                        max_iter=L)
                    cg_iters = max(cg_iters, cg_it)
                    cg_resid = max(cg_resid, cg_rel)

                    # ── Block 2: exact draw of x_tau, no solve, no step size ──
                    xi_0 = randn_like_cpu(x0)                        # l.14
                    if diag_xi0_use_xih:
                        xi_0 = xi_h_drawn               # probe only
                    if diag_noise_off and si >= diag_noise_off_from_stage \
                            and "xi_0" in diag_noise_off:
                        xi_0 = torch.zeros_like(xi_0)   # probe only
                    if diag_xi0_off_from_frame is not None \
                            and frame >= int(diag_xi0_off_from_frame):
                        xi_0 = torch.zeros_like(xi_0)   # probe only
                    if diag_block2_langevin is not None:
                        # probe: Algorithm 2's Block 2 on x0 in place of (23).
                        # x0_hat from the l.10 velocity at the CURRENT x_tau
                        # (l.11); x0 recomputed with the Block-1 x1 (l.15);
                        # Langevin step with the xi_0 drawn above (l.17).
                        h0 = float(diag_block2_langevin)
                        x0_hat = score_solve(x_tau, v, s_k, e_k, tau, gamma2,
                                             eff_si, cg_tol, L)
                        x0_cur = (x_tau - apply_H_tau(x1, tau, s_k, e_k, eff_si)) \
                            / float(sigma_tau)
                        x0_new = x0_cur - (h0 / 2.0) * (
                            x0_cur + float(diag_block2_langevin_sign) * x0_hat) \
                            + math.sqrt(h0) * xi_0
                        x_tau = apply_H_tau(x1, tau, s_k, e_k, eff_si) + \
                            float(sigma_tau) * x0_new
                    else:
                        x_tau = apply_H_tau(x1, tau, s_k, e_k, eff_si) + \
                            float(sigma_tau) * xi_0                  # (23)
                    if diag is not None:
                        diag.on_inner(
                            frame=frame, stage=si, step=step_idx, inner=s,
                            tau=tau, sigma_tau=float(sigma_tau),
                            x1_in=x1_in, x1_hat=x1_hat, x1_out=x1,
                            x_tau=x_tau, v=v, gt_k=pyr[si], hole_k=hole_k,
                            s_k=s_k, e_k=e_k, eff_si=eff_si)

                # l.16 — leave the coordinates. Provably equal to the last
                # xi_0 (see A5); computed from the state as the listing writes it.
                x0 = (x_tau - apply_H_tau(x1, tau, s_k, e_k, eff_si)) / float(sigma_tau)

            row = dict(stage=si, step=step_idx, tau=tau,
                       sigma_tau=float(sigma_tau),
                       s2=s2, gamma2=gamma2,
                       mse_x1=float(((x1 - pyr[si]) ** 2).mean()),
                       meas_resid=measurement_residual(Ak, x1, y, eta),
                       x0_rms=float((x0 ** 2).mean().sqrt()),
                       # Never let a non-converged Block-1 solve pass silently
                       # again: these two columns are the receipt.
                       blk1_cg_iters=cg_iters, blk1_cg_resid=cg_resid,
                       blk1_cg_converged=int(cg_iters == 0 or cg_resid < cg_tol))
            if hole_k is not None:
                row["mse_hole"] = mse_masked(x1, pyr[si], hole_k)
                row["mse_obs"] = mse_masked(x1, pyr[si], 1.0 - hole_k)
            row["frame"] = frame
            rows.append(row)
            if diag is not None:
                diag.on_frame_end(frame=frame, row=row, x1=x1, x0=x0,
                                  x1_hat=x1_hat, gt_k=pyr[si], hole_k=hole_k,
                                  tau=tau, sigma_tau=float(sigma_tau),
                                  s_k=s_k, e_k=e_k, eff_si=eff_si)
            if record_trajectory:
                x_tau_rec = apply_H_tau(x1, tau, s_k, e_k, eff_si) + \
                    float(sigma_tau) * x0
                traj.append((x_tau_rec[0].cpu(), x1[0].cpu(),
                             (x1_hat if x1_hat is not None else x1)[0].cpu()))

    # POST-sampling terminal projection (inpainting: snap observed pixels to y).
    # Pipeline convention (CONSTRAINTS: inpainting tr=1, blur/SR tr=0); applied
    # AFTER the loop, so per-step rows/traj metrics above are unaffected.
    if terminal_replace_weight > 0:
        m = operator.get_mask(x=y).float().to(x1.device)
        x1 = terminal_replace_weight * (m * y + (1.0 - m) * x1) + \
            (1.0 - terminal_replace_weight) * x1
    return x1, rows, traj


# ── Algorithm 4: making the Block-1 solve actually converge ────────────────
# Line 13 solves M_tau^den x1 = b + zeta. Lemma 9's guarantee -- that the
# solution is a draw from N(M^-1 b, M^-1) -- holds for the EXACT solution. A
# truncated CG returns something else, and the repo's inherited cap of L = 50
# is not enough here: with A_k = A . U^(K-1-k) the upsample destroys the
# two-eigenvalue structure that makes the other solves terminate immediately,
# and plain CG needs 60-77 iterations at stages 1-2 to reach tol 1e-5.
#
# The fix does not change what is being solved. A preconditioner alters only
# the path CG takes to the same x, so the sampled distribution is untouched;
# what changes is whether the iteration actually gets there within the cap.
#
# NOTE these are additions. cg_solve, score_solve and the alg2 / wv / reg
# samplers are untouched, and line 11 keeps plain cg_solve because its operator
# is a polynomial in G, has at most two distinct eigenvalues, and therefore
# terminates in <= 2 iterations exactly (measured: 1 or 2 at every stage).

def make_jacobi_precond(M_fn, shape, device, floor=1e-12):
    """Diagonal (Jacobi) preconditioner for an SPD matrix-free operator.

    Uses M applied to the all-ones vector, which is the exact diagonal when M
    is diagonal -- the inpainting case, where A_k^T A_k counts observed
    children per coarse pixel -- and the row sums otherwise. That distinction
    does not affect correctness: ANY symmetric positive-definite preconditioner
    leaves the solution of M x = b unchanged, so the draw stays exact either
    way. It only affects how fast CG gets there.

    Returns None if the probe is not strictly positive, so the caller falls
    back to unpreconditioned CG rather than dividing by something invalid.
    """
    d = M_fn(torch.ones(shape, device=device))
    if not bool(torch.isfinite(d).all()) or float(d.min()) <= floor:
        return None
    inv = 1.0 / d
    return lambda r: inv * r


def pcg_solve(A_fn, b, M_inv=None, x0=None, tol=1e-5, max_iter=50):
    """Preconditioned CG. With M_inv=None this is cg_solve's iteration.

    Returns (x, iters, rel_residual) -- the caller can tell whether the solve
    converged instead of assuming it did, which is the failure this exists to
    stop.
    """
    B = b.shape[0]

    def dot(u, v):
        return (u * v).reshape(B, -1).sum(dim=1)

    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - A_fn(x)
    z = r if M_inv is None else M_inv(r)
    p = z.clone()
    rz = dot(r, z)
    b_norm = dot(b, b).sqrt().clamp(min=1e-12)
    rel = float((dot(r, r).sqrt() / b_norm).max())
    it = 0
    for it in range(1, int(max_iter) + 1):
        Ap = A_fn(p)
        alpha = (rz / dot(p, Ap).clamp(min=1e-12)).reshape(B, 1, 1, 1)
        x = x + alpha * p
        r = r - alpha * Ap
        rel = float((dot(r, r).sqrt() / b_norm).max())
        if rel < tol:
            break
        z = r if M_inv is None else M_inv(r)
        rz_new = dot(r, z)
        p = z + (rz_new / rz.clamp(min=1e-12)).reshape(B, 1, 1, 1) * p
        rz = rz_new
    return x, it, rel
