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
    has_ker = stage_idx != 3                   # G = I at stage 3, so ker(G) = {0}
    if lam_range <= 0 or (has_ker and lam_ker <= 0):
        raise ValueError(
            f"H_tau is singular at tau={tau}, s_k={s_k}: x1_hat is undefined. "
            "Start the grid at the first tau > 0.")
    Gx = apply_G(x, stage_idx=stage_idx)
    out = Gx / lam_range
    return out + (x - Gx) / lam_ker if has_ker else out


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


def make_M_tau_den(A_fn, AT_fn, eta, sigma_tau, tau, s_k, e_k, eff_si, s2):
    """draft l.7 / (12) + (21). Returns (M_fn, Cinv_fn, inv_e2, inv_s2, inv_S).

        C^-1     = H_tau^T H_tau / sigma_tau^2 + S^-1          (12)
        M_tau^den = A_k^T A_k / eta^2 + C^-1                    (21)

    with the isotropic surrogate S = s2 * I, so S^-1 x = x / s2 and the
    Lemma-9 factor S^-1/2 is a division by sqrt(s2) -- no square root of a
    matrix is formed (draft Sec. 6.1).

    NO RIDGE and no power iteration, unlike make_M_tau: by Prop. 4(a),
    C^-1 >= S^-1 > 0 for every tau in [0,1] including tau=0, where the first
    term degenerates to s_k^2 G^2 / sigma_tau^2 and vanishes on ker(G). The
    surrogate supplies the entire precision in exactly the directions the
    interpolant does not constrain.
    """
    if not (s2 > 0):
        raise ValueError(f"S_prior s2 must be > 0 (got {s2!r}); "
                         "S^-1 is what makes M_tau^den positive definite at tau=0")
    inv_e2, inv_s2 = 1.0 / eta ** 2, 1.0 / float(sigma_tau) ** 2
    inv_S = 1.0 / float(s2)

    def Cinv(x):
        return inv_s2 * apply_H_tau(
            apply_H_tau(x, tau, s_k, e_k, eff_si), tau, s_k, e_k, eff_si) + inv_S * x

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
    num_langevin=10,                 # draft S_it: INNER iterations (l.9).
                                     # The name is kept so task_kw/config keys
                                     # stay shared with the other samplers; it
                                     # is not a count of Langevin steps here
                                     # because there are none.
    ode_steps_per_stage=10, shift=1.0,
    # Structural / physical
    guidance_scale=0.0, class_label=10,
    g_bypass_stage3=True,
    cg_tol=1e-5, cg_max_iter=50,     # draft: CG iterations L
    make_Ak_fns_fn=None,
    seed=42,
    record_trajectory=False,
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

    B = gt.shape[0]
    S_it = int(float(num_langevin))      # draft symbol S (inner iterations)
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

    # ── main loop ──────────────────────────────────────────────────────
    for si in range(num_stages):                                     # l.2
        sc = copy.deepcopy(scheduler)
        ode_steps_si = int(float(_per_stage(ode_steps_per_stage, si, num_stages)))
        sc.set_timesteps(ode_steps_si, si, device=device, shift=shift)
        s_k = float(sc.start_t[si])
        e_k = float(sc.end_t[si])
        eff_si = si if g_bypass_stage3 else None

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
            # sigma_tau == 0 exactly at tau=1 of the final stage (the draft
            # notes this in Sec. 2.1), where 1/sigma_tau^2 does not exist. The
            # threshold is the repo's existing one so the schedules stay
            # comparable across samplers; it is NOT a "skip Langevin" test
            # here, since Algorithm 4 has no Langevin step.
            if sigma_tau >= sigma_min:
                s2 = float(s2_fn(si, float(sigma_tau)))
                # l.6 / l.7 — built once per tau, shared by every inner iteration
                M_den, Cinv, inv_e2, inv_s2, inv_S = make_M_tau_den(
                    Ak, ATk, eta, sigma_tau, tau, s_k, e_k, eff_si, s2)
                inv_S_half = 1.0 / math.sqrt(s2)

                x_tau = apply_H_tau(x1, tau, s_k, e_k, eff_si) + sigma_tau * x0  # l.8

                for s in range(S_it):                                # l.9
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
                    # Lemma 9 with R1 = A/eta, R2 = H_tau/sigma_tau, R3 = S^-1/2:
                    # zeta = R1^T xi_y + R2^T xi_h + R3^T xi_s has covariance
                    # exactly M_tau^den, so the solve below IS a draw from
                    # N(M^-1 b, M^-1). No ridge term is needed anywhere.
                    b_tilde = (inv_e2 * ATk(y) + Cinv(x1_hat)
                               + (1.0 / eta) * ATk(xi_y)
                               + (1.0 / float(sigma_tau)) * apply_H_tau(
                                   xi_h, tau, s_k, e_k, eff_si)
                               + inv_S_half * xi_s)                  # l.13 (22)
                    x1 = cg_solve(M_den, b_tilde, x0=x1.clone(), tol=cg_tol,
                                  max_iter=L)

                    # ── Block 2: exact draw of x_tau, no solve, no step size ──
                    xi_0 = randn_like_cpu(x0)                        # l.14
                    x_tau = apply_H_tau(x1, tau, s_k, e_k, eff_si) + \
                        float(sigma_tau) * xi_0                      # (23)

                # l.16 — leave the coordinates. Provably equal to the last
                # xi_0 (see A5); computed from the state as the listing writes it.
                x0 = (x_tau - apply_H_tau(x1, tau, s_k, e_k, eff_si)) / float(sigma_tau)

            row = dict(stage=si, step=step_idx, tau=tau,
                       sigma_tau=float(sigma_tau),
                       s2=s2, gamma2=gamma2,
                       mse_x1=float(((x1 - pyr[si]) ** 2).mean()),
                       meas_resid=measurement_residual(Ak, x1, y, eta),
                       x0_rms=float((x0 ** 2).mean().sqrt()))
            if hole_k is not None:
                row["mse_hole"] = mse_masked(x1, pyr[si], hole_k)
                row["mse_obs"] = mse_masked(x1, pyr[si], 1.0 - hole_k)
            rows.append(row)
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
