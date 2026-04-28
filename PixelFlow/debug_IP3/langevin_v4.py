"""
PRINCIPLE v4 — frequency-aware posterior sampling.

Key insight: the model CAN generate high-freq (SOTA FID), but the posterior
sampling algorithm doesn't extract it. Three orthogonal fixes:

1. x1_hat HF boost: unsharp-mask x1_hat before computing score → prior
   pushes toward sharper target
2. Score HF boost: amplify high-freq component of s_flow → g_prior has
   more high-freq energy → CG update has more high-freq on unobserved
3. Annealed noise: start with noise_scale > 0 to explore high-freq modes,
   anneal to 0 for refinement. Pure MAP (noise=0) collapses to smooth mode.
"""

import math
import torch
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ms_posterior_sampling_article_version_final_utils import (
    apply_G, apply_H_tau, apply_HT_tau, compute_sigma_tau,
    direct_estimate_x1, cg_solve,
)


def _unsharp_mask(x, boost, kernel_size=5):
    """Unsharp masking: x + boost * (x - blur(x)).
    Amplifies high frequencies by factor (1 + boost)."""
    if boost <= 0:
        return x
    C = x.shape[1]
    pad = kernel_size // 2
    k = torch.ones(C, 1, kernel_size, kernel_size, device=x.device, dtype=x.dtype)
    k = k / (kernel_size ** 2)
    blurred = F.conv2d(F.pad(x, [pad]*4, mode='reflect'), k, groups=C)
    return x + boost * (x - blurred)


@torch.no_grad()
def _langevin_step_v4(x1_k, eps_k, x_tau_k, tau, s_k, e_k, sigma_tau,
                      velocity_fn, A_k_fn, AT_k_fn, b, eta,
                      h_x, h_eps, lambda_x, lambda_reg,
                      rho_s, rho_e, cg_tol, cg_max_iter,
                      stage_idx=None, x1_init_mode="model",
                      noise_scale=0.0,
                      # ── v4 additions ──
                      x1hat_hf_boost=0.0,
                      score_hf_boost=0.0,
                      hf_kernel=5,
                      ):
    """PRINCIPLE v4 refinement step with frequency-aware enhancements."""

    sigma_ref_sq = 0.01
    score_precision = 1.0 / (float(sigma_tau) ** 2 + sigma_ref_sq)

    # (a) velocity → endpoints
    mu = velocity_fn(x_tau_k)
    xs_hat = x_tau_k - tau * mu
    xe_hat = x_tau_k + (1.0 - tau) * mu

    # (b) x1 estimate
    x1_hat = direct_estimate_x1(xs_hat, xe_hat, s_k, e_k)

    # ★ v4: sharpen x1_hat before score computation
    if x1hat_hf_boost > 0:
        x1_hat = _unsharp_mask(x1_hat, x1hat_hf_boost, hf_kernel)

    H_x1_hat = apply_H_tau(x1_hat, tau, s_k, e_k, stage_idx=stage_idx)

    # (c) Tweedie score
    s_flow = score_precision * (H_x1_hat - x_tau_k)

    # ★ v4: boost high-freq of score
    if score_hf_boost > 0:
        s_flow = _unsharp_mask(s_flow, score_hf_boost, hf_kernel)

    # (d) posterior gradients
    residual = b - A_k_fn(x1_k)
    g_x1 = (1.0 / eta ** 2) * AT_k_fn(residual) + apply_HT_tau(s_flow, tau, s_k, e_k, stage_idx=stage_idx)
    g_eps = sigma_tau * s_flow - eps_k

    # (e) CG-preconditioned update
    xi_1 = torch.randn_like(b) * noise_scale
    xi_2 = torch.randn_like(x1_k) * noise_scale
    def system(x):
        return (1.0 / eta ** 2) * AT_k_fn(A_k_fn(x)) + lambda_reg * x
    rhs = (h_x / 2) * g_x1 + math.sqrt(h_x) * (
        (1.0 / eta) * AT_k_fn(xi_1) + math.sqrt(lambda_reg) * xi_2)
    delta = cg_solve(system, rhs, tol=cg_tol, max_iter=cg_max_iter)
    x1_k = x1_k + delta

    # (f) eps update
    x3 = torch.randn_like(eps_k) * noise_scale
    eps_k = eps_k + (h_eps / 2) * g_eps + math.sqrt(h_eps) * x3

    # (g) reconstruct
    x_tau_k = apply_H_tau(x1_k, tau, s_k, e_k, stage_idx=stage_idx) + sigma_tau * eps_k

    log = {
        "loss": float((residual ** 2).sum()),
        "l3": float((b - A_k_fn(x1_k)).pow(2).sum()),
        "grad_norm": float(g_x1.norm()),
        "delta_norm": float(delta.norm()),
    }
    return x1_k, eps_k, x_tau_k, log


@torch.no_grad()
def v4_langevin_sample(x1_init, eps_init, tau, s_k, e_k,
                       velocity_fn, A_k_fn, AT_k_fn,
                       y, sigma_n, h_x, h_epsilon,
                       lambda_x, lambda_reg, rho_s, rho_e,
                       cg_tol, cg_max_iter, num_Langevin, device,
                       stage_idx=None, x1_init_mode="model",
                       # ── v4 params ──
                       noise_init=0.0,
                       noise_anneal_frac=0.7,  # fraction of steps to anneal over
                       x1hat_hf_boost=0.0,
                       score_hf_boost=0.0,
                       hf_kernel=5,
                       return_traj=False, record_every=1):
    """v4 refinement loop with annealed noise + HF boost."""
    sigma_tau = compute_sigma_tau(tau, s_k, e_k)
    x1_k = x1_init.clone().detach().to(device)
    eps_k = eps_init.clone().detach().to(device)
    x_tau_k = apply_H_tau(x1_k, tau, s_k, e_k, stage_idx=stage_idx) + sigma_tau * eps_k

    traj, logs = [], []
    record_every = max(1, int(record_every))
    N = int(num_Langevin)
    anneal_steps = max(1, int(N * noise_anneal_frac))

    for i in range(N):
        # ★ Annealed noise schedule
        if noise_init > 0 and i < anneal_steps:
            noise_scale = noise_init * (1.0 - i / anneal_steps)
        else:
            noise_scale = 0.0

        x1_k, eps_k, x_tau_k, log = _langevin_step_v4(
            x1_k, eps_k, x_tau_k, tau, s_k, e_k, sigma_tau,
            velocity_fn, A_k_fn, AT_k_fn, y, sigma_n,
            h_x, h_epsilon, lambda_x, lambda_reg,
            rho_s, rho_e, cg_tol, cg_max_iter,
            stage_idx=stage_idx, x1_init_mode=x1_init_mode,
            noise_scale=noise_scale,
            x1hat_hf_boost=x1hat_hf_boost,
            score_hf_boost=score_hf_boost,
            hf_kernel=hf_kernel)

        if return_traj and i % record_every == 0:
            traj.append(x1_k.detach().clone())
            logs.append(log)

    if return_traj:
        return x1_k.detach(), eps_k.detach(), traj, logs
    return x1_k.detach(), eps_k.detach()
