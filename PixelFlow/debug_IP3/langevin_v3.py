"""
PRINCIPLE v3 methods — null-space prior boost + Tikhonov one-shot.
Does NOT modify any existing code. Import and use from debug_IP3/.

Lesson from v3a failure: applying g_prior directly on observed pixels
DESTROYS data consistency (CG damps prior 450x on observed — essential,
not a bug). The correct fix is null-space targeted or Tikhonov.

Methods:
1. null_boost: standard coupled CG (preserves data consistency) +
   EXTRA prior boost only on unobserved pixels (null space of A).
2. reduced_lambda: lower lambda_reg → prior influence on unobserved
   increases (1/lambda_reg scaling). Simplest single-number change.
3. tikhonov: one-shot CG per ODE step, no inner loop.
   Exact solution: observed→data, unobserved→model prediction.
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


# ═══════════════════════════════════════════════════════════════════════════
# Method 1: Tikhonov one-shot (no inner loop, no renoising)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def tikhonov_update(x1_hat, A_k_fn, AT_k_fn, b, eta, lambda_tik,
                    cg_tol=1e-5, cg_max_iter=100):
    """One-shot Tikhonov regularization.

    Solves: (A^T A/η² + λ I) x1 = A^T b/η² + λ x1_hat

    On observed pixels:  x1 ≈ (b + η²λ x1_hat) / (1 + η²λ)
    On unobserved pixels: x1 = x1_hat  (pure model prediction)

    No step size, no inner loop, no renoising. Mathematically clean.
    """
    inv_eta_sq = 1.0 / (eta ** 2)

    def system(x):
        return inv_eta_sq * AT_k_fn(A_k_fn(x)) + lambda_tik * x

    rhs = inv_eta_sq * AT_k_fn(b) + lambda_tik * x1_hat
    x1 = cg_solve(system, rhs, tol=cg_tol, max_iter=cg_max_iter)

    residual = b - A_k_fn(x1)
    log = {
        "loss": float((residual ** 2).sum()),
        "l3": float((b - A_k_fn(x1)).pow(2).sum()),
        "grad_data_norm": 0.0,
        "grad_prior_norm": 0.0,
        "prior_data_ratio": 0.0,
    }
    return x1, log


# ═══════════════════════════════════════════════════════════════════════════
# Method 2: Null-space boosted Langevin
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def nullboost_langevin_step(x1_k, eps_k, x_tau_k, tau, s_k, e_k, sigma_tau,
                            velocity_fn, A_k_fn, AT_k_fn, b, eta,
                            h_x, h_eps, lambda_x, lambda_reg,
                            rho_s, rho_e, cg_tol, cg_max_iter,
                            stage_idx=None, x1_init_mode="model",
                            noise_scale=0.0,
                            sigma_ref_sq=0.01,
                            max_score_norm=500.0,
                            h_null=0.0):
    """Coupled Langevin step (same as baseline) + null-space prior boost.

    After the standard CG update, adds:
        x1 += h_null * null_mask * g_prior

    where null_mask = (I - A^T A) selects unobserved pixels.
    This boosts prior influence only where data term is zero.
    """
    # (a) velocity → score
    mu = velocity_fn(x_tau_k)
    xs_hat = x_tau_k - tau * mu
    xe_hat = x_tau_k + (1.0 - tau) * mu

    if x1_init_mode == "model":
        x1_hat = direct_estimate_x1(xs_hat, xe_hat, s_k, e_k)
    else:
        from ms_posterior_sampling_article_version_final_utils import wls_estimate_x1
        x1_hat = wls_estimate_x1(xs_hat, xe_hat, s_k, e_k,
                                  rho_s, rho_e, lambda_x, cg_tol, cg_max_iter,
                                  stage_idx=stage_idx)

    H_x1_hat = apply_H_tau(x1_hat, tau, s_k, e_k, stage_idx=stage_idx)

    # Soft-damped Tweedie score
    score_prec = 1.0 / (float(sigma_tau) ** 2 + sigma_ref_sq)
    s_flow = score_prec * (H_x1_hat - x_tau_k)

    # Score clip
    B = x_tau_k.shape[0]
    s_norm = s_flow.reshape(B, -1).norm(dim=1).reshape(B, 1, 1, 1).clamp(min=1e-8)
    clip_scale = (max_score_norm / s_norm).clamp(max=1.0)
    s_flow = s_flow * clip_scale

    # (b) Standard coupled gradient
    residual = b - A_k_fn(x1_k)
    g_data = (1.0 / eta ** 2) * AT_k_fn(residual)
    g_prior = apply_HT_tau(s_flow, tau, s_k, e_k, stage_idx=stage_idx)
    g_x1 = g_data + g_prior

    g_eps = sigma_tau * s_flow - eps_k

    # (c) Standard CG update (coupled, same as baseline)
    xi_1 = torch.randn_like(b) * noise_scale
    xi_2 = torch.randn_like(x1_k) * noise_scale
    def system(x):
        return (1.0 / eta ** 2) * AT_k_fn(A_k_fn(x)) + lambda_reg * x
    rhs = (h_x / 2) * g_x1 + math.sqrt(h_x) * (
        (1.0 / eta) * AT_k_fn(xi_1) + math.sqrt(lambda_reg) * xi_2)
    delta = cg_solve(system, rhs, tol=cg_tol, max_iter=cg_max_iter)
    x1_k = x1_k + delta

    # (d) ★ NULL-SPACE PRIOR BOOST ★
    # Compute null-space mask: (I - A^T A) applied to g_prior
    # For inpainting: A^T A = mask on observed, so null = (1 - mask) on unobserved
    if h_null > 0:
        # A^T(A(g_prior)) projects g_prior onto observed space
        # g_prior - A^T(A(g_prior)) = null-space component
        g_prior_null = g_prior - AT_k_fn(A_k_fn(g_prior))
        x1_k = x1_k + h_null * g_prior_null

    # (e) eps update
    x3 = torch.randn_like(eps_k) * noise_scale
    eps_k = eps_k + (h_eps / 2) * g_eps + math.sqrt(h_eps) * x3

    # (f) reconstruct
    x_tau_k = apply_H_tau(x1_k, tau, s_k, e_k, stage_idx=stage_idx) + sigma_tau * eps_k

    log = {
        "loss": float((residual ** 2).sum()),
        "l3": float((b - A_k_fn(x1_k)).pow(2).sum()),
        "grad_data_norm": float(g_data.norm()),
        "grad_prior_norm": float(g_prior.norm()),
        "prior_data_ratio": float(g_prior.norm() / g_data.norm().clamp(min=1e-8)),
        "delta_norm": float(delta.norm()),
        "null_boost_norm": float((h_null * g_prior_null).norm()) if h_null > 0 else 0.0,
        "clip_frac": float((clip_scale < 1.0).float().mean()),
    }
    return x1_k, eps_k, x_tau_k, log


@torch.no_grad()
def nullboost_sample(x1_init, eps_init, tau, s_k, e_k,
                     velocity_fn, A_k_fn, AT_k_fn,
                     y, sigma_n, h_x, h_epsilon,
                     lambda_x, lambda_reg, rho_s, rho_e,
                     cg_tol, cg_max_iter, num_Langevin, device,
                     stage_idx=None, x1_init_mode="model",
                     noise_scale=0.0,
                     sigma_ref_sq=0.01, max_score_norm=500.0,
                     h_null=0.0,
                     return_traj=False, record_every=1):
    """Null-space boosted refinement loop."""
    sigma_tau = compute_sigma_tau(tau, s_k, e_k)
    x1_k = x1_init.clone().detach().to(device)
    eps_k = eps_init.clone().detach().to(device)
    x_tau_k = apply_H_tau(x1_k, tau, s_k, e_k, stage_idx=stage_idx) + sigma_tau * eps_k

    traj, logs = [], []
    record_every = max(1, int(record_every))

    for i in range(int(num_Langevin)):
        x1_k, eps_k, x_tau_k, log = nullboost_langevin_step(
            x1_k, eps_k, x_tau_k, tau, s_k, e_k, sigma_tau,
            velocity_fn, A_k_fn, AT_k_fn, y, sigma_n,
            h_x, h_epsilon, lambda_x, lambda_reg,
            rho_s, rho_e, cg_tol, cg_max_iter,
            stage_idx=stage_idx, x1_init_mode=x1_init_mode,
            noise_scale=noise_scale,
            sigma_ref_sq=sigma_ref_sq,
            max_score_norm=max_score_norm,
            h_null=h_null)
        if return_traj and i % record_every == 0:
            traj.append(x1_k.detach().clone())
            logs.append(log)

    if return_traj:
        return x1_k.detach(), eps_k.detach(), traj, logs
    return x1_k.detach(), eps_k.detach()
