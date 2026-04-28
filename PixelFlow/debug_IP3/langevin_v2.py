"""
PRINCIPLE v2 Langevin step — standalone module for testing improvements.
Does NOT modify any existing code. Import and use directly.

Changes vs current _langevin_step:
1. Soft sigma damping: 1/(sigma_tau² + sigma_ref²) replaces hard floor
2. Tau-dependent rho_s: rho_s * tau² downweights unreliable x_start_hat
3. Score norm clipping: prevents catastrophic updates from bad x1_hat
4. Gradient norm logging: track prior/data balance at every step
"""

import math
import torch
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ms_posterior_sampling_article_version_final_utils import (
    apply_G, apply_H_tau, apply_HT_tau, compute_sigma_tau,
    direct_estimate_x1, wls_estimate_x1, cg_solve, make_Ak_fns,
    make_velocity_fn, sample_block_noise, get_stage_inference_steps,
)


@torch.no_grad()
def langevin_step_v2(x1_k, eps_k, x_tau_k, tau, s_k, e_k, sigma_tau,
                     velocity_fn, A_k_fn, AT_k_fn, b, eta,
                     h_x, h_eps, lambda_x, lambda_reg,
                     rho_s, rho_e, cg_tol, cg_max_iter,
                     stage_idx=None, x1_init_mode="model",
                     noise_scale=0.0,
                     # ── v2 additions ──
                     sigma_ref=0.1,
                     max_score_norm=100.0,
                     rho_s_tau_scaling=True,
                     ):
    """
    PRINCIPLE v2 refinement step.

    New params:
        sigma_ref: soft damping reference. score_precision = 1/(sigma_tau² + sigma_ref²)
        max_score_norm: clip score norm to this value
        rho_s_tau_scaling: if True, rho_s *= tau² (downweight x_start_hat early)
    """

    # ── (a) velocity → endpoints ──
    mu = velocity_fn(x_tau_k)
    xs_hat = x_tau_k - tau * mu
    xe_hat = x_tau_k + (1.0 - tau) * mu

    # ── (b) x1 estimate with tau-dependent rho ──
    rho_s_eff = rho_s * (tau ** 2) if rho_s_tau_scaling else rho_s
    rho_e_eff = rho_e

    if x1_init_mode == "model" or rho_s_eff < 1e-6:
        x1_hat = direct_estimate_x1(xs_hat, xe_hat, s_k, e_k)
    else:
        x1_hat = wls_estimate_x1(xs_hat, xe_hat, s_k, e_k,
                                  rho_s_eff, rho_e_eff, lambda_x,
                                  cg_tol, cg_max_iter, stage_idx=stage_idx)

    H_x1_hat = apply_H_tau(x1_hat, tau, s_k, e_k, stage_idx=stage_idx)

    # ── (c) Tweedie score with SOFT damping ──
    # Instead of: 1/max(sigma_tau, floor)²
    # Use:        1/(sigma_tau² + sigma_ref²)
    score_precision = 1.0 / (float(sigma_tau) ** 2 + sigma_ref ** 2)
    s_flow = score_precision * (H_x1_hat - x_tau_k)

    # Score norm clipping
    B = x_tau_k.shape[0]
    score_norm = s_flow.reshape(B, -1).norm(dim=1).reshape(B, 1, 1, 1).clamp(min=1e-8)
    clip_scale = (max_score_norm / score_norm).clamp(max=1.0)
    s_flow = s_flow * clip_scale

    # ── (d) posterior gradients ──
    residual = b - A_k_fn(x1_k)
    g_data = (1.0 / eta ** 2) * AT_k_fn(residual)
    g_prior = apply_HT_tau(s_flow, tau, s_k, e_k, stage_idx=stage_idx)
    g_x1 = g_data + g_prior

    g_eps = sigma_tau * s_flow - eps_k

    # ── (e) CG-preconditioned update ──
    xi_1 = torch.randn_like(b) * noise_scale
    xi_2 = torch.randn_like(x1_k) * noise_scale
    def system(x):
        return (1.0 / eta ** 2) * AT_k_fn(A_k_fn(x)) + lambda_reg * x
    rhs = (h_x / 2) * g_x1 + math.sqrt(h_x) * (
        (1.0 / eta) * AT_k_fn(xi_1) + math.sqrt(lambda_reg) * xi_2)
    delta = cg_solve(system, rhs, tol=cg_tol, max_iter=cg_max_iter)
    x1_k = x1_k + delta

    # ── (f) eps update ──
    x3 = torch.randn_like(eps_k) * noise_scale
    eps_k = eps_k + (h_eps / 2) * g_eps + math.sqrt(h_eps) * x3

    # ── (g) reconstruct ──
    x_tau_k = apply_H_tau(x1_k, tau, s_k, e_k, stage_idx=stage_idx) + sigma_tau * eps_k

    log = {
        "loss": float((residual ** 2).sum()),
        "l3": float((b - A_k_fn(x1_k)).pow(2).sum()),
        "grad_data_norm": float(g_data.norm()),
        "grad_prior_norm": float(g_prior.norm()),
        "prior_data_ratio": float(g_prior.norm() / g_data.norm().clamp(min=1e-8)),
        "score_precision": float(score_precision),
        "score_norm_raw": float(score_norm.mean()),
        "clip_fraction": float((clip_scale < 1.0).float().mean()),
        "delta_norm": float(delta.norm()),
        "rho_s_eff": float(rho_s_eff),
        "z_mean": float(x1_k.mean()),
        "z_std": float(x1_k.std()),
    }
    return x1_k, eps_k, x_tau_k, log


def langevin_sample_v2(x1_init, eps_init, tau, s_k, e_k,
                       velocity_fn, A_k_fn, AT_k_fn,
                       y, sigma_n, h_x, h_epsilon,
                       lambda_x, lambda_reg, rho_s, rho_e,
                       cg_tol, cg_max_iter, num_Langevin, device,
                       stage_idx=None, x1_init_mode="model",
                       noise_scale=0.0,
                       sigma_ref=0.1,
                       max_score_norm=100.0,
                       rho_s_tau_scaling=True,
                       return_traj=False, record_every=1):
    """PRINCIPLE v2 refinement loop."""
    sigma_tau = compute_sigma_tau(tau, s_k, e_k)
    x1_k = x1_init.clone().detach().to(device)
    eps_k = eps_init.clone().detach().to(device)
    x_tau_k = apply_H_tau(x1_k, tau, s_k, e_k, stage_idx=stage_idx) + sigma_tau * eps_k

    traj, logs = [], []
    record_every = max(1, int(record_every))

    for i in range(int(num_Langevin)):
        x1_k, eps_k, x_tau_k, log = langevin_step_v2(
            x1_k, eps_k, x_tau_k, tau, s_k, e_k, sigma_tau,
            velocity_fn, A_k_fn, AT_k_fn, y, sigma_n,
            h_x, h_epsilon, lambda_x, lambda_reg,
            rho_s, rho_e, cg_tol, cg_max_iter,
            stage_idx=stage_idx, x1_init_mode=x1_init_mode,
            noise_scale=noise_scale,
            sigma_ref=sigma_ref,
            max_score_norm=max_score_norm,
            rho_s_tau_scaling=rho_s_tau_scaling)
        if return_traj and i % record_every == 0:
            traj.append(x1_k.detach().clone())
            logs.append(log)

    if return_traj:
        return x1_k.detach(), eps_k.detach(), traj, logs
    return x1_k.detach(), eps_k.detach()
