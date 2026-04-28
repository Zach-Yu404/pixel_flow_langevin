"""
debug_IP4 / langevin_v5.py

Extends principle_langevin_sample with axes NOT covered by IP3:
  - per-stage schedules: lambda_reg, h_x, h_epsilon, num_langevin, noise_scale
  - mask-aware h_x (different step size inside/outside the observation mask)
  - CFG guidance_scale > 0 (never tested in IP3)
  - DPS gradient pre-kick before Langevin (hybrid)
  - terminal replacement (observed <- y at stage 3, end of chain)
  - optional: post-Langevin soft replacement at stage 3 with mixing weight

All math stays in the PRINCIPLE pseudo-code framework from
ms_posterior_sampling_article_version_final.py. We only parameterize the knobs.
"""

import math
import torch
import torch.nn.functional as F

from ms_posterior_sampling_article_version_final_utils import (
    apply_G, apply_H_tau, apply_HT_tau, compute_sigma_tau,
    direct_estimate_x1, wls_estimate_x1, cg_solve,
)


def _per_stage(val, stage_idx, default):
    """Resolve a scalar / list value at a stage."""
    if val is None:
        return default
    if isinstance(val, (list, tuple)):
        return float(val[stage_idx])
    return float(val)


@torch.no_grad()
def langevin_step_v5(
    x1_k, eps_k, x_tau_k, tau, s_k, e_k, sigma_tau,
    velocity_fn, A_k_fn, AT_k_fn, y, eta,
    h_x, h_eps, lambda_x, lambda_reg,
    rho_s, rho_e, cg_tol, cg_max_iter,
    stage_idx=None, x1_init_mode="model",
    noise_scale=0.0,
    mask_k=None, h_x_obs_ratio=1.0,
    lambda_prox=0.0,
    skip_eps_update=False,
    lambda_eps_norm=0.0,        # N1: ||eps||² normalizer coefficient
    mask_aware_eps=False,       # N2: in unobs region, drop Tweedie term (only -eps prior)
    lambda_eps_prox=0.0,        # N3: proximal to reference eps
    eps_ref=None,               # reference eps for N3
    lambda_eps_inject=0.0,      # N4: extra noise injection magnitude per step
    sigma_ref_sq=0.01,          # soft-damping floor; smaller -> stronger Tweedie pull
):
    """One PRINCIPLE refinement step with mask-aware step size.

    h_x_obs_ratio=1.0 reproduces original behavior.
    h_x_obs_ratio<1 shrinks step inside observed pixels (they are already
      anchored by the measurement residual term anyway).
    h_x_obs_ratio>1 lets the prior push harder on unobserved pixels.
    """
    score_precision = 1.0 / (float(sigma_tau) ** 2 + float(sigma_ref_sq))

    mu = velocity_fn(x_tau_k)
    xs_hat = x_tau_k - tau * mu
    xe_hat = x_tau_k + (1.0 - tau) * mu

    if x1_init_mode == "model":
        x1_hat = direct_estimate_x1(xs_hat, xe_hat, s_k, e_k)
    else:
        x1_hat = wls_estimate_x1(xs_hat, xe_hat, s_k, e_k,
                                 rho_s, rho_e, lambda_x, cg_tol, cg_max_iter,
                                 stage_idx=stage_idx)

    H_x1_hat = apply_H_tau(x1_hat, tau, s_k, e_k, stage_idx=stage_idx)
    s_flow = score_precision * (H_x1_hat - x_tau_k)

    residual = y - A_k_fn(x1_k)
    # Step (d) in paper: g_x1 = (1/η²) A^T (b - A x1) + (H_τ)^T s_flow
    #                         + λ_prox (sg(x1_hat) - x1_k)     ← paper's 3rd term
    # Paper's λ = lambda_reg; we decouple via lambda_prox so =0 reproduces current code.
    g_x1 = (1.0 / eta ** 2) * AT_k_fn(residual) + \
        apply_HT_tau(s_flow, tau, s_k, e_k, stage_idx=stage_idx)
    if lambda_prox > 0:
        g_x1 = g_x1 + lambda_prox * (x1_hat.detach() - x1_k)

    # g_eps augmentation (Round-6 new terms)
    g_eps = sigma_tau * s_flow - eps_k

    if mask_aware_eps and mask_k is not None:
        # In unobs region, drop the Tweedie term so eps stays near N(0,I) prior
        H_, W_ = eps_k.shape[-2:]
        m = mask_k if mask_k.shape[-2:] == (H_, W_) else F.interpolate(mask_k, size=(H_, W_), mode="nearest")
        g_eps_unobs = -eps_k  # pure prior gradient, no Tweedie
        g_eps_obs = sigma_tau * s_flow - eps_k  # original
        g_eps = m * g_eps_obs + (1.0 - m) * g_eps_unobs

    if lambda_eps_norm > 0:
        # N1: push ||eps||² toward N (elem-count). Scales elementwise by (1 - ||eps||²/N)
        N_elem = float(eps_k.numel() / eps_k.shape[0])  # per-sample dim
        per_sample_norm2 = eps_k.pow(2).reshape(eps_k.shape[0], -1).sum(dim=1, keepdim=True)
        scale = (1.0 - per_sample_norm2 / N_elem).view(-1, 1, 1, 1)
        g_eps = g_eps + lambda_eps_norm * scale * eps_k

    if lambda_eps_prox > 0 and eps_ref is not None:
        # N3: proximal pull toward a reference eps (e.g., initial eps at chain start)
        g_eps = g_eps + lambda_eps_prox * (eps_ref - eps_k)

    if lambda_eps_inject > 0:
        # N4: additive Gaussian injection each step
        g_eps = g_eps + lambda_eps_inject * torch.randn_like(eps_k)

    xi_1 = torch.randn_like(y) * noise_scale
    xi_2 = torch.randn_like(x1_k) * noise_scale

    def system(x):
        return (1.0 / eta ** 2) * AT_k_fn(A_k_fn(x)) + lambda_reg * x

    rhs = (h_x / 2) * g_x1 + math.sqrt(h_x) * (
        (1.0 / eta) * AT_k_fn(xi_1) + math.sqrt(lambda_reg) * xi_2)
    delta = cg_solve(system, rhs, tol=cg_tol, max_iter=cg_max_iter)

    if mask_k is not None and abs(h_x_obs_ratio - 1.0) > 1e-6:
        # Scale update differently inside observed region.
        # Upsample mask if it does not match delta resolution.
        H_, W_ = delta.shape[-2:]
        if mask_k.shape[-2:] != (H_, W_):
            m = F.interpolate(mask_k, size=(H_, W_), mode="nearest")
        else:
            m = mask_k
        scale = h_x_obs_ratio * m + 1.0 * (1.0 - m)
        delta = delta * scale

    x1_k = x1_k + delta

    if not skip_eps_update:
        x3 = torch.randn_like(eps_k) * noise_scale
        eps_k = eps_k + (h_eps / 2) * g_eps + math.sqrt(h_eps) * x3

    x_tau_k = apply_H_tau(x1_k, tau, s_k, e_k, stage_idx=stage_idx) + sigma_tau * eps_k

    return x1_k, eps_k, x_tau_k


@torch.no_grad()
def principle_langevin_v5(
    x1_init, eps_init, tau, s_k, e_k,
    velocity_fn, A_k_fn, AT_k_fn,
    y, sigma_n, h_x, h_epsilon,
    lambda_x, lambda_reg, rho_s, rho_e,
    cg_tol, cg_max_iter, num_Langevin,
    device, stage_idx=None, x1_init_mode="model",
    noise_scale=0.0, mask_k=None, h_x_obs_ratio=1.0,
    lambda_prox=0.0, skip_eps_update=False,
    lambda_eps_norm=0.0, mask_aware_eps=False,
    lambda_eps_prox=0.0, eps_ref=None,
    lambda_eps_inject=0.0,
    sigma_ref_sq=0.01,
):
    """Identical contract to principle_langevin_sample but no traj recording
    and supports mask-aware step size."""
    sigma_tau = compute_sigma_tau(tau, s_k, e_k)
    x1_k = x1_init.clone().detach().to(device)
    eps_k = eps_init.clone().detach().to(device)
    x_tau_k = apply_H_tau(x1_k, tau, s_k, e_k, stage_idx=stage_idx) + sigma_tau * eps_k

    for _ in range(int(num_Langevin)):
        x1_k, eps_k, x_tau_k = langevin_step_v5(
            x1_k, eps_k, x_tau_k, tau, s_k, e_k, sigma_tau,
            velocity_fn, A_k_fn, AT_k_fn, y, sigma_n,
            h_x, h_epsilon, lambda_x, lambda_reg,
            rho_s, rho_e, cg_tol, cg_max_iter,
            stage_idx=stage_idx, x1_init_mode=x1_init_mode,
            noise_scale=noise_scale, mask_k=mask_k,
            h_x_obs_ratio=h_x_obs_ratio,
            lambda_prox=lambda_prox,
            skip_eps_update=skip_eps_update,
            lambda_eps_norm=lambda_eps_norm,
            mask_aware_eps=mask_aware_eps,
            lambda_eps_prox=lambda_eps_prox, eps_ref=eps_ref,
            lambda_eps_inject=lambda_eps_inject,
            sigma_ref_sq=sigma_ref_sq,
        )
    return x1_k.detach(), eps_k.detach()


@torch.no_grad()
def dps_gradient_kick(x1_k, A_k_fn, AT_k_fn, y, zeta):
    """DPS-style gradient correction: x1 <- x1 + zeta * A^T(y - A x1) / ||y - A x1||.
    Use BEFORE Langevin inner loop (hybrid, following IP3 dps+L3 winner)."""
    B = x1_k.shape[0]
    residual = y - A_k_fn(x1_k)
    res_norm = residual.reshape(B, -1).norm(dim=1).reshape(B, 1, 1, 1).clamp(min=1e-8)
    return x1_k + zeta * AT_k_fn(residual) / res_norm


@torch.no_grad()
def terminal_replacement(x1_k, y, mask_full, weight=1.0):
    """At stage 3 (full 256x256), replace observed pixels with measurement.
    weight=1.0 full replacement, 0<w<1 soft blend.
    mask_full must be 1=observed, 0=unobserved, at 256x256."""
    return weight * (mask_full * y + (1 - mask_full) * x1_k) + (1 - weight) * x1_k
