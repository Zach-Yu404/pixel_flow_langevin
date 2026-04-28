"""
Utilities for PRINCIPLE final version — fixes identified via debug_IP diagnostics.

Key fixes vs article_version_utils:
1. apply_G: bypass at stage 3 (identity) — matches old DownUp_operation behavior
2. Warm restart: re-initialize x1 from model prediction at each ODE step
3. Tuned default parameters: h_x=0.1, lambda_reg=1.0 for stability
4. Direct x1 estimation option alongside WLS
"""

import math
import torch
import torch.nn.functional as F

from ms_posterior_sampling_utils import (
    resolve_path,
    get_stage_inference_steps,
    build_experiment_paths,
    center_crop_arr,
    sample_block_noise,
    class_guidance_scale,
    save_posterior_sampling_videos,
    save_langevin_logs_csv,
)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_run_config(config_path):
    """Load and validate config for PRINCIPLE final version."""
    import json
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    required_keys = [
        "exp_name", "seed", "num_stages", "resolution", "num_examples",
        "sigma_n", "class_label", "shift", "inference_each_step",
        "guidance_scale", "num_langevin",
        "device", "data_dir", "model_dir", "dict_path",
        "box_operator", "random_operator",
        "active_operator", "measurement_mode",
    ]
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        raise KeyError(f"Missing required keys: {missing}")

    for op_key in ("box_operator", "random_operator"):
        op_cfg = cfg[op_key]
        if not isinstance(op_cfg, dict):
            raise TypeError(f"'{op_key}' must be a dict")
        if "mask_type" not in op_cfg:
            raise KeyError(f"'{op_key}.mask_type' is required")

    # PRINCIPLE defaults — tuned based on debug_IP diagnostics
    # lambda_reg=50 prevents noise accumulation on unobserved pixels (see TEST_SUMMARY.md)
    cfg.setdefault("h_x", 0.1)
    cfg.setdefault("h_epsilon", 0.01)
    cfg.setdefault("lambda_x", 1e-2)
    cfg.setdefault("lambda_reg", 50.0)
    cfg.setdefault("rho_s", 1.0)
    cfg.setdefault("rho_e", 1.0)
    cfg.setdefault("cg_tol", 1e-5)
    cfg.setdefault("cg_max_iter", 50)
    cfg.setdefault("warm_restart", True)
    cfg.setdefault("g_bypass_stage3", True)
    cfg.setdefault("x1_init_mode", "model")  # "model" or "wls"
    cfg.setdefault("noise_scale", 0.0)  # 0.0=noise-free GD (best), 1.0=ULA
    cfg.setdefault("lambda_prox", 0.0)  # paper step-(d) 3rd term; 0=off, None=use lambda_reg

    # measurement / likelihood consistency check
    import warnings
    mode = cfg.get("measurement_mode", "measure")
    sn = float(cfg.get("sigma_n", 0.0))
    if mode == "call" and sn > 1e-6:
        warnings.warn(
            f"measurement_mode='call' produces noiseless y, but sigma_n={sn} > 0. "
            f"The likelihood term (1/sigma_n^2) will treat clean measurements as noisy.",
            UserWarning, stacklevel=2,
        )

    num_stages = int(cfg["num_stages"])
    for key in ("rho_s", "rho_e", "inference_each_step"):
        val = cfg[key]
        if isinstance(val, list):
            if len(val) != num_stages:
                raise ValueError(f"'{key}' list length {len(val)} != num_stages {num_stages}")

    return cfg


# ---------------------------------------------------------------------------
# Conjugate Gradient solver
# ---------------------------------------------------------------------------

def cg_solve(A_fn, b, x0=None, tol=1e-5, max_iter=50):
    """Solve A x = b via CG where A_fn applies an SPD operator."""
    B = b.shape[0]
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - A_fn(x)
    p = r.clone()
    rs_old = (r * r).reshape(B, -1).sum(dim=1)
    b_norm = (b * b).reshape(B, -1).sum(dim=1).sqrt().clamp(min=1e-12)

    for _ in range(max_iter):
        Ap = A_fn(p)
        pAp = (p * Ap).reshape(B, -1).sum(dim=1).clamp(min=1e-12)
        alpha = (rs_old / pAp).reshape(B, 1, 1, 1)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = (r * r).reshape(B, -1).sum(dim=1)
        if (rs_new.sqrt() / b_norm).max() < tol:
            break
        beta = (rs_new / rs_old.clamp(min=1e-12)).reshape(B, 1, 1, 1)
        p = r + beta * p
        rs_old = rs_new

    return x


# ---------------------------------------------------------------------------
# G operator — with stage 3 identity bypass (FIX #1)
# ---------------------------------------------------------------------------

def apply_G(x, scale_factor=2, stage_idx=None):
    """G = nearest_up . bilinear_down: low-pass projection.
    At stage 3 (256x256 → 256x256), returns identity to preserve high-freq details.
    This matches old DownUp_operation behavior."""
    if stage_idx == 3:
        return x
    _, _, H, W = x.shape
    small = (H // scale_factor, W // scale_factor)
    return F.interpolate(
        F.interpolate(x, size=small, mode="bilinear", align_corners=False),
        size=(H, W), mode="nearest",
    )


def apply_H_tau(x, tau, s_k, e_k, stage_idx=None):
    """H_tau^k(x) = (1-tau)*s_k*G(x) + tau*e_k*x"""
    return (1.0 - tau) * s_k * apply_G(x, stage_idx=stage_idx) + tau * e_k * x


def apply_HT_tau(x, tau, s_k, e_k, stage_idx=None):
    """(H_tau^k)^T(x) = (1-tau)*s_k*G^T(x) + tau*e_k*x  (G self-adjoint)"""
    return (1.0 - tau) * s_k * apply_G(x, stage_idx=stage_idx) + tau * e_k * x


def compute_sigma_tau(tau, s_k, e_k):
    """sigma_tau = (1-tau)(1-s_k) + tau(1-e_k)"""
    return (1.0 - tau) * (1.0 - s_k) + tau * (1.0 - e_k)


# ---------------------------------------------------------------------------
# WLS clean-image estimate (Eq. 9) — with stage_idx awareness
# ---------------------------------------------------------------------------

def wls_estimate_x1(x_start_hat, x_end_hat, s_k, e_k, rho_s, rho_e,
                    lambda_x, cg_tol, cg_max_iter, stage_idx=None):
    """WLS estimate of x_1 via CG (Algorithm 1, step b)."""
    var_eps = 1e-2
    denom_s = max((1.0 - s_k) ** 2, var_eps)
    denom_e = max((1.0 - e_k) ** 2, var_eps)
    coeff_GTG = rho_s * s_k ** 2 / denom_s
    coeff_I = rho_e * e_k ** 2 / denom_e + lambda_x

    def M_fn(x):
        return coeff_GTG * apply_G(apply_G(x, stage_idx=stage_idx), stage_idx=stage_idx) + coeff_I * x

    r = (rho_s * s_k / denom_s) * apply_G(x_start_hat, stage_idx=stage_idx) + \
        (rho_e * e_k / denom_e) * x_end_hat
    return cg_solve(M_fn, r, tol=cg_tol, max_iter=cg_max_iter)


def direct_estimate_x1(x_start_hat, x_end_hat, s_k, e_k):
    """Direct x1 estimate: x1 = ((1-s)*xe - (1-e)*xs) / (e-s).
    Simpler than WLS, sometimes more accurate (see debug_IP TEST 8)."""
    denom = max(e_k - s_k, 1e-8)
    return ((1.0 - s_k) * x_end_hat - (1.0 - e_k) * x_start_hat) / denom


# ---------------------------------------------------------------------------
# A_k = A . U^{K-1-k}  with exact adjoint
# ---------------------------------------------------------------------------

def _interpolate_adjoint(y, target_size):
    """Exact adjoint of F.interpolate via autograd."""
    with torch.enable_grad():
        xp = torch.zeros(*y.shape[:2], *target_size,
                          device=y.device, dtype=y.dtype, requires_grad=True)
        Ux = F.interpolate(xp, size=y.shape[-2:], mode="bilinear", align_corners=False)
        (grad,) = torch.autograd.grad(Ux, xp, grad_outputs=y.detach())
    return grad.detach()


def make_Ak_fns(operator, y, stage_shape, device):
    """Build A_k and A_k^T closures."""
    if hasattr(operator, "get_mask"):
        mask = operator.get_mask(x=y).to(device).float()
    elif hasattr(operator, "mask"):
        mask = operator.mask.to(device).float()
    else:
        raise ValueError("Operator must provide get_mask() or mask attribute")

    full_h, full_w = mask.shape[-2:]
    stage_h, stage_w = stage_shape[-2:]
    need_resize = (stage_h != full_h) or (stage_w != full_w)

    def A_k(x):
        x_up = F.interpolate(x, size=(full_h, full_w), mode="bilinear", align_corners=False) if need_resize else x
        return mask * x_up

    def AT_k(r):
        masked = mask * r
        return _interpolate_adjoint(masked, (stage_h, stage_w)) if need_resize else masked

    return A_k, AT_k


# ---------------------------------------------------------------------------
# Velocity function factory
# ---------------------------------------------------------------------------

def make_velocity_fn(model, T, prompt_embeds, size_tensor, rope_pos,
                     do_cfg, guidance_scale_base, stage_idx):
    """Wrap model + CFG into velocity_fn(x_tau) closure."""
    def velocity_fn(x_tau):
        inp = torch.cat([x_tau] * 2) if do_cfg else x_tau
        ts = T.expand(inp.shape[0]).to(inp.dtype)
        with torch.no_grad():
            pred = model(inp, timestep=ts, class_labels=prompt_embeds,
                         latent_size=size_tensor, pos_embed=rope_pos)
        if do_cfg:
            scale = class_guidance_scale(guidance_scale_base, stage_idx)
            u, c = pred.chunk(2)
            pred = u + scale * (c - u)
        return pred
    return velocity_fn


# ---------------------------------------------------------------------------
# PRINCIPLE Langevin step (Algorithm 1, steps a–g) — with fixes
# ---------------------------------------------------------------------------

@torch.no_grad()
def _langevin_step(x1_k, eps_k, x_tau_k, tau, s_k, e_k, sigma_tau,
                   velocity_fn, A_k_fn, AT_k_fn, b, eta,
                   h_x, h_eps, lambda_x, lambda_reg,
                   rho_s, rho_e, cg_tol, cg_max_iter,
                   stage_idx=None, x1_init_mode="model",
                   noise_scale=0.0, lambda_prox=0.0,
                   sigma_ref_sq=0.01):
    """One PRINCIPLE refinement step.
    noise_scale=0.0 (default): pure CG-preconditioned gradient descent (best quality).
    noise_scale=1.0: standard ULA with stochastic noise (original PRINCIPLE formulation).
    lambda_prox: weight of the paper's step-(d) 3rd term λ(sg(x1_hat) - x1_k).
        0.0  (default) => disable; reproduces pre-IP4 numerics.
        None           => use lambda_reg (matches paper's single-λ convention).
        float          => explicit weight, decoupled from lambda_reg.
    sigma_ref_sq: Tweedie soft-damping floor in score_precision = 1/(σ_τ² + σ_ref²).
        Smaller -> stronger Tweedie pull at small sigma_tau (i.e. late timesteps).
    """

    # Soft damping: 1/(sigma_tau² + sigma_ref²) instead of 1/max(sigma_tau, floor)²
    # Caps prior weight at ~1/sigma_ref². Default 0.01 (sigma_ref = 0.1) preserves
    # legacy behavior; smaller values strengthen Tweedie at small sigma_tau.
    score_precision = 1.0 / (float(sigma_tau) ** 2 + float(sigma_ref_sq))

    # (a) velocity → endpoints
    mu = velocity_fn(x_tau_k)
    xs_hat = x_tau_k - tau * mu
    xe_hat = x_tau_k + (1.0 - tau) * mu

    # (b) x1 estimate — model-direct or WLS
    if x1_init_mode == "model":
        x1_hat = direct_estimate_x1(xs_hat, xe_hat, s_k, e_k)
    else:
        x1_hat = wls_estimate_x1(xs_hat, xe_hat, s_k, e_k,
                                  rho_s, rho_e, lambda_x, cg_tol, cg_max_iter,
                                  stage_idx=stage_idx)

    H_x1_hat = apply_H_tau(x1_hat, tau, s_k, e_k, stage_idx=stage_idx)

    # (c) Tweedie score with soft damping
    s_flow = score_precision * (H_x1_hat - x_tau_k)

    # (d) posterior gradients (paper Algorithm 1, step (d))
    # g_x1 = (1/η²) A^T (b - A x1) + (H_τ)^T s_flow + λ (sg(x1_hat) - x1_k)
    # The third term is a proximal pull toward the WLS/model estimate x1_hat;
    # paper uses λ = lambda_reg, we let callers override via lambda_prox.
    residual = b - A_k_fn(x1_k)
    g_x1 = (1.0 / eta ** 2) * AT_k_fn(residual) + apply_HT_tau(s_flow, tau, s_k, e_k, stage_idx=stage_idx)
    lp = float(lambda_reg) if lambda_prox is None else float(lambda_prox)
    if lp > 0:
        g_x1 = g_x1 + lp * (x1_hat.detach() - x1_k)
    g_eps = sigma_tau * s_flow - eps_k

    # (e) CG-preconditioned update on x1
    # noise_scale=0: pure gradient descent (no xi noise), best image quality
    # noise_scale=1: standard ULA with stochastic noise
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

    # (g) reconstruct x_tau
    x_tau_k = apply_H_tau(x1_k, tau, s_k, e_k, stage_idx=stage_idx) + sigma_tau * eps_k

    log = {
        "loss": float((residual ** 2).sum()),
        "l3": float((b - A_k_fn(x1_k)).pow(2).sum()),
        "grad_norm": float(g_x1.norm()),
        "delta_norm": float(delta.norm()),
        "z_mean": float(x1_k.mean()),
        "z_std": float(x1_k.std()),
    }
    return x1_k, eps_k, x_tau_k, log


def principle_langevin_sample(x1_init, eps_init, tau, s_k, e_k,
                              velocity_fn, A_k_fn, AT_k_fn,
                              y, sigma_n, h_x, h_epsilon,
                              lambda_x, lambda_reg, rho_s, rho_e,
                              cg_tol, cg_max_iter, num_Langevin, device,
                              stage_idx=None, x1_init_mode="model",
                              noise_scale=0.0, lambda_prox=0.0,
                              sigma_ref_sq=0.01,
                              return_traj=False, record_every=1):
    """PRINCIPLE refinement (CG-preconditioned gradient descent by default).
    noise_scale=0.0: deterministic GD (best quality). 1.0: stochastic ULA.
    Returns (x1, eps, traj, logs) if return_traj else (x1, eps)."""

    sigma_tau = compute_sigma_tau(tau, s_k, e_k)
    x1_k = x1_init.clone().detach().to(device)
    eps_k = eps_init.clone().detach().to(device)
    x_tau_k = apply_H_tau(x1_k, tau, s_k, e_k, stage_idx=stage_idx) + sigma_tau * eps_k

    traj, logs = [], []
    record_every = max(1, int(record_every))

    for i in range(int(num_Langevin)):
        x1_k, eps_k, x_tau_k, log = _langevin_step(
            x1_k, eps_k, x_tau_k, tau, s_k, e_k, sigma_tau,
            velocity_fn, A_k_fn, AT_k_fn, y, sigma_n,
            h_x, h_epsilon, lambda_x, lambda_reg,
            rho_s, rho_e, cg_tol, cg_max_iter,
            stage_idx=stage_idx, x1_init_mode=x1_init_mode,
            noise_scale=noise_scale, lambda_prox=lambda_prox,
            sigma_ref_sq=sigma_ref_sq)
        if return_traj and i % record_every == 0:
            traj.append(x1_k.detach().clone())
            logs.append(log)

    if return_traj:
        return x1_k.detach(), eps_k.detach(), traj, logs
    return x1_k.detach(), eps_k.detach()
