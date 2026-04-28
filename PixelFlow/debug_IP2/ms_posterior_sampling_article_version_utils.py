"""
Utilities for PRINCIPLE: PRogressive INterpolant with CG-Inner Preconditioned Langevin Estimation.

New functions for the article version algorithm. Common utilities are re-exported from
ms_posterior_sampling_utils for backward compatibility.
"""

import math
import torch
import torch.nn.functional as F

# ── Re-export only what the article-version main script actually uses ──
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
# Config loader  (standalone — no legacy field dependency)
# ---------------------------------------------------------------------------

def load_run_config(config_path):
    """
    Load and validate config for PRINCIPLE article version.
    Standalone validator — does NOT call the old load_run_config which
    requires legacy fields (proj, lr_base, lr_min_ratio) absent from
    the article version JSON.
    """
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
        raise KeyError(f"Missing required keys in article version config: {missing}")

    for op_key in ("box_operator", "random_operator"):
        op_cfg = cfg[op_key]
        if not isinstance(op_cfg, dict):
            raise TypeError(f"'{op_key}' must be a dict, got {type(op_cfg)}")
        if "mask_type" not in op_cfg:
            raise KeyError(f"'{op_key}.mask_type' is required")

    # PRINCIPLE defaults
    cfg.setdefault("h_x", 1e-3)
    cfg.setdefault("h_epsilon", 1e-3)
    cfg.setdefault("lambda_x", 1e-2)
    cfg.setdefault("lambda_reg", 1e-2)
    cfg.setdefault("rho_s", 1.0)
    cfg.setdefault("rho_e", 1.0)
    cfg.setdefault("cg_tol", 1e-5)
    cfg.setdefault("cg_max_iter", 50)

    # measurement / likelihood consistency check
    import warnings
    mode = cfg.get("measurement_mode", "measure")
    sn = float(cfg.get("sigma_n", 0.0))
    if mode == "call" and sn > 1e-6:
        warnings.warn(
            f"measurement_mode='call' produces noiseless y, but sigma_n={sn} > 0. "
            f"The likelihood term (1/sigma_n^2) will treat clean measurements as noisy. "
            f"Either use measurement_mode='measure' or set sigma_n very small.",
            UserWarning, stacklevel=2,
        )

    # per-stage list length checks
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
    """
    Solve A x = b via CG where A_fn applies an SPD operator.
    Batch-aware (per-sample alpha/beta). Relative stopping: ||r||/||b|| < tol.
    """
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
# G operator  (verified self-adjoint to float32 precision at all stage sizes)
# ---------------------------------------------------------------------------

def apply_G(x, scale_factor=2):
    """G = nearest_up ∘ bilinear_down: low-pass projection.
    Numerically verified: G ≈ G^T to ~1e-7 relative error at all scales."""
    _, _, H, W = x.shape
    small = (H // scale_factor, W // scale_factor)
    return F.interpolate(
        F.interpolate(x, size=small, mode="bilinear", align_corners=False),
        size=(H, W), mode="nearest",
    )


def apply_H_tau(x, tau, s_k, e_k):
    """H_tau^k(x) = (1-tau)*s_k*G(x) + tau*e_k*x"""
    return (1.0 - tau) * s_k * apply_G(x) + tau * e_k * x


def apply_HT_tau(x, tau, s_k, e_k):
    """(H_tau^k)^T(x) ≈ (1-tau)*s_k*G(x) + tau*e_k*x   (G self-adjoint)"""
    return (1.0 - tau) * s_k * apply_G(x) + tau * e_k * x


def compute_sigma_tau(tau, s_k, e_k):
    """sigma_tau = (1-tau)(1-s_k) + tau(1-e_k)"""
    return (1.0 - tau) * (1.0 - s_k) + tau * (1.0 - e_k)


# ---------------------------------------------------------------------------
# WLS clean-image estimate (Eq. 9)
# ---------------------------------------------------------------------------

def wls_estimate_x1(x_start_hat, x_end_hat, s_k, e_k, rho_s, rho_e,
                    lambda_x, cg_tol, cg_max_iter):
    """
    WLS estimate of x_1 via CG  (Algorithm 1, step b).
    M x = r  where
      M = rho_s*s_k^2/(1-s_k)^2 * G^T G + (rho_e*e_k^2/(1-e_k)^2 + lambda_x) * I
      r = rho_s*s_k/(1-s_k)^2 * G^T x_start + rho_e*e_k/(1-e_k)^2 * x_end
    """
    var_eps = 1e-2
    denom_s = max((1.0 - s_k) ** 2, var_eps)
    denom_e = max((1.0 - e_k) ** 2, var_eps)
    coeff_GTG = rho_s * s_k ** 2 / denom_s
    coeff_I = rho_e * e_k ** 2 / denom_e + lambda_x

    def M_fn(x):
        return coeff_GTG * apply_G(apply_G(x)) + coeff_I * x

    r = (rho_s * s_k / denom_s) * apply_G(x_start_hat) + \
        (rho_e * e_k / denom_e) * x_end_hat
    return cg_solve(M_fn, r, tol=cg_tol, max_iter=cg_max_iter)


# ---------------------------------------------------------------------------
# A_k = A ∘ U^{K-1-k}  with exact adjoint
# ---------------------------------------------------------------------------

def _interpolate_adjoint(y, target_size):
    """Exact adjoint of F.interpolate(·, size=y.shape[-2:], mode='bilinear', align_corners=False).
    Uses one autograd backward; verified to float precision at all scales."""
    with torch.enable_grad():
        xp = torch.zeros(*y.shape[:2], *target_size,
                          device=y.device, dtype=y.dtype, requires_grad=True)
        Ux = F.interpolate(xp, size=y.shape[-2:], mode="bilinear", align_corners=False)
        (grad,) = torch.autograd.grad(Ux, xp, grad_outputs=y.detach())
    return grad.detach()


def make_Ak_fns(operator, y, stage_shape, device):
    """Build A_k and A_k^T closures. A_k^T uses exact adjoint via autograd."""
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
# PRINCIPLE Langevin  (Algorithm 1, steps a–g)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _langevin_step(x1_k, eps_k, x_tau_k, tau, s_k, e_k, sigma_tau,
                   velocity_fn, A_k_fn, AT_k_fn, b, eta,
                   h_x, h_eps, lambda_x, lambda_reg,
                   rho_s, rho_e, cg_tol, cg_max_iter):
    """One PRINCIPLE Langevin refinement (steps a–g).
    Returns (x1_k, eps_k, x_tau_k, log)."""

    # sigma_tau_safe: floor for 1/sigma² division only.
    # Raw sigma_tau is used in g_eps (multiplication) and x_tau reconstruction.
    # Floor 0.01 bounds worst-case 1/sigma² to 10,000 (comparable to stage 2 levels).
    # Empirically verified: 1e-3 floor (1/σ²=1M) causes divergence at stage 3 last step.
    sigma_safe = max(float(sigma_tau), 0.01)

    # (a) velocity → endpoints
    mu = velocity_fn(x_tau_k)
    xs_hat = x_tau_k - tau * mu
    xe_hat = x_tau_k + (1.0 - tau) * mu

    # (b) WLS x1 estimate via CG
    x1_hat = wls_estimate_x1(xs_hat, xe_hat, s_k, e_k,
                              rho_s, rho_e, lambda_x, cg_tol, cg_max_iter)
    H_x1_hat = apply_H_tau(x1_hat, tau, s_k, e_k)

    # (c) Tweedie score
    s_flow = (1.0 / sigma_safe ** 2) * (H_x1_hat - x_tau_k)

    # (d) posterior gradients
    residual = b - A_k_fn(x1_k)
    g_x1 = (1.0 / eta ** 2) * AT_k_fn(residual) + apply_HT_tau(s_flow, tau, s_k, e_k)
    g_eps = sigma_tau * s_flow - eps_k          # uses raw sigma_tau (multiplication, no risk)

    # (e) CG-preconditioned ULA on x1
    #     Solve: (A_k^T A_k / η² + λI) Δx = rhs   →   Δx = P_k · rhs
    xi_1, xi_2 = torch.randn_like(b), torch.randn_like(x1_k)
    def system(x):
        return (1.0 / eta ** 2) * AT_k_fn(A_k_fn(x)) + lambda_reg * x
    rhs = (h_x / 2) * g_x1 + math.sqrt(h_x) * (
        (1.0 / eta) * AT_k_fn(xi_1) + math.sqrt(lambda_reg) * xi_2)
    delta = cg_solve(system, rhs, tol=cg_tol, max_iter=cg_max_iter)
    x1_k = x1_k + delta

    # (f) Langevin on eps
    x3 = torch.randn_like(eps_k)
    eps_k = eps_k + (h_eps / 2) * g_eps + math.sqrt(h_eps) * x3

    # (g) reconstruct x_tau (uses raw sigma_tau)
    x_tau_k = apply_H_tau(x1_k, tau, s_k, e_k) + sigma_tau * eps_k

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
                              return_traj=False, record_every=1):
    """PRINCIPLE Langevin refinement (Algorithm 1, lines 12–40).
    Returns (x1, eps, traj, logs) if return_traj else (x1, eps)."""

    sigma_tau = compute_sigma_tau(tau, s_k, e_k)
    x1_k = x1_init.clone().detach().to(device)
    eps_k = eps_init.clone().detach().to(device)
    x_tau_k = apply_H_tau(x1_k, tau, s_k, e_k) + sigma_tau * eps_k

    traj, logs = [], []
    record_every = max(1, int(record_every))

    for i in range(int(num_Langevin)):
        x1_k, eps_k, x_tau_k, log = _langevin_step(
            x1_k, eps_k, x_tau_k, tau, s_k, e_k, sigma_tau,
            velocity_fn, A_k_fn, AT_k_fn, y, sigma_n,
            h_x, h_epsilon, lambda_x, lambda_reg,
            rho_s, rho_e, cg_tol, cg_max_iter)
        if return_traj and i % record_every == 0:
            traj.append(x1_k.detach().clone())
            logs.append(log)

    if return_traj:
        return x1_k.detach(), eps_k.detach(), traj, logs
    return x1_k.detach(), eps_k.detach()
