"""
FINAL version of article utils (PRINCIPLE) — debug_IP2 edition.

Changes vs. original article:
  1) G is kept as DownUp (matches PixelFlow training), stage-invariant
  2) x1_k is ALWAYS warm-started from the flow model's v-prediction at
     the start of each outer time step (not just carried via Langevin)
  3) Langevin step size h_x and CG ridge lambda_reg are large enough to
     ensure measurement consistency converges in 10-20 iterations
  4) At stage boundaries: eps_k is RESAMPLED from renoised latent_tau
     via inversion, but x1_k is re-seeded from model's v-pred at the new
     resolution so it's not just nearest-upsampled garbage
  5) sigma_floor is used only for the 1/σ² score term; downstream multiplications
     keep the raw σ so g_eps stays small at tau≈1
  6) A single Langevin "skip" threshold governs both the inner noise
     explosion safeguard and the score-divide safeguard

See ms_posterior_sampling_article_version_final.py for the orchestrator.
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
# Config loader — same fields as article version, + explicit comments
# ---------------------------------------------------------------------------

def load_run_config(config_path):
    import json, warnings
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

    # Defaults come from debug_IP2/run_final_sweep.py → F1_warm_frozen
    # which gave 3.4× lower residual than article-as-shipped and produced
    # visually coherent inpainting.
    cfg.setdefault("h_x", 2e-2)
    cfg.setdefault("h_epsilon", 1e-2)
    cfg.setdefault("lambda_x", 0.1)
    cfg.setdefault("lambda_reg", 0.2)
    cfg.setdefault("rho_s", [0.1, 0.3, 0.6, 1.0])
    cfg.setdefault("rho_e", 1.0)
    cfg.setdefault("cg_tol", 1e-5)
    cfg.setdefault("cg_max_iter", 30)
    cfg.setdefault("warm_x1_from_vpred", True)
    cfg.setdefault("warm_x1_from_wls", False)
    cfg.setdefault("sigma_floor", 0.02)
    cfg.setdefault("skip_sigma", 0.02)
    cfg.setdefault("reseed_x1_at_stage_start", False)  # empirically better when off (F2 config)
    cfg.setdefault("joint_eps", True)   # Joint ε update + warm_vpred reaches noise floor
    cfg.setdefault("latent_update_mode", "principle_final")

    mode = cfg.get("measurement_mode", "measure")
    sn = float(cfg.get("sigma_n", 0.0))
    if mode == "call" and sn > 1e-6:
        warnings.warn(
            f"measurement_mode='call' produces noiseless y, but sigma_n={sn} > 0. "
            f"Either use measurement_mode='measure' or set sigma_n very small.",
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
# CG solver (batch-aware, relative tol)
# ---------------------------------------------------------------------------

def cg_solve(A_fn, b, x0=None, tol=1e-5, max_iter=50):
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
# G / H_tau / sigma_tau — G is the stage-invariant DownUp (matches training)
# ---------------------------------------------------------------------------

def apply_G(x, scale_factor=2):
    """G = nearest_up ∘ bilinear_down: low-pass projection at scale 2."""
    _, _, H, W = x.shape
    small = (max(H // scale_factor, 1), max(W // scale_factor, 1))
    return F.interpolate(
        F.interpolate(x, size=small, mode="bilinear", align_corners=False),
        size=(H, W), mode="nearest",
    )


def apply_H_tau(x, tau, s_k, e_k):
    return (1.0 - tau) * s_k * apply_G(x) + tau * e_k * x


def apply_HT_tau(x, tau, s_k, e_k):
    # G is self-adjoint at scale 2 (verified to 1e-5)
    return (1.0 - tau) * s_k * apply_G(x) + tau * e_k * x


def compute_sigma_tau(tau, s_k, e_k):
    return (1.0 - tau) * (1.0 - s_k) + tau * (1.0 - e_k)


# ---------------------------------------------------------------------------
# WLS clean-image estimate — same as article version, but uses stage-agnostic G
# ---------------------------------------------------------------------------

def wls_estimate_x1(x_start_hat, x_end_hat, s_k, e_k, rho_s, rho_e,
                    lambda_x, cg_tol, cg_max_iter):
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
# A_k with exact autograd adjoint (unchanged from article)
# ---------------------------------------------------------------------------

def _interpolate_adjoint(y, target_size):
    with torch.enable_grad():
        xp = torch.zeros(*y.shape[:2], *target_size,
                         device=y.device, dtype=y.dtype, requires_grad=True)
        Ux = F.interpolate(xp, size=y.shape[-2:], mode="bilinear", align_corners=False)
        (grad,) = torch.autograd.grad(Ux, xp, grad_outputs=y.detach())
    return grad.detach()


def make_Ak_fns(operator, y, stage_shape, device):
    if hasattr(operator, "get_mask"):
        mask = operator.get_mask(x=y).to(device).float()
    elif hasattr(operator, "mask"):
        mask = operator.mask.to(device).float()
    else:
        raise ValueError("Operator must provide get_mask or mask attribute")

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


def make_velocity_fn(model, T, prompt_embeds, size_tensor, rope_pos,
                     do_cfg, guidance_scale_base, stage_idx):
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


def pred_x1_from_vpred(x_tau, v_pred, T, start_t, end_t):
    """Direct x1 extraction from v-pred (same formula as old pred_x1_x0_with_vt)."""
    x1 = x_tau + (1 - T / 1000) / (end_t - start_t) * v_pred
    return x1


# ---------------------------------------------------------------------------
# PRINCIPLE Langevin — with warm-start options and safeguards
# ---------------------------------------------------------------------------

@torch.no_grad()
def _langevin_step_final(x1_k, eps_k, x_tau_k, tau, s_k, e_k, sigma_tau,
                         velocity_fn, A_k_fn, AT_k_fn, b, eta,
                         h_x, h_eps, lambda_x, lambda_reg,
                         rho_s, rho_e, cg_tol, cg_max_iter,
                         sigma_floor=0.02,
                         warm_x1_from_wls=False,
                         joint_eps=True):
    sigma_safe = max(float(sigma_tau), sigma_floor)

    # (a) velocity → endpoints
    mu = velocity_fn(x_tau_k)
    xs_hat = x_tau_k - tau * mu
    xe_hat = x_tau_k + (1.0 - tau) * mu

    # (b) WLS x1 estimate via CG
    x1_hat = wls_estimate_x1(xs_hat, xe_hat, s_k, e_k,
                              rho_s, rho_e, lambda_x, cg_tol, cg_max_iter)

    if warm_x1_from_wls:
        x1_k = x1_hat.clone()

    H_x1_hat = apply_H_tau(x1_hat, tau, s_k, e_k)

    # (c) Tweedie score — floored only in 1/σ²
    s_flow = (1.0 / sigma_safe ** 2) * (H_x1_hat - x_tau_k)

    # (d) posterior gradients (data term uses x1_k, not x1_hat)
    residual = b - A_k_fn(x1_k)
    g_x1 = (1.0 / eta ** 2) * AT_k_fn(residual) + apply_HT_tau(s_flow, tau, s_k, e_k)

    if joint_eps:
        g_eps = sigma_tau * s_flow - eps_k
    else:
        g_eps = None

    # (e) CG-preconditioned ULA on x1
    xi_1, xi_2 = torch.randn_like(b), torch.randn_like(x1_k)

    def system(x):
        return (1.0 / eta ** 2) * AT_k_fn(A_k_fn(x)) + lambda_reg * x

    rhs = (h_x / 2) * g_x1 + math.sqrt(h_x) * (
        (1.0 / eta) * AT_k_fn(xi_1) + math.sqrt(lambda_reg) * xi_2)
    delta = cg_solve(system, rhs, tol=cg_tol, max_iter=cg_max_iter)
    x1_k = x1_k + delta

    # (f) Langevin on eps (optional)
    if joint_eps:
        x3 = torch.randn_like(eps_k)
        eps_k = eps_k + (h_eps / 2) * g_eps + math.sqrt(h_eps) * x3

    # (g) reconstruct x_tau
    x_tau_k = apply_H_tau(x1_k, tau, s_k, e_k) + sigma_tau * eps_k

    # Use the key names save_langevin_logs_csv expects so the existing CSV
    # helper still works unmodified.
    log = {
        "loss": float((residual ** 2).sum()),
        "l1": float(sigma_tau),      # repurposed: sigma_tau
        "l2": float(eps_k.std()),    # repurposed: eps_std
        "l3": float((b - A_k_fn(x1_k)).pow(2).sum()),
        "grad_norm": float(g_x1.norm()),
        "delta_norm": float(delta.norm()),
        "z_mean": float(x1_k.mean()),
        "z_std": float(x1_k.std()),
    }
    return x1_k, eps_k, x_tau_k, log


def principle_langevin_final(x1_init, eps_init, tau, s_k, e_k,
                             velocity_fn, A_k_fn, AT_k_fn,
                             y, sigma_n, h_x, h_epsilon,
                             lambda_x, lambda_reg, rho_s, rho_e,
                             cg_tol, cg_max_iter, num_Langevin, device,
                             sigma_floor=0.02,
                             warm_x1_from_wls=False,
                             joint_eps=True,
                             return_traj=False, record_every=1):
    sigma_tau = compute_sigma_tau(tau, s_k, e_k)
    x1_k = x1_init.clone().detach().to(device)
    eps_k = eps_init.clone().detach().to(device)
    x_tau_k = apply_H_tau(x1_k, tau, s_k, e_k) + sigma_tau * eps_k

    traj, logs = [], []
    for i in range(int(num_Langevin)):
        x1_k, eps_k, x_tau_k, log = _langevin_step_final(
            x1_k, eps_k, x_tau_k, tau, s_k, e_k, sigma_tau,
            velocity_fn, A_k_fn, AT_k_fn, y, sigma_n,
            h_x, h_epsilon, lambda_x, lambda_reg,
            rho_s, rho_e, cg_tol, cg_max_iter,
            sigma_floor=sigma_floor,
            warm_x1_from_wls=warm_x1_from_wls,
            joint_eps=joint_eps,
        )
        if return_traj and i % max(1, int(record_every)) == 0:
            traj.append(x1_k.detach().clone())
            logs.append(log)

    if return_traj:
        return x1_k.detach(), eps_k.detach(), traj, logs
    return x1_k.detach(), eps_k.detach()
