#!/usr/bin/env python
"""
Explore improvements to reach DAPS-comparable posterior sampling quality.
Tests: projection, noise-free GD, range-null decomposition, adaptive steps.
"""
import sys, os, math, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from omegaconf import OmegaConf
from diffusers.models.embeddings import get_2d_rotary_pos_embed
from diffusers.utils.torch_utils import randn_tensor

from inpaintingStart import get_operator
from ms_posterior_sampling_article_version_final_utils import (
    apply_G, apply_H_tau, compute_sigma_tau, direct_estimate_x1,
    make_Ak_fns, make_velocity_fn, cg_solve, wls_estimate_x1,
    sample_block_noise,
)
from pixelflow.scheduling_pixelflow import PixelFlowScheduler
from pixelflow.utils import config as config_utils
from pixelflow.utils.misc import seed_everything

DEVICE = "cuda:0"
RESULT_DIR = os.path.join(os.path.dirname(__file__), "results")


# ═══════════════════════════════════════════════════════════
# Improved Langevin variants
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def langevin_step_improved(x1_k, eps_k, x_tau_k, tau, s_k, e_k, sigma_tau,
                           velocity_fn, A_k_fn, AT_k_fn, y, eta,
                           h_x, h_eps, lambda_x, lambda_reg,
                           rho_s, rho_e, cg_tol, cg_max_iter,
                           stage_idx=None, x1_init_mode="model",
                           noise_scale=1.0, project_fn=None):
    """
    Improved Langevin step with:
    - noise_scale: 0.0 = pure gradient descent, 1.0 = standard ULA
    - project_fn: if provided, applied after x1 update for data consistency
    """
    sigma_safe = max(float(sigma_tau), 0.01)

    # (a) velocity → endpoints
    mu = velocity_fn(x_tau_k)
    xs_hat = x_tau_k - tau * mu
    xe_hat = x_tau_k + (1.0 - tau) * mu

    # (b) x1 estimate
    if x1_init_mode == "model":
        x1_hat = direct_estimate_x1(xs_hat, xe_hat, s_k, e_k)
    else:
        x1_hat = wls_estimate_x1(xs_hat, xe_hat, s_k, e_k,
                                  rho_s, rho_e, lambda_x, cg_tol, cg_max_iter,
                                  stage_idx=stage_idx)
    H_x1_hat = apply_H_tau(x1_hat, tau, s_k, e_k, stage_idx=stage_idx)

    # (c) Tweedie score
    s_flow = (1.0 / sigma_safe ** 2) * (H_x1_hat - x_tau_k)

    # (d) posterior gradients
    residual = y - A_k_fn(x1_k)
    g_x1 = (1.0 / eta ** 2) * AT_k_fn(residual) + apply_H_tau(
        s_flow, tau, s_k, e_k, stage_idx=stage_idx)  # HT = H for self-adjoint G
    g_eps = sigma_tau * s_flow - eps_k

    # (e) CG-preconditioned update on x1
    xi_1 = torch.randn_like(y) * noise_scale
    xi_2 = torch.randn_like(x1_k) * noise_scale

    def system(x):
        return (1.0 / eta ** 2) * AT_k_fn(A_k_fn(x)) + lambda_reg * x

    rhs = (h_x / 2) * g_x1 + math.sqrt(h_x) * (
        (1.0 / eta) * AT_k_fn(xi_1) + math.sqrt(lambda_reg) * xi_2)
    delta = cg_solve(system, rhs, tol=cg_tol, max_iter=cg_max_iter)
    x1_k = x1_k + delta

    # (e.1) Optional projection for data consistency
    if project_fn is not None:
        x1_k = project_fn(x1_k)

    # (f) Langevin on eps (with noise scaling)
    x3 = torch.randn_like(eps_k) * noise_scale
    eps_k = eps_k + (h_eps / 2) * g_eps + math.sqrt(h_eps) * x3

    # (g) reconstruct x_tau
    x_tau_k = apply_H_tau(x1_k, tau, s_k, e_k, stage_idx=stage_idx) + sigma_tau * eps_k

    return x1_k, eps_k, x_tau_k, {
        "loss": float((residual ** 2).sum()),
        "l3": float((y - A_k_fn(x1_k)).pow(2).sum()),
        "delta_norm": float(delta.norm()),
    }


def improved_langevin_sample(x1_init, eps_init, tau, s_k, e_k,
                              velocity_fn, A_k_fn, AT_k_fn,
                              y, sigma_n, h_x, h_epsilon,
                              lambda_x, lambda_reg, rho_s, rho_e,
                              cg_tol, cg_max_iter, num_Langevin, device,
                              stage_idx=None, x1_init_mode="model",
                              noise_scale=1.0, project_fn=None):
    sigma_tau = compute_sigma_tau(tau, s_k, e_k)
    x1_k = x1_init.clone().detach().to(device)
    eps_k = eps_init.clone().detach().to(device)
    x_tau_k = apply_H_tau(x1_k, tau, s_k, e_k, stage_idx=stage_idx) + sigma_tau * eps_k

    logs = []
    for i in range(int(num_Langevin)):
        x1_k, eps_k, x_tau_k, log = langevin_step_improved(
            x1_k, eps_k, x_tau_k, tau, s_k, e_k, sigma_tau,
            velocity_fn, A_k_fn, AT_k_fn, y, sigma_n,
            h_x, h_epsilon, lambda_x, lambda_reg,
            rho_s, rho_e, cg_tol, cg_max_iter,
            stage_idx=stage_idx, x1_init_mode=x1_init_mode,
            noise_scale=noise_scale, project_fn=project_fn)
        logs.append(log)

    return x1_k.detach(), eps_k.detach(), logs


# ═══════════════════════════════════════════════════════════
# Pipeline runner
# ═══════════════════════════════════════════════════════════

def run_pipeline(model, config, gt, y, operator, sigma_n, label="", **kw):
    B = gt.shape[0]
    seed_everything(20000120)
    scheduler = PixelFlowScheduler(config.scheduler.num_train_timesteps,
                                   num_stages=config.scheduler.num_stages, gamma=-1/3)
    prompt_embeds = torch.tensor([10]*B, dtype=torch.int32, device=DEVICE)

    h_x = kw.get("h_x", 0.1)
    h_eps = kw.get("h_eps", 0.01)
    lambda_reg = kw.get("lambda_reg", 50.0)
    lambda_x = kw.get("lambda_x", 0.01)
    num_langevin = kw.get("num_langevin", 10)
    noise_scale = kw.get("noise_scale", 1.0)
    use_projection = kw.get("use_projection", False)
    x1_init_mode = kw.get("x1_init_mode", "model")
    adaptive_hx = kw.get("adaptive_hx", False)

    # Build projection function for inpainting
    mask = operator.get_mask(x=gt).float().to(DEVICE)
    def project_fn(x1):
        """Replace observed pixels in x1 with measurement values.
        For inpainting: x1_proj[observed] = y_at_stage[observed]"""
        _, _, hh, ww = x1.shape
        if hh == 256 and ww == 256:
            # Full resolution: direct replacement
            return mask * y + (1 - mask) * x1
        else:
            # Lower resolution: downsample mask and y
            mask_ds = F.interpolate(mask, size=(hh, ww), mode="nearest")
            y_ds = F.interpolate(y, size=(hh, ww), mode="bilinear", align_corners=False)
            return mask_ds * y_ds + (1 - mask_ds) * x1

    h, w = 32, 32
    x1_k = torch.randn(B, 3, h, w, device=DEVICE)
    eps_k = randn_tensor((B, 3, h, w), device=DEVICE)
    lat = eps_k.clone()
    residuals = []

    for si in range(4):
        sched = copy.deepcopy(scheduler)
        sched.set_timesteps(10, si, device=DEVICE, shift=1.0)
        sk = float(sched.start_t[si]); ek = float(sched.end_t[si])

        if si > 0:
            h *= 2; w *= 2
            lat = F.interpolate(lat, size=(h, w), mode="nearest")
            ost = sched.original_start_t[si]; gam = sched.gamma
            al = 1/(math.sqrt(1-(1/gam))*(1-ost)+ost)
            be = al*(1-ost)/math.sqrt(-gam)
            noise = sample_block_noise(sched, B, 3, h, w).to(device=DEVICE, dtype=lat.dtype)
            lat = al*lat + be*noise
            x1_k = F.interpolate(x1_k, size=(h, w), mode="nearest")
            eps_k = (lat - sk*apply_G(x1_k, stage_idx=si))/max(1-sk, 1e-8)

        Ak, ATk = make_Ak_fns(operator, y, (B, 3, h, w), DEVICE)
        st = torch.tensor([h//model.patch_size], dtype=torch.int32, device=DEVICE)
        pe = get_2d_rotary_pos_embed(embed_dim=model.attention_head_dim,
            crops_coords=((0, 0), (h//model.patch_size, w//model.patch_size)),
            grid_size=(h//model.patch_size, w//model.patch_size),
            device=DEVICE, output_type="pt")
        rope = torch.stack(pe, -1)

        for step_idx, T in enumerate(sched.Timesteps):
            tau = float(sched.t[step_idx].to(DEVICE))
            sig = compute_sigma_tau(tau, sk, ek)
            xtau = apply_H_tau(x1_k, tau, sk, ek, stage_idx=si) + sig*eps_k
            vfn = make_velocity_fn(model, T, prompt_embeds, st, rope, False, 0.0, si)

            # Warm restart
            with torch.no_grad(): mu = vfn(xtau)
            x1_k = direct_estimate_x1(xtau - tau*mu, xtau + (1-tau)*mu, sk, ek).detach()
            if sig > 1e-8:
                eps_k = (xtau - apply_H_tau(x1_k, tau, sk, ek, stage_idx=si))/sig

            # Adaptive h_x: scale by sigma_tau (larger correction when noise is high)
            cur_hx = h_x
            if adaptive_hx:
                # Scale h_x proportional to sigma_tau^2 (when noise high, correct more)
                cur_hx = h_x * max(sig, 0.01) ** 2 / 0.1  # normalized so sig=0.316 → h_x

            if sig >= 0.01:
                x1_k, eps_k, logs = improved_langevin_sample(
                    x1_init=x1_k, eps_init=eps_k, tau=tau, s_k=sk, e_k=ek,
                    velocity_fn=vfn, A_k_fn=Ak, AT_k_fn=ATk,
                    y=y, sigma_n=sigma_n, h_x=cur_hx, h_epsilon=h_eps,
                    lambda_x=lambda_x, lambda_reg=lambda_reg, rho_s=1.0, rho_e=1.0,
                    cg_tol=1e-5, cg_max_iter=50, num_Langevin=num_langevin,
                    device=DEVICE, stage_idx=si, x1_init_mode=x1_init_mode,
                    noise_scale=noise_scale,
                    project_fn=project_fn if use_projection else None)

            lat = apply_H_tau(x1_k, tau, sk, ek, stage_idx=si) + sig*eps_k

            if h == 256:
                res = (y - Ak(x1_k)).pow(2).sum().item()
            else:
                x1u = F.interpolate(x1_k, size=(256, 256), mode="bilinear", align_corners=False)
                Af, _ = make_Ak_fns(operator, y, (B, 3, 256, 256), DEVICE)
                res = (y - Af(x1u)).pow(2).sum().item()
            residuals.append(res)

    x1_f = x1_k.detach()
    psnr_obs = -10*torch.log10(((mask*x1_f - mask*gt)**2).mean()).item()
    psnr_all = -10*torch.log10(((x1_f - gt)**2).mean()).item()
    return x1_f.cpu(), residuals, psnr_obs, psnr_all


def main():
    print("Loading...", flush=True)
    d = torch.load("article_v_results/test_box_inpainting/test_box_inpainting_mode-f_x1.pt",
                    map_location="cpu", weights_only=False)
    gt = d["gt"][:2].to(DEVICE)
    B = gt.shape[0]

    model_dir = "pretrained_models/c2img"
    config = OmegaConf.load(os.path.join(model_dir, "config.yaml"))
    model = config_utils.instantiate_from_config(config.model).to(DEVICE)
    ckpt = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt, strict=True); model.eval()
    print("Model loaded.", flush=True)

    results = {}

    for mask_type, op_kw in [
        ("box", dict(mask_type="box", mask_len_range=(80,160), mask_prob_range=None)),
        ("random", dict(mask_type="random", mask_len_range=None, mask_prob_range=(0.8,0.8))),
    ]:
        operator = get_operator("inpainting", resolution=256, device=DEVICE, sigma=0.05, **op_kw)
        y = operator(gt).detach()
        sigma_n = 0.05

        print(f"\n{'='*60}", flush=True)
        print(f"  {mask_type.upper()} INPAINTING", flush=True)
        print(f"{'='*60}", flush=True)

        experiments = [
            # Baseline: current best
            ("A_baseline", dict(h_x=0.1, lambda_reg=50, num_langevin=10,
                                noise_scale=1.0, use_projection=False, x1_init_mode="model")),

            # B: Projection after each Langevin step
            ("B_proj", dict(h_x=0.1, lambda_reg=50, num_langevin=10,
                            noise_scale=1.0, use_projection=True, x1_init_mode="model")),

            # C: No noise (pure CG-preconditioned gradient descent)
            ("C_noisefree", dict(h_x=0.1, lambda_reg=50, num_langevin=10,
                                 noise_scale=0.0, use_projection=False, x1_init_mode="model")),

            # D: No noise + projection
            ("D_nf_proj", dict(h_x=0.1, lambda_reg=50, num_langevin=10,
                               noise_scale=0.0, use_projection=True, x1_init_mode="model")),

            # E: No noise + projection + more steps
            ("E_nf_proj_nl20", dict(h_x=0.1, lambda_reg=50, num_langevin=20,
                                    noise_scale=0.0, use_projection=True, x1_init_mode="model")),

            # F: No noise + proj + smaller lambda_reg (stronger correction)
            ("F_nf_proj_lr5", dict(h_x=0.1, lambda_reg=5.0, num_langevin=10,
                                   noise_scale=0.0, use_projection=True, x1_init_mode="model")),

            # G: No noise + proj + WLS
            ("G_nf_proj_wls", dict(h_x=0.1, lambda_reg=50, num_langevin=10,
                                   noise_scale=0.0, use_projection=True, x1_init_mode="wls",
                                   lambda_x=0.001)),

            # H: Reduced noise (0.3) + projection
            ("H_lownoise_proj", dict(h_x=0.1, lambda_reg=50, num_langevin=10,
                                     noise_scale=0.3, use_projection=True, x1_init_mode="model")),

            # I: Adaptive h_x + no noise + projection
            ("I_adaptive_nf_proj", dict(h_x=0.5, lambda_reg=50, num_langevin=10,
                                        noise_scale=0.0, use_projection=True,
                                        x1_init_mode="model", adaptive_hx=True)),

            # J: Aggressive: noisefree, proj, small lambda_reg, more steps, larger h_x
            ("J_aggressive", dict(h_x=0.5, lambda_reg=5.0, num_langevin=20,
                                  noise_scale=0.0, use_projection=True, x1_init_mode="model")),
        ]

        for name, kw in experiments:
            print(f"  {name}: {kw}", flush=True)
            try:
                x1, res, po, pa = run_pipeline(model, config, gt, y, operator, sigma_n, **kw)
                results[f"{mask_type}_{name}"] = (x1, res, po, pa)
                print(f"    → residual={res[-1]:.0f}  PSNR_obs={po:.2f}  PSNR_all={pa:.2f}", flush=True)
            except Exception as e:
                print(f"    → FAILED: {e}", flush=True)
                results[f"{mask_type}_{name}"] = (None, [float('nan')], float('nan'), float('nan'))

    # ── Summary ──
    print(f"\n{'='*80}", flush=True)
    print("SUMMARY TABLE", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"{'Config':<35} {'Residual':>10} {'PSNR_obs':>10} {'PSNR_all':>10}", flush=True)
    print("-"*70, flush=True)
    for name in sorted(results.keys()):
        _, res, po, pa = results[name]
        fr = res[-1] if res else float('nan')
        print(f"{name:<35} {fr:>10.0f} {po:>10.2f} {pa:>10.2f}", flush=True)

    # ── Visualization ──
    for mask_type in ["box", "random"]:
        keys = sorted([k for k in results if k.startswith(mask_type) and results[k][0] is not None])
        n = len(keys)
        if n == 0: continue
        cols = min(6, n+1)
        rows = ((n+1) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols*4.5, rows*4.5))
        if rows == 1: axes = axes[np.newaxis, :]
        axes = axes.flatten()

        img_gt = (gt[0].cpu().permute(1,2,0)*0.5+0.5).clamp(0,1).numpy()
        axes[0].imshow(img_gt); axes[0].set_title("GT", fontsize=10); axes[0].axis("off")
        for i, k in enumerate(keys):
            x1, _, po, pa = results[k]
            img = (x1[0].permute(1,2,0)*0.5+0.5).clamp(0,1).numpy()
            short = k.replace(f"{mask_type}_", "")
            axes[i+1].imshow(img)
            axes[i+1].set_title(f"{short}\nPSNR={pa:.1f}", fontsize=8)
            axes[i+1].axis("off")
        for j in range(i+2, len(axes)): axes[j].axis("off")
        plt.suptitle(f"{mask_type} inpainting — improvement exploration", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{RESULT_DIR}/explore_{mask_type}.png", dpi=150)
        plt.close()
        print(f"Saved explore_{mask_type}.png", flush=True)

    # Residual curves for best configs
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for idx, mask_type in enumerate(["box", "random"]):
        keys = sorted([k for k in results if k.startswith(mask_type)])
        for k in keys:
            _, res, _, pa = results[k]
            if not any(math.isnan(r) for r in res):
                axes[idx].plot(res, label=f"{k.replace(f'{mask_type}_', '')} ({pa:.1f})")
        axes[idx].set_yscale("log"); axes[idx].legend(fontsize=7)
        axes[idx].set_title(f"{mask_type}"); axes[idx].set_xlabel("step")
        axes[idx].set_ylabel("||A(x1)-y||^2"); axes[idx].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/explore_residuals.png", dpi=150)
    plt.close()
    print("Saved explore_residuals.png", flush=True)
    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()
