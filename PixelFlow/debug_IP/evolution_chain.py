#!/usr/bin/env python
"""
Evolution chain: systematically test fixes from old→new to find what breaks IP.
Uses actual model for velocity predictions. Tests at stage 3 (256x256) where
the issue is most visible, running just 10 ODE steps with different Langevin configs.

Each experiment:
1. Loads model once
2. Runs a small pipeline at stage 3 only (starts from saved stage 2 output or random)
3. Saves x1 image after Langevin
4. Measures residual ||A(x1) - y||²

Experiments:
A. Old-style Langevin (autograd, model-predicted x1 init, lr=0.02)
B. New-style Langevin but with model-predicted x1 init each step (warm restart)
C. New-style with warm restart + stage 3 G bypass
D. New-style with warm restart + G bypass + tuned h_x/lambda_reg
E. New-style original (cold x1, G always low-pass, h_x=0.001)
"""
import sys, os, math, copy, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from omegaconf import OmegaConf
from diffusers.models.embeddings import get_2d_rotary_pos_embed
from diffusers.utils.torch_utils import randn_tensor

from inpaintingStart import get_operator
from ms_posterior_sampling_utils import (
    center_crop_arr, sample_block_noise, DownUp_operation,
    langevin_update_split_prior, pred_x1_x0_with_vt,
    get_xt_from_x1_sameStage, get_lr, class_guidance_scale,
    get_x1_and_noise_with_stage_start_and_end2,
)
from ms_posterior_sampling_article_version_utils import (
    apply_G, apply_H_tau, apply_HT_tau, compute_sigma_tau,
    wls_estimate_x1, make_Ak_fns, cg_solve, _langevin_step,
    principle_langevin_sample, make_velocity_fn,
)
from pixelflow.scheduling_pixelflow import PixelFlowScheduler
from pixelflow.utils import config as config_utils
from pixelflow.utils.misc import seed_everything

DEVICE = "cuda:0"
RESULT_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULT_DIR, exist_ok=True)

LOG = []
def log(msg):
    print(msg)
    LOG.append(msg)


def load_model():
    model_dir = "pretrained_models/c2img"
    config = OmegaConf.load(os.path.join(model_dir, "config.yaml"))
    model = config_utils.instantiate_from_config(config.model).to(DEVICE)
    ckpt = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt, strict=True)
    model.eval()
    return model, config


def load_gt_and_operator():
    """Load GT from existing saved results."""
    d = torch.load(
        "article_v_results/test_box_inpainting/test_box_inpainting_mode-f_x1.pt",
        map_location="cpu", weights_only=False
    )
    gt = d["gt"][:2].to(DEVICE)

    sigma_n = 0.05
    operator = get_operator(
        "inpainting", mask_type="box",
        mask_len_range=(80, 160), mask_prob_range=None,
        resolution=256, device=DEVICE, sigma=sigma_n,
    )
    y = operator(gt).detach()  # noiseless measurement for cleaner testing
    return gt, y, operator, sigma_n


def run_full_pipeline(model, config, gt, y, operator, sigma_n, experiment_name, **kwargs):
    """Run full 4-stage pipeline with specified Langevin config."""
    log(f"\n{'='*60}")
    log(f"EXPERIMENT: {experiment_name}")
    log(f"  kwargs: {kwargs}")
    log(f"{'='*60}")

    B = gt.shape[0]
    seed_everything(20000120)
    num_stages = 4
    resolution = 256
    num_langevin = kwargs.get("num_langevin", 20)
    h_x = kwargs.get("h_x", 0.001)
    h_eps = kwargs.get("h_eps", 0.001)
    lambda_x = kwargs.get("lambda_x", 0.01)
    lambda_reg = kwargs.get("lambda_reg", 0.01)
    rho_s_k = kwargs.get("rho_s", 1.0)
    rho_e_k = kwargs.get("rho_e", 1.0)
    use_warm_restart = kwargs.get("warm_restart", False)
    g_bypass_stage3 = kwargs.get("g_bypass_stage3", False)
    use_old_langevin = kwargs.get("use_old_langevin", False)
    lr_base = kwargs.get("lr_base", 0.02)
    lambda_prior = kwargs.get("lambda_prior", 0.00001)

    scheduler = PixelFlowScheduler(
        config.scheduler.num_train_timesteps,
        num_stages=config.scheduler.num_stages, gamma=-1/3,
    )

    # CFG setup (no guidance)
    uncond_label = int(model.num_classes)
    prompt_embeds = torch.tensor([10] * B, dtype=torch.int32, device=DEVICE)

    init_factor = 2 ** (num_stages - 1)
    height = resolution // init_factor
    width = resolution // init_factor

    x1_k = torch.randn(B, 3, height, width, device=DEVICE)
    eps_k = randn_tensor((B, 3, height, width), device=DEVICE, dtype=torch.float32)
    latent_tau = eps_k.clone()

    residuals_per_step = []
    x1_images = []

    for stage_idx in range(num_stages):
        sched = copy.deepcopy(scheduler)
        stage_steps = 10
        sched.set_timesteps(stage_steps, stage_idx, device=DEVICE, shift=1.0)
        Timesteps_k = sched.Timesteps
        start_t = sched.start_t[stage_idx]
        end_t = sched.end_t[stage_idx]
        s_k = float(start_t)
        e_k = float(end_t)

        if stage_idx > 0:
            height *= 2
            width *= 2
            latent_tau = F.interpolate(latent_tau, size=(height, width), mode="nearest")
            original_start_t = sched.original_start_t[stage_idx]
            gamma = sched.gamma
            alpha = 1 / (math.sqrt(1 - (1 / gamma)) * (1 - original_start_t) + original_start_t)
            beta = alpha * (1 - original_start_t) / math.sqrt(-gamma)
            noise = sample_block_noise(sched, B, 3, height, width).to(device=DEVICE, dtype=latent_tau.dtype)
            latent_tau = alpha * latent_tau + beta * noise

            x1_k = F.interpolate(x1_k, size=(height, width), mode="nearest")
            s0 = float(start_t)
            eps_k = (latent_tau - s0 * apply_G(x1_k)) / max(1.0 - s0, 1e-8)

        stage_shape = (B, 3, height, width)
        A_k_fn, AT_k_fn = make_Ak_fns(operator, y, stage_shape, DEVICE)

        size_tensor = torch.tensor([height // model.patch_size], dtype=torch.int32, device=DEVICE)
        pos_embed = get_2d_rotary_pos_embed(
            embed_dim=model.attention_head_dim,
            crops_coords=((0, 0), (height // model.patch_size, width // model.patch_size)),
            grid_size=(height // model.patch_size, width // model.patch_size),
            device=DEVICE, output_type="pt",
        )
        rope_pos = torch.stack(pos_embed, -1)

        for step_idx, T in enumerate(Timesteps_k):
            t_curr = sched.t[step_idx].to(device=DEVICE, dtype=torch.float32)
            tau = float(t_curr)
            sigma_tau_val = compute_sigma_tau(tau, s_k, e_k)

            # Model prediction
            x_tau_k = apply_H_tau(x1_k, tau, s_k, e_k) + sigma_tau_val * eps_k
            inp = x_tau_k
            ts = T.expand(B).to(inp.dtype)
            with torch.no_grad():
                noise_pred = model(inp, timestep=ts, class_labels=prompt_embeds,
                                   latent_size=size_tensor, pos_embed=rope_pos)

            if use_old_langevin:
                # Old-style: extract x1 from model, then standard ULA
                x1_pred = x_tau_k + (1.0 - tau) / max(e_k - s_k, 1e-8) * noise_pred
                xs_pred = x_tau_k - tau * noise_pred
                xe_pred = x_tau_k + (1.0 - tau) * noise_pred

                ratio = float(T.detach().item()) / 1000.0
                lr = get_lr(ratio, lr_base=lr_base, lr_min_ratio=0.01)

                x1_refined = x1_pred.clone()
                for _ in range(num_langevin):
                    x1_refined, _ = langevin_update_split_prior(
                        x1_refined, y, operator, end_t, start_t,
                        xe_pred.detach(), xs_pred.detach(),
                        sigma_n, step_size=lr, proj=False,
                        o_M=None, lambda_prior=lambda_prior, device=DEVICE,
                    )
                x1_k = x1_refined.detach()

                # Reconstruct latent at next timestep using f_x1 mode
                t_next_val = float(sched.t[step_idx + 1].to(DEVICE))
                if stage_idx == 3:
                    DU_x1 = x1_k  # identity at stage 3
                else:
                    DU_x1 = DownUp_operation(x1_k, scale_factor=2, stage_idx=None)
                xe_recon = e_k * x1_k + (1-e_k) / max(1-s_k, 1e-8) * (xs_pred - s_k * DU_x1)
                latent_tau = t_next_val * xe_recon + (1 - t_next_val) * xs_pred
                # Update eps_k for consistency
                sigma_next = compute_sigma_tau(t_next_val, s_k, e_k)
                if sigma_next > 1e-8:
                    eps_k = (latent_tau - apply_H_tau(x1_k, t_next_val, s_k, e_k)) / sigma_next
                else:
                    eps_k = torch.randn_like(x1_k)

            else:
                # New-style PRINCIPLE Langevin
                if use_warm_restart:
                    # Re-init x1_k from model prediction (like old code)
                    x1_model = x_tau_k + (1.0 - tau) / max(e_k - s_k, 1e-8) * noise_pred
                    x1_k = x1_model.detach().clone()

                if sigma_tau_val < 0.01:
                    pass
                else:
                    # Optionally bypass G at stage 3
                    if g_bypass_stage3 and stage_idx == 3:
                        # Use identity instead of G at stage 3
                        _orig_apply_G = None  # we'll monkey-patch
                        import ms_posterior_sampling_article_version_utils as av_utils
                        _orig_G = av_utils.apply_G
                        av_utils.apply_G = lambda x, scale_factor=2: x  # identity
                        try:
                            x1_k, eps_k, _, _ = principle_langevin_sample(
                                x1_init=x1_k, eps_init=eps_k,
                                tau=tau, s_k=s_k, e_k=e_k,
                                velocity_fn=make_velocity_fn(model, T, prompt_embeds, size_tensor, rope_pos, False, 0.0, stage_idx),
                                A_k_fn=A_k_fn, AT_k_fn=AT_k_fn,
                                y=y, sigma_n=sigma_n, h_x=h_x, h_epsilon=h_eps,
                                lambda_x=lambda_x, lambda_reg=lambda_reg,
                                rho_s=rho_s_k, rho_e=rho_e_k,
                                cg_tol=1e-5, cg_max_iter=50,
                                num_Langevin=num_langevin, device=DEVICE,
                                return_traj=True, record_every=1,
                            )
                        finally:
                            av_utils.apply_G = _orig_G
                    else:
                        x1_k, eps_k, _, _ = principle_langevin_sample(
                            x1_init=x1_k, eps_init=eps_k,
                            tau=tau, s_k=s_k, e_k=e_k,
                            velocity_fn=make_velocity_fn(model, T, prompt_embeds, size_tensor, rope_pos, False, 0.0, stage_idx),
                            A_k_fn=A_k_fn, AT_k_fn=AT_k_fn,
                            y=y, sigma_n=sigma_n, h_x=h_x, h_epsilon=h_eps,
                            lambda_x=lambda_x, lambda_reg=lambda_reg,
                            rho_s=rho_s_k, rho_e=rho_e_k,
                            cg_tol=1e-5, cg_max_iter=50,
                            num_Langevin=num_langevin, device=DEVICE,
                            return_traj=True, record_every=1,
                        )

                sigma_tau_next = compute_sigma_tau(tau, s_k, e_k)
                latent_tau = apply_H_tau(x1_k, tau, s_k, e_k) + sigma_tau_next * eps_k

            # Measure residual
            if height == 256:
                res = (y - A_k_fn(x1_k)).pow(2).sum().item()
            else:
                x1_up = F.interpolate(x1_k, size=(256, 256), mode="bilinear", align_corners=False)
                A_full, _ = make_Ak_fns(operator, y, (B, 3, 256, 256), DEVICE)
                res = (y - A_full(x1_up)).pow(2).sum().item()
            residuals_per_step.append(res)

        x1_images.append(x1_k.detach().cpu())
        log(f"  Stage {stage_idx} ({height}x{width}): final residual = {residuals_per_step[-1]:.2f}")

    return x1_images[-1], residuals_per_step


def main():
    log("Loading model...")
    model, config = load_model()
    log("Loading GT and operator...")
    gt, y, operator, sigma_n = load_gt_and_operator()
    log(f"GT: {gt.shape}, y: {y.shape}")

    experiments = {}

    # A: Old-style Langevin (baseline that works)
    x1_A, res_A = run_full_pipeline(
        model, config, gt, y, operator, sigma_n,
        "A: Old-style Langevin",
        use_old_langevin=True, lr_base=0.02, lambda_prior=0.00001,
        num_langevin=20,
    )
    experiments["A_old_style"] = (x1_A, res_A)

    # B: New-style with warm restart (re-init x1 from model each step)
    x1_B, res_B = run_full_pipeline(
        model, config, gt, y, operator, sigma_n,
        "B: New PRINCIPLE + warm restart",
        warm_restart=True, h_x=0.001, h_eps=0.001,
        lambda_reg=0.01, num_langevin=20,
    )
    experiments["B_warm_restart"] = (x1_B, res_B)

    # C: New-style + warm restart + G bypass at stage 3
    x1_C, res_C = run_full_pipeline(
        model, config, gt, y, operator, sigma_n,
        "C: New + warm restart + G bypass stage3",
        warm_restart=True, g_bypass_stage3=True,
        h_x=0.001, h_eps=0.001, lambda_reg=0.01, num_langevin=20,
    )
    experiments["C_warm_G_bypass"] = (x1_C, res_C)

    # D: New-style + warm restart + G bypass + tuned params
    x1_D, res_D = run_full_pipeline(
        model, config, gt, y, operator, sigma_n,
        "D: New + warm + G bypass + h_x=0.1 lambda_reg=1.0",
        warm_restart=True, g_bypass_stage3=True,
        h_x=0.1, h_eps=0.01, lambda_reg=1.0, num_langevin=20,
    )
    experiments["D_tuned"] = (x1_D, res_D)

    # E: Original new-style (expected to fail)
    x1_E, res_E = run_full_pipeline(
        model, config, gt, y, operator, sigma_n,
        "E: Original new-style (should fail)",
        warm_restart=False, g_bypass_stage3=False,
        h_x=0.001, h_eps=0.001, lambda_reg=0.01, num_langevin=20,
    )
    experiments["E_original_new"] = (x1_E, res_E)

    # F: New-style cold start + G bypass + aggressive params (no warm restart)
    x1_F, res_F = run_full_pipeline(
        model, config, gt, y, operator, sigma_n,
        "F: Cold start + G bypass + h_x=0.1 lambda_reg=1.0",
        warm_restart=False, g_bypass_stage3=True,
        h_x=0.1, h_eps=0.01, lambda_reg=1.0, num_langevin=20,
    )
    experiments["F_cold_tuned"] = (x1_F, res_F)

    # Save comparison figure
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    labels = list(experiments.keys())
    for i, name in enumerate(labels[:4]):
        x1, res = experiments[name]
        img = (x1[0].permute(1,2,0)*0.5+0.5).clamp(0,1).numpy()
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"{name}\nres={res[-1]:.0f}")
        axes[0, i].axis("off")

    for i, name in enumerate(labels[4:]):
        x1, res = experiments[name]
        img = (x1[0].permute(1,2,0)*0.5+0.5).clamp(0,1).numpy()
        axes[1, i].imshow(img)
        axes[1, i].set_title(f"{name}\nres={res[-1]:.0f}")
        axes[1, i].axis("off")

    # GT and y
    img_gt = (gt[0].cpu().permute(1,2,0)*0.5+0.5).clamp(0,1).numpy()
    img_y = (y[0].cpu().permute(1,2,0)*0.5+0.5).clamp(0,1).numpy()
    axes[1, 2].imshow(img_gt); axes[1, 2].set_title("GT"); axes[1, 2].axis("off")
    axes[1, 3].imshow(img_y); axes[1, 3].set_title("y (measurement)"); axes[1, 3].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "evolution_chain.png"), dpi=150)
    plt.close()
    log("\nSaved evolution_chain.png")

    # Residual curves
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    for name, (_, res) in experiments.items():
        ax.plot(res, label=name)
    ax.set_xlabel("Step (across all stages)")
    ax.set_ylabel("||A(x1) - y||²")
    ax.set_title("Residual evolution across experiments")
    ax.legend()
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "evolution_residuals.png"), dpi=150)
    plt.close()
    log("Saved evolution_residuals.png")

    # Save log
    with open(os.path.join(RESULT_DIR, "evolution_chain.log"), "w") as f:
        f.write("\n".join(LOG))


if __name__ == "__main__":
    main()
