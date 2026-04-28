#!/usr/bin/env python
"""
Quick parameter sweep to find optimal h_x, lambda_reg, num_langevin.
Runs stage 3 only (256x256) with model, using saved GT.
Tests: (1) lambda_reg values, (2) num_langevin values, (3) h_x values
"""
import sys, os, math, copy
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
from ms_posterior_sampling_article_version_final_utils import (
    apply_G, apply_H_tau, compute_sigma_tau, direct_estimate_x1,
    make_Ak_fns, make_velocity_fn, principle_langevin_sample,
    sample_block_noise,
)
from pixelflow.scheduling_pixelflow import PixelFlowScheduler
from pixelflow.utils import config as config_utils
from pixelflow.utils.misc import seed_everything

DEVICE = "cuda:0"
RESULT_DIR = os.path.join(os.path.dirname(__file__), "results")


def run_single_stage3(model, config, gt, y, operator, sigma_n,
                      h_x=0.1, h_eps=0.01, lambda_reg=1.0, num_langevin=20,
                      num_ode_steps=10, warm_restart=True, label=""):
    """Run just stage 3 (256x256) with model. Returns final x1 and residual list."""
    B = gt.shape[0]
    seed_everything(42)  # fixed seed for fair comparison
    num_stages = 4
    resolution = 256

    scheduler = PixelFlowScheduler(
        config.scheduler.num_train_timesteps,
        num_stages=config.scheduler.num_stages, gamma=-1/3,
    )

    prompt_embeds = torch.tensor([10] * B, dtype=torch.int32, device=DEVICE)

    # Run stages 0-2 with ZERO Langevin (just flow) to get to stage 3
    init_factor = 2 ** (num_stages - 1)
    height = resolution // init_factor
    width = resolution // init_factor

    x1_k = torch.randn(B, 3, height, width, device=DEVICE)
    eps_k = randn_tensor((B, 3, height, width), device=DEVICE)
    latent_tau = eps_k.clone()

    for stage_idx in range(3):  # stages 0-2, no Langevin
        sched = copy.deepcopy(scheduler)
        sched.set_timesteps(num_ode_steps, stage_idx, device=DEVICE, shift=1.0)
        Timesteps_k = sched.Timesteps
        start_t = sched.start_t[stage_idx]
        end_t = sched.end_t[stage_idx]
        s_k = float(start_t)
        e_k = float(end_t)

        if stage_idx > 0:
            height *= 2; width *= 2
            latent_tau = F.interpolate(latent_tau, size=(height, width), mode="nearest")
            original_start_t = sched.original_start_t[stage_idx]
            gamma = sched.gamma
            alpha = 1 / (math.sqrt(1 - (1 / gamma)) * (1 - original_start_t) + original_start_t)
            beta_val = alpha * (1 - original_start_t) / math.sqrt(-gamma)
            noise = sample_block_noise(sched, B, 3, height, width).to(device=DEVICE, dtype=latent_tau.dtype)
            latent_tau = alpha * latent_tau + beta_val * noise
            x1_k = F.interpolate(x1_k, size=(height, width), mode="nearest")
            eps_k = (latent_tau - s_k * apply_G(x1_k, stage_idx=stage_idx)) / max(1.0 - s_k, 1e-8)

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
            sigma_t = compute_sigma_tau(tau, s_k, e_k)
            x_tau_k = apply_H_tau(x1_k, tau, s_k, e_k, stage_idx=stage_idx) + sigma_t * eps_k

            velocity_fn = make_velocity_fn(model, T, prompt_embeds, size_tensor, rope_pos, False, 0.0, stage_idx)
            with torch.no_grad():
                mu = velocity_fn(x_tau_k)
            xs_hat = x_tau_k - tau * mu
            xe_hat = x_tau_k + (1.0 - tau) * mu
            x1_k = direct_estimate_x1(xs_hat, xe_hat, s_k, e_k).detach()
            if sigma_t > 1e-8:
                eps_k = (x_tau_k - apply_H_tau(x1_k, tau, s_k, e_k, stage_idx=stage_idx)) / sigma_t
            else:
                eps_k = torch.randn_like(x1_k)
            latent_tau = apply_H_tau(x1_k, tau, s_k, e_k, stage_idx=stage_idx) + sigma_t * eps_k

    # Now run stage 3 WITH Langevin
    stage_idx = 3
    height *= 2; width *= 2  # 256x256
    sched = copy.deepcopy(scheduler)
    sched.set_timesteps(num_ode_steps, stage_idx, device=DEVICE, shift=1.0)
    Timesteps_k = sched.Timesteps
    start_t = sched.start_t[stage_idx]
    end_t = sched.end_t[stage_idx]
    s_k = float(start_t)
    e_k = float(end_t)

    latent_tau = F.interpolate(latent_tau, size=(height, width), mode="nearest")
    original_start_t = sched.original_start_t[stage_idx]
    gamma = sched.gamma
    alpha = 1 / (math.sqrt(1 - (1 / gamma)) * (1 - original_start_t) + original_start_t)
    beta_val = alpha * (1 - original_start_t) / math.sqrt(-gamma)
    noise = sample_block_noise(sched, B, 3, height, width).to(device=DEVICE, dtype=latent_tau.dtype)
    latent_tau = alpha * latent_tau + beta_val * noise
    x1_k = F.interpolate(x1_k, size=(height, width), mode="nearest")
    eps_k = (latent_tau - s_k * apply_G(x1_k, stage_idx=stage_idx)) / max(1.0 - s_k, 1e-8)

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

    residuals = []
    for step_idx, T in enumerate(Timesteps_k):
        t_curr = sched.t[step_idx].to(device=DEVICE, dtype=torch.float32)
        tau = float(t_curr)
        sigma_t = compute_sigma_tau(tau, s_k, e_k)
        x_tau_k = apply_H_tau(x1_k, tau, s_k, e_k, stage_idx=stage_idx) + sigma_t * eps_k

        velocity_fn = make_velocity_fn(model, T, prompt_embeds, size_tensor, rope_pos, False, 0.0, stage_idx)

        if warm_restart:
            with torch.no_grad():
                mu = velocity_fn(x_tau_k)
            xs_hat = x_tau_k - tau * mu
            xe_hat = x_tau_k + (1.0 - tau) * mu
            x1_k = direct_estimate_x1(xs_hat, xe_hat, s_k, e_k).detach()
            if sigma_t > 1e-8:
                eps_k = (x_tau_k - apply_H_tau(x1_k, tau, s_k, e_k, stage_idx=stage_idx)) / sigma_t

        if sigma_t >= 0.01:
            x1_k, eps_k, _, _ = principle_langevin_sample(
                x1_init=x1_k, eps_init=eps_k,
                tau=tau, s_k=s_k, e_k=e_k,
                velocity_fn=velocity_fn, A_k_fn=A_k_fn, AT_k_fn=AT_k_fn,
                y=y, sigma_n=sigma_n, h_x=h_x, h_epsilon=h_eps,
                lambda_x=0.01, lambda_reg=lambda_reg,
                rho_s=1.0, rho_e=1.0,
                cg_tol=1e-5, cg_max_iter=50,
                num_Langevin=num_langevin, device=DEVICE,
                stage_idx=stage_idx, x1_init_mode="model",
                return_traj=True, record_every=5,
            )

        latent_tau = apply_H_tau(x1_k, tau, s_k, e_k, stage_idx=stage_idx) + sigma_t * eps_k
        res = (y - A_k_fn(x1_k)).pow(2).sum().item()
        residuals.append(res)

    return x1_k.detach().cpu(), residuals


def main():
    print("Loading...")
    d = torch.load("article_v_results/test_box_inpainting/test_box_inpainting_mode-f_x1.pt",
                    map_location="cpu", weights_only=False)
    gt = d["gt"][:2].to(DEVICE)

    model_dir = "pretrained_models/c2img"
    config = OmegaConf.load(os.path.join(model_dir, "config.yaml"))
    model = config_utils.instantiate_from_config(config.model).to(DEVICE)
    ckpt = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt, strict=True)
    model.eval()

    sigma_n = 0.05
    operator = get_operator("inpainting", mask_type="box", mask_len_range=(80, 160),
                            mask_prob_range=None, resolution=256, device=DEVICE, sigma=sigma_n)
    y = operator(gt).detach()

    results = {}

    # Sweep lambda_reg
    for lr_val in [0.1, 1.0, 10.0, 50.0]:
        print(f"\n--- lambda_reg={lr_val}, h_x=0.1, num_langevin=20 ---")
        x1, res = run_single_stage3(model, config, gt, y, operator, sigma_n,
                                    h_x=0.1, lambda_reg=lr_val, num_langevin=20)
        results[f"lreg={lr_val}"] = (x1, res)
        print(f"  Final residual: {res[-1]:.1f}")

    # Sweep num_langevin
    for nl in [3, 5, 10, 20]:
        print(f"\n--- num_langevin={nl}, h_x=0.1, lambda_reg=1.0 ---")
        x1, res = run_single_stage3(model, config, gt, y, operator, sigma_n,
                                    h_x=0.1, lambda_reg=1.0, num_langevin=nl)
        results[f"nl={nl}"] = (x1, res)
        print(f"  Final residual: {res[-1]:.1f}")

    # No warm restart
    print(f"\n--- no warm restart, h_x=0.1, lambda_reg=1.0, nl=20 ---")
    x1, res = run_single_stage3(model, config, gt, y, operator, sigma_n,
                                h_x=0.1, lambda_reg=1.0, num_langevin=20, warm_restart=False)
    results["no_warm"] = (x1, res)
    print(f"  Final residual: {res[-1]:.1f}")

    # Larger h_x with larger lambda_reg
    print(f"\n--- h_x=0.5, lambda_reg=10.0, nl=10 ---")
    x1, res = run_single_stage3(model, config, gt, y, operator, sigma_n,
                                h_x=0.5, lambda_reg=10.0, num_langevin=10)
    results["h05_lr10"] = (x1, res)
    print(f"  Final residual: {res[-1]:.1f}")

    # Save comparison
    n_results = len(results)
    cols = min(4, n_results)
    rows = (n_results + cols - 1) // cols + 1  # +1 for GT/y row
    fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*5))
    axes = axes.flatten()

    # First row: GT, y, and residual curves
    img_gt = (gt[0].cpu().permute(1,2,0)*0.5+0.5).clamp(0,1).numpy()
    img_y = (y[0].cpu().permute(1,2,0)*0.5+0.5).clamp(0,1).numpy()
    axes[0].imshow(img_gt); axes[0].set_title("GT"); axes[0].axis("off")
    axes[1].imshow(img_y); axes[1].set_title("y (measurement)"); axes[1].axis("off")
    for name, (_, res) in results.items():
        axes[2].plot(res, label=name)
    axes[2].set_yscale("log"); axes[2].legend(fontsize=7); axes[2].set_title("Residuals")
    axes[3].axis("off")

    # Remaining: x1 images
    for i, (name, (x1, res)) in enumerate(results.items()):
        idx = cols + i
        if idx < len(axes):
            img = (x1[0].permute(1,2,0)*0.5+0.5).clamp(0,1).numpy()
            axes[idx].imshow(img)
            axes[idx].set_title(f"{name}\nres={res[-1]:.0f}", fontsize=9)
            axes[idx].axis("off")

    for ax in axes[cols + n_results:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "param_sweep_stage3.png"), dpi=150)
    plt.close()
    print("\nSaved param_sweep_stage3.png")


if __name__ == "__main__":
    main()
