#!/usr/bin/env python
"""
Quick test of the final version — uses saved GT, runs full 4-stage pipeline,
saves images for comparison with old version and broken article version.
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
os.makedirs(RESULT_DIR, exist_ok=True)


def main():
    print("Loading GT from saved results...")
    d = torch.load(
        "article_v_results/test_box_inpainting/test_box_inpainting_mode-f_x1.pt",
        map_location="cpu", weights_only=False,
    )
    gt = d["gt"][:2].to(DEVICE)
    B = gt.shape[0]

    print("Loading model...")
    model_dir = "pretrained_models/c2img"
    config = OmegaConf.load(os.path.join(model_dir, "config.yaml"))
    model = config_utils.instantiate_from_config(config.model).to(DEVICE)
    ckpt = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt, strict=True)
    model.eval()

    sigma_n = 0.05
    operator = get_operator(
        "inpainting", mask_type="box",
        mask_len_range=(80, 160), mask_prob_range=None,
        resolution=256, device=DEVICE, sigma=sigma_n,
    )
    y = operator(gt).detach()

    scheduler = PixelFlowScheduler(
        config.scheduler.num_train_timesteps,
        num_stages=config.scheduler.num_stages, gamma=-1/3,
    )

    seed_everything(20000120)
    num_stages = 4
    resolution = 256
    num_langevin = 20
    h_x = 0.1
    h_eps = 0.01
    lambda_x = 0.01
    lambda_reg = 1.0

    uncond_label = int(model.num_classes)
    prompt_embeds = torch.tensor([10] * B, dtype=torch.int32, device=DEVICE)

    init_factor = 2 ** (num_stages - 1)
    height = resolution // init_factor
    width = resolution // init_factor

    x1_k = torch.randn(B, 3, height, width, device=DEVICE)
    eps_k = randn_tensor((B, 3, height, width), device=DEVICE, dtype=torch.float32)
    latent_tau = eps_k.clone()

    x1_images_per_stage = []
    residuals = []

    for stage_idx in range(num_stages):
        sched = copy.deepcopy(scheduler)
        sched.set_timesteps(10, stage_idx, device=DEVICE, shift=1.0)
        Timesteps_k = sched.Timesteps
        start_t = sched.start_t[stage_idx]
        end_t = sched.end_t[stage_idx]
        s_k = float(start_t)
        e_k = float(end_t)
        eff_stage_idx = stage_idx  # G bypass at stage 3

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
            eps_k = (latent_tau - s0 * apply_G(x1_k, stage_idx=eff_stage_idx)) / max(1.0 - s0, 1e-8)

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
            sigma_t = compute_sigma_tau(tau, s_k, e_k)
            x_tau_k = apply_H_tau(x1_k, tau, s_k, e_k, stage_idx=eff_stage_idx) + sigma_t * eps_k

            velocity_fn = make_velocity_fn(
                model, T, prompt_embeds, size_tensor, rope_pos, False, 0.0, stage_idx,
            )

            # Warm restart: re-init x1 from model prediction
            with torch.no_grad():
                mu = velocity_fn(x_tau_k)
            xs_hat = x_tau_k - tau * mu
            xe_hat = x_tau_k + (1.0 - tau) * mu
            x1_model = direct_estimate_x1(xs_hat, xe_hat, s_k, e_k)
            x1_k = x1_model.detach().clone()
            if sigma_t > 1e-8:
                eps_k = (x_tau_k - apply_H_tau(x1_k, tau, s_k, e_k, stage_idx=eff_stage_idx)) / sigma_t
            else:
                eps_k = torch.randn_like(x1_k)

            if sigma_t < 0.01:
                pass
            else:
                x1_k, eps_k, _, logs = principle_langevin_sample(
                    x1_init=x1_k, eps_init=eps_k,
                    tau=tau, s_k=s_k, e_k=e_k,
                    velocity_fn=velocity_fn, A_k_fn=A_k_fn, AT_k_fn=AT_k_fn,
                    y=y, sigma_n=sigma_n, h_x=h_x, h_epsilon=h_eps,
                    lambda_x=lambda_x, lambda_reg=lambda_reg,
                    rho_s=1.0, rho_e=1.0,
                    cg_tol=1e-5, cg_max_iter=50,
                    num_Langevin=num_langevin, device=DEVICE,
                    stage_idx=eff_stage_idx, x1_init_mode="model",
                    return_traj=True, record_every=5,
                )
                if logs:
                    print(f"  Stage {stage_idx} step {step_idx}: l3={logs[-1]['l3']:.1f} delta={logs[-1]['delta_norm']:.4f}")

            latent_tau = apply_H_tau(x1_k, tau, s_k, e_k, stage_idx=eff_stage_idx) + sigma_t * eps_k

            # Residual at full resolution
            if height == 256:
                res = (y - A_k_fn(x1_k)).pow(2).sum().item()
            else:
                x1_up = F.interpolate(x1_k, size=(256, 256), mode="bilinear", align_corners=False)
                A_full, _ = make_Ak_fns(operator, y, (B, 3, 256, 256), DEVICE)
                res = (y - A_full(x1_up)).pow(2).sum().item()
            residuals.append(res)

        x1_images_per_stage.append(x1_k.detach().cpu())
        print(f"Stage {stage_idx} ({height}x{width}) done. Final residual: {residuals[-1]:.1f}")

    # Save comparison
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    # Row 0: GT, y, final x1 (sample 0), final x1 (sample 1), residual curve
    img_gt0 = (gt[0].cpu().permute(1,2,0)*0.5+0.5).clamp(0,1).numpy()
    img_gt1 = (gt[1].cpu().permute(1,2,0)*0.5+0.5).clamp(0,1).numpy() if B > 1 else img_gt0
    img_y = (y[0].cpu().permute(1,2,0)*0.5+0.5).clamp(0,1).numpy()
    img_x1 = (x1_images_per_stage[-1][0].permute(1,2,0)*0.5+0.5).clamp(0,1).numpy()
    img_x1_2 = (x1_images_per_stage[-1][1].permute(1,2,0)*0.5+0.5).clamp(0,1).numpy() if B > 1 else img_x1

    axes[0,0].imshow(img_gt0); axes[0,0].set_title("GT (sample 0)"); axes[0,0].axis("off")
    axes[0,1].imshow(img_y); axes[0,1].set_title("y (measurement)"); axes[0,1].axis("off")
    axes[0,2].imshow(img_x1); axes[0,2].set_title(f"x1 final (sample 0)\nres={residuals[-1]:.0f}"); axes[0,2].axis("off")
    axes[0,3].imshow(img_x1_2); axes[0,3].set_title("x1 final (sample 1)"); axes[0,3].axis("off")
    axes[0,4].plot(residuals); axes[0,4].set_ylabel("||A(x1)-y||²"); axes[0,4].set_xlabel("step"); axes[0,4].set_title("Residual curve"); axes[0,4].set_yscale("log")

    # Row 1: stage-by-stage x1 (sample 0)
    for i, x1_stage in enumerate(x1_images_per_stage):
        if i < 4:
            h = 32 * (2**i)
            x1_up = F.interpolate(x1_stage[:1], size=(256,256), mode="bilinear", align_corners=False) if h < 256 else x1_stage[:1]
            img = (x1_up[0].permute(1,2,0)*0.5+0.5).clamp(0,1).numpy()
            axes[1,i].imshow(img); axes[1,i].set_title(f"Stage {i} ({h}x{h})"); axes[1,i].axis("off")
    axes[1,4].imshow(img_gt1); axes[1,4].set_title("GT (sample 1)"); axes[1,4].axis("off")

    plt.suptitle("PRINCIPLE Final Version — Fixed", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "final_version_test.png"), dpi=150)
    plt.close()
    print(f"Saved final_version_test.png")
    print(f"Final residual: {residuals[-1]:.1f}")


if __name__ == "__main__":
    main()
