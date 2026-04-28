#!/usr/bin/env python
"""Test refined parameters: lambda_reg=50, num_langevin=5/10, full 4-stage pipeline."""
import sys, os, math, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
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


def run_pipeline(model, config, gt, y, operator, sigma_n,
                 h_x=0.1, h_eps=0.01, lambda_reg=50.0, num_langevin=10):
    B = gt.shape[0]
    seed_everything(20000120)
    scheduler = PixelFlowScheduler(
        config.scheduler.num_train_timesteps,
        num_stages=config.scheduler.num_stages, gamma=-1/3,
    )
    prompt_embeds = torch.tensor([10] * B, dtype=torch.int32, device=DEVICE)
    height, width = 32, 32
    x1_k = torch.randn(B, 3, height, width, device=DEVICE)
    eps_k = randn_tensor((B, 3, height, width), device=DEVICE)
    latent_tau = eps_k.clone()
    residuals = []

    for stage_idx in range(4):
        sched = copy.deepcopy(scheduler)
        sched.set_timesteps(10, stage_idx, device=DEVICE, shift=1.0)
        s_k = float(sched.start_t[stage_idx])
        e_k = float(sched.end_t[stage_idx])

        if stage_idx > 0:
            height *= 2; width *= 2
            latent_tau = F.interpolate(latent_tau, size=(height, width), mode="nearest")
            ost = sched.original_start_t[stage_idx]; gamma = sched.gamma
            alpha = 1/(math.sqrt(1-(1/gamma))*(1-ost)+ost)
            beta = alpha*(1-ost)/math.sqrt(-gamma)
            noise = sample_block_noise(sched, B, 3, height, width).to(device=DEVICE, dtype=latent_tau.dtype)
            latent_tau = alpha*latent_tau + beta*noise
            x1_k = F.interpolate(x1_k, size=(height, width), mode="nearest")
            eps_k = (latent_tau - s_k*apply_G(x1_k, stage_idx=stage_idx))/max(1-s_k, 1e-8)

        A_k_fn, AT_k_fn = make_Ak_fns(operator, y, (B, 3, height, width), DEVICE)
        st = torch.tensor([height//model.patch_size], dtype=torch.int32, device=DEVICE)
        pe = get_2d_rotary_pos_embed(
            embed_dim=model.attention_head_dim,
            crops_coords=((0,0),(height//model.patch_size,width//model.patch_size)),
            grid_size=(height//model.patch_size,width//model.patch_size),
            device=DEVICE, output_type="pt",
        )
        rope = torch.stack(pe, -1)

        for step_idx, T in enumerate(sched.Timesteps):
            tau = float(sched.t[step_idx].to(DEVICE))
            sigma_t = compute_sigma_tau(tau, s_k, e_k)
            x_tau_k = apply_H_tau(x1_k, tau, s_k, e_k, stage_idx=stage_idx) + sigma_t*eps_k

            vfn = make_velocity_fn(model, T, prompt_embeds, st, rope, False, 0.0, stage_idx)

            # Warm restart
            with torch.no_grad():
                mu = vfn(x_tau_k)
            x1_k = direct_estimate_x1(x_tau_k - tau*mu, x_tau_k + (1-tau)*mu, s_k, e_k).detach()
            if sigma_t > 1e-8:
                eps_k = (x_tau_k - apply_H_tau(x1_k, tau, s_k, e_k, stage_idx=stage_idx))/sigma_t

            if sigma_t >= 0.01:
                x1_k, eps_k, _, _ = principle_langevin_sample(
                    x1_init=x1_k, eps_init=eps_k, tau=tau, s_k=s_k, e_k=e_k,
                    velocity_fn=vfn, A_k_fn=A_k_fn, AT_k_fn=AT_k_fn,
                    y=y, sigma_n=sigma_n, h_x=h_x, h_epsilon=h_eps,
                    lambda_x=0.01, lambda_reg=lambda_reg, rho_s=1.0, rho_e=1.0,
                    cg_tol=1e-5, cg_max_iter=50, num_Langevin=num_langevin,
                    device=DEVICE, stage_idx=stage_idx, x1_init_mode="model",
                    return_traj=True, record_every=5)

            latent_tau = apply_H_tau(x1_k, tau, s_k, e_k, stage_idx=stage_idx) + sigma_t*eps_k
            if height == 256:
                res = (y - A_k_fn(x1_k)).pow(2).sum().item()
            else:
                x1_up = F.interpolate(x1_k, size=(256,256), mode="bilinear", align_corners=False)
                A_full, _ = make_Ak_fns(operator, y, (B,3,256,256), DEVICE)
                res = (y - A_full(x1_up)).pow(2).sum().item()
            residuals.append(res)

    return x1_k.detach().cpu(), residuals


def main():
    print("Loading...", flush=True)
    d = torch.load("article_v_results/test_box_inpainting/test_box_inpainting_mode-f_x1.pt",
                    map_location="cpu", weights_only=False)
    gt = d["gt"][:2].to(DEVICE)
    model_dir = "pretrained_models/c2img"
    config = OmegaConf.load(os.path.join(model_dir, "config.yaml"))
    model = config_utils.instantiate_from_config(config.model).to(DEVICE)
    ckpt = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt, strict=True); model.eval()

    sigma_n = 0.05
    operator = get_operator("inpainting", mask_type="box", mask_len_range=(80,160),
                            mask_prob_range=None, resolution=256, device=DEVICE, sigma=sigma_n)
    y = operator(gt).detach()

    configs = [
        ("lreg50_nl10", dict(h_x=0.1, lambda_reg=50.0, num_langevin=10)),
        ("lreg50_nl5",  dict(h_x=0.1, lambda_reg=50.0, num_langevin=5)),
        ("lreg50_nl20", dict(h_x=0.1, lambda_reg=50.0, num_langevin=20)),
        ("lreg100_nl10", dict(h_x=0.1, lambda_reg=100.0, num_langevin=10)),
    ]

    results = {}
    for name, kw in configs:
        print(f"\n--- {name}: {kw} ---", flush=True)
        x1, res = run_pipeline(model, config, gt, y, operator, sigma_n, **kw)
        results[name] = (x1, res)
        print(f"  Final residual: {res[-1]:.1f}", flush=True)

    # Save figure
    ncols = len(results) + 2
    fig, axes = plt.subplots(2, ncols, figsize=(ncols*5, 10))

    # Row 0: GT, y, then each result (sample 0)
    img_gt = (gt[0].cpu().permute(1,2,0)*0.5+0.5).clamp(0,1).numpy()
    img_y = (y[0].cpu().permute(1,2,0)*0.5+0.5).clamp(0,1).numpy()
    axes[0,0].imshow(img_gt); axes[0,0].set_title("GT"); axes[0,0].axis("off")
    axes[0,1].imshow(img_y); axes[0,1].set_title("y"); axes[0,1].axis("off")
    for i, (name, (x1, res)) in enumerate(results.items()):
        img = (x1[0].permute(1,2,0)*0.5+0.5).clamp(0,1).numpy()
        axes[0,i+2].imshow(img)
        axes[0,i+2].set_title(f"{name}\nres={res[-1]:.0f}", fontsize=9)
        axes[0,i+2].axis("off")

    # Row 1: sample 1 + residual curves
    if gt.shape[0] > 1:
        img_gt1 = (gt[1].cpu().permute(1,2,0)*0.5+0.5).clamp(0,1).numpy()
        axes[1,0].imshow(img_gt1); axes[1,0].set_title("GT (s1)"); axes[1,0].axis("off")
        img_y1 = (y[1].cpu().permute(1,2,0)*0.5+0.5).clamp(0,1).numpy()
        axes[1,1].imshow(img_y1); axes[1,1].set_title("y (s1)"); axes[1,1].axis("off")
        for i, (name, (x1, res)) in enumerate(results.items()):
            if x1.shape[0] > 1:
                img = (x1[1].permute(1,2,0)*0.5+0.5).clamp(0,1).numpy()
                axes[1,i+2].imshow(img)
                axes[1,i+2].set_title(f"{name} (s1)", fontsize=9)
                axes[1,i+2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "final_refined.png"), dpi=150)
    plt.close()

    # Residual plot
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, (_, res) in results.items():
        ax.plot(res, label=name)
    ax.set_yscale("log"); ax.legend(); ax.set_xlabel("Step"); ax.set_ylabel("||A(x1)-y||²")
    ax.set_title("Residual convergence — refined params")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "final_refined_residuals.png"), dpi=150)
    plt.close()
    print("\nSaved final_refined.png and final_refined_residuals.png", flush=True)


if __name__ == "__main__":
    main()
