#!/usr/bin/env python
"""
Diagnose why box region is smooth: track per-step score, gradient,
and x1 update decomposed by observed/unobserved pixels and frequency.
"""
import sys, os, math, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
import numpy as np

from omegaconf import OmegaConf
from diffusers.models.embeddings import get_2d_rotary_pos_embed
from diffusers.utils.torch_utils import randn_tensor

from inpaintingStart import get_operator
from ms_posterior_sampling_article_version_final_utils import (
    apply_G, apply_H_tau, apply_HT_tau, compute_sigma_tau, direct_estimate_x1,
    make_Ak_fns, make_velocity_fn, sample_block_noise, cg_solve,
)
from pixelflow.scheduling_pixelflow import PixelFlowScheduler
from pixelflow.utils import config as config_utils
from pixelflow.utils.misc import seed_everything

DEVICE = "cuda:0"


def high_freq_energy(x):
    """Fraction of energy in high-frequency (top-left quadrant excluded in FFT)."""
    fft = torch.fft.fft2(x)
    mag = fft.abs()
    H, W = x.shape[-2:]
    total = mag.pow(2).sum().item()
    # low-freq = center quarter
    lf = mag[:, :, :H//4, :W//4].pow(2).sum().item()
    return 1 - lf / max(total, 1e-12)


def main():
    print("Loading...", flush=True)
    pt_candidates = [
        "trajectory_videos/posterior_sampling/baseline_05/baseline_05_mode-f_x1.pt",
    ]
    gt = None
    for pp in pt_candidates:
        if os.path.exists(pp):
            d = torch.load(pp, map_location="cpu", weights_only=False)
            gt = d["gt"][:1].to(DEVICE)  # single image for clarity
            break
    if gt is None:
        from torchvision import transforms
        from torchvision.datasets import ImageFolder
        from ms_posterior_sampling_article_version_final_utils import center_crop_arr
        data_dir = "/data/Zach_dataset/imageNet256/ILSVRC/Data/CLS-LOC/train/"
        transform = transforms.Compose([
            transforms.Lambda(lambda pil: center_crop_arr(pil, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3, inplace=True),
        ])
        dataset = ImageFolder(data_dir, transform=transform)
        torch.manual_seed(20000120)
        ci = [i for i, (_, yl) in enumerate(dataset.samples) if int(yl) == 10]
        gt = dataset[ci[0]][0].unsqueeze(0).to(DEVICE)

    config = OmegaConf.load("pretrained_models/c2img/config.yaml")
    model = config_utils.instantiate_from_config(config.model).to(DEVICE)
    ckpt = torch.load("pretrained_models/c2img/model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt, strict=True); model.eval()
    print("Model loaded.", flush=True)

    B = 1
    sigma_n = 0.05
    operator = get_operator("inpainting", resolution=256, device=DEVICE, sigma=sigma_n,
                           mask_type="box", mask_len_range=(80,160), mask_prob_range=None)
    y = operator(gt).detach()
    full_mask = operator.get_mask(x=gt).float().to(DEVICE)  # 1=observed, 0=masked

    seed_everything(20000120)
    scheduler = PixelFlowScheduler(config.scheduler.num_train_timesteps,
                                   num_stages=config.scheduler.num_stages, gamma=-1/3)
    prompt_embeds = torch.tensor([10]*B, dtype=torch.int32, device=DEVICE)

    h, w = 32, 32
    x1_k = torch.randn(B, 3, h, w, device=DEVICE)
    eps_k = randn_tensor((B, 3, h, w), device=DEVICE)
    lat = eps_k.clone()

    lambda_reg = 50.0
    h_x = 0.1
    h_eps = 0.01
    sigma_ref_sq = 0.01

    print(f"\n{'Stage':>5} {'Step':>4} {'tau':>6} {'sig_tau':>7} "
          f"{'|g_data|':>9} {'|g_prior|':>10} "
          f"{'|delta|':>8} {'|delta|_unobs':>13} {'|delta|_obs':>12} "
          f"{'x1_hf_unobs':>12} {'x1_hf_obs':>10} "
          f"{'warm_chg':>9} {'score_prec':>10}", flush=True)
    print("-" * 140, flush=True)

    for si in range(4):
        sc = copy.deepcopy(scheduler)
        sc.set_timesteps(10, si, device=DEVICE, shift=1.0)
        sk = float(sc.start_t[si]); ek = float(sc.end_t[si])

        if si > 0:
            h *= 2; w *= 2
            lat = F.interpolate(lat, size=(h, w), mode="nearest")
            ost = sc.original_start_t[si]; gam = sc.gamma
            al = 1/(math.sqrt(1-(1/gam))*(1-ost)+ost)
            be = al*(1-ost)/math.sqrt(-gam)
            noise = sample_block_noise(sc, B, 3, h, w).to(device=DEVICE, dtype=lat.dtype)
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

        # Mask at current resolution
        if h < 256:
            mask_k = F.interpolate(full_mask, size=(h, w), mode="nearest")
        else:
            mask_k = full_mask
        unobs_mask = 1 - mask_k  # 1 where unobserved

        for step_idx, T in enumerate(sc.Timesteps):
            tau = float(sc.t[step_idx].to(DEVICE))
            sig = compute_sigma_tau(tau, sk, ek)
            xtau = apply_H_tau(x1_k, tau, sk, ek, stage_idx=si) + sig*eps_k
            vfn = make_velocity_fn(model, T, prompt_embeds, st, rope, False, 0.0, si)

            # Warm restart
            x1_before = x1_k.clone()
            with torch.no_grad(): mu = vfn(xtau)
            x1_k = direct_estimate_x1(xtau - tau*mu, xtau + (1-tau)*mu, sk, ek).detach()
            if sig > 1e-8:
                eps_k = (xtau - apply_H_tau(x1_k, tau, sk, ek, stage_idx=si))/sig
            warm_change = (x1_k - x1_before).pow(2).mean().sqrt().item()

            if sig < 0.01:
                print(f"{si:>5} {step_idx:>4} {tau:>6.3f} {sig:>7.4f}  "
                      f"{'SKIP':>9} {'':>10} {'':>8} {'':>13} {'':>12} "
                      f"{'':>12} {'':>10} {warm_change:>9.4f} {'':>10}", flush=True)
                continue

            # Run ONE Langevin step manually to see what happens
            score_prec = 1.0 / (float(sig) ** 2 + sigma_ref_sq)
            x_tau_k = apply_H_tau(x1_k, tau, sk, ek, stage_idx=si) + sig*eps_k

            mu2 = vfn(x_tau_k)
            xs_hat = x_tau_k - tau * mu2
            xe_hat = x_tau_k + (1 - tau) * mu2
            x1_hat = direct_estimate_x1(xs_hat, xe_hat, sk, ek)

            H_x1_hat = apply_H_tau(x1_hat, tau, sk, ek, stage_idx=si)
            s_flow = score_prec * (H_x1_hat - x_tau_k)

            residual = y - Ak(x1_k)
            g_data = (1.0 / sigma_n**2) * ATk(residual)
            g_prior = apply_HT_tau(s_flow, tau, sk, ek, stage_idx=si)
            g_x1 = g_data + g_prior

            def system(x):
                return (1.0 / sigma_n**2) * ATk(Ak(x)) + lambda_reg * x
            delta = cg_solve(system, (h_x/2) * g_x1, tol=1e-5, max_iter=50)

            # Decompose delta by observed/unobserved
            delta_obs_norm = (delta * mask_k).pow(2).sum().sqrt().item()
            delta_unobs_norm = (delta * unobs_mask).pow(2).sum().sqrt().item()

            # High-freq energy of x1 in observed/unobserved
            x1_obs = x1_k * mask_k
            x1_unobs = x1_k * unobs_mask
            hf_obs = high_freq_energy(x1_obs)
            hf_unobs = high_freq_energy(x1_unobs)

            print(f"{si:>5} {step_idx:>4} {tau:>6.3f} {sig:>7.4f}  "
                  f"{g_data.norm().item():>9.1f} {g_prior.norm().item():>10.1f} "
                  f"{delta.norm().item():>8.4f} {delta_unobs_norm:>13.4f} {delta_obs_norm:>12.4f} "
                  f"{hf_unobs:>12.4f} {hf_obs:>10.4f} "
                  f"{warm_change:>9.4f} {score_prec:>10.1f}", flush=True)

            # Actually apply the step for next iteration
            x1_k = x1_k + delta
            g_eps = sig * s_flow - eps_k
            eps_k = eps_k + (h_eps/2) * g_eps
            lat = apply_H_tau(x1_k, tau, sk, ek, stage_idx=si) + sig*eps_k

    # Final analysis
    print(f"\n{'='*60}", flush=True)
    xf = x1_k.detach()
    print(f"Final PSNR_all: {-10*torch.log10(((xf-gt)**2).mean()).item():.2f} dB", flush=True)

    # High-freq analysis
    xf_obs = xf * full_mask
    xf_unobs = xf * (1 - full_mask)
    gt_obs = gt * full_mask
    gt_unobs = gt * (1 - full_mask)

    print(f"\nHigh-freq energy (fraction above DC quarter):", flush=True)
    print(f"  GT observed:    {high_freq_energy(gt_obs):.4f}", flush=True)
    print(f"  GT unobserved:  {high_freq_energy(gt_unobs):.4f}", flush=True)
    print(f"  Output observed:    {high_freq_energy(xf_obs):.4f}", flush=True)
    print(f"  Output unobserved:  {high_freq_energy(xf_unobs):.4f}", flush=True)
    print(f"\nPixel std (texture proxy):", flush=True)
    n_obs = full_mask.sum().item()
    n_unobs = (1-full_mask).sum().item()
    print(f"  GT observed std:    {(gt_obs).pow(2).sum().item()/max(n_obs,1):.4f}", flush=True)
    print(f"  GT unobserved std:  {(gt_unobs).pow(2).sum().item()/max(n_unobs,1):.4f}", flush=True)
    print(f"  Out observed std:   {(xf_obs).pow(2).sum().item()/max(n_obs,1):.4f}", flush=True)
    print(f"  Out unobserved std: {(xf_unobs).pow(2).sum().item()/max(n_unobs,1):.4f}", flush=True)

    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()
