#!/usr/bin/env python
"""
Test PSNR for ms_posterior_sampling with box and random inpainting.
Runs the full pipeline and computes PSNR between reconstruction and ground truth.
"""
import argparse
import copy
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.datasets import ImageFolder

from diffusers.models.embeddings import get_2d_rotary_pos_embed
from diffusers.utils.torch_utils import randn_tensor

from inpaintingStart import get_operator
from ms_posterior_sampling_utils import (
    center_crop_arr,
    class_guidance_scale,
    get_lr,
    get_stage_inference_steps,
    get_x1_and_noise_with_stage_start_and_end2,
    load_run_config,
    langevin_sample_split_prior,
    pred_x1_x0_with_vt,
    sample_block_noise,
    update_latents_from_mode,
)
from pixelflow.scheduling_pixelflow import PixelFlowScheduler
from pixelflow.utils import config as config_utils
from pixelflow.utils.misc import seed_everything


def compute_psnr(recon, gt):
    """Compute PSNR between reconstruction and ground truth. Both in [-1,1]."""
    mse = ((recon - gt) ** 2).mean(dim=(1, 2, 3))  # per-sample MSE
    # data range is 2.0 (from -1 to 1)
    psnr_per_sample = 10 * torch.log10(4.0 / mse)
    return psnr_per_sample


def run_posterior_sampling(cfg, device):
    """Run ms_posterior_sampling and return (reconstruction_vpred, reconstruction_exppred, gt, y, mask)."""
    seed = int(cfg["seed"])
    num_stages = int(cfg["num_stages"])
    resolution = int(cfg["resolution"])
    num_examples = int(cfg["num_examples"])
    sigma_n = float(cfg["sigma_n"])
    class_label = cfg["class_label"]
    if class_label is not None:
        class_label = int(class_label)
    shift = float(cfg["shift"])
    inference_each_step = cfg["inference_each_step"]
    guidance_scale = float(cfg["guidance_scale"])
    num_langevin = int(cfg["num_langevin"])
    proj = bool(cfg["proj"])
    lr_base = float(cfg["lr_base"])
    lr_min_ratio = float(cfg["lr_min_ratio"])
    lambda_prior = float(cfg.get("lambda_prior", 1.0))
    latent_update_mode = str(cfg["latent_update_mode"])
    measurement_mode = cfg["measurement_mode"]
    active_operator_name = cfg["active_operator"]
    record_every = int(cfg.get("record_every", 1))

    config_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = cfg["data_dir"]
    model_dir = cfg["model_dir"]
    if not os.path.isabs(model_dir):
        model_dir = os.path.join(config_dir, model_dir)

    seed_everything(seed)

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    dataset = ImageFolder(data_dir, transform=transform)

    if class_label is None:
        selected_indices = torch.randperm(len(dataset))[:num_examples].tolist()
    else:
        class_indices = [i for i, (_, y_label) in enumerate(dataset.samples) if int(y_label) == int(class_label)]
        perm = torch.randperm(len(class_indices))[:num_examples]
        selected_indices = [class_indices[i] for i in perm.tolist()]

    gt = torch.stack([dataset[i][0] for i in selected_indices], dim=0).to(device)

    box_cfg = cfg["box_operator"]
    random_cfg = cfg["random_operator"]
    box_operator = get_operator(
        "inpainting",
        mask_type=box_cfg["mask_type"],
        mask_len_range=None if box_cfg.get("mask_len_range") is None else tuple(box_cfg["mask_len_range"]),
        mask_prob_range=None if box_cfg.get("mask_prob_range") is None else tuple(box_cfg["mask_prob_range"]),
        resolution=resolution, device=device, sigma=sigma_n,
    )
    random_operator = get_operator(
        "inpainting",
        mask_type=random_cfg["mask_type"],
        mask_len_range=None if random_cfg.get("mask_len_range") is None else tuple(random_cfg["mask_len_range"]),
        mask_prob_range=None if random_cfg.get("mask_prob_range") is None else tuple(random_cfg["mask_prob_range"]),
        resolution=resolution, device=device, sigma=sigma_n,
    )
    if active_operator_name == "random":
        active_operator = random_operator
    elif active_operator_name == "box":
        active_operator = box_operator
    else:
        raise ValueError(f"active_operator must be 'random' or 'box', got {active_operator_name!r}")

    config = OmegaConf.load(os.path.join(model_dir, "config.yaml"))
    model = config_utils.instantiate_from_config(config.model).to(device)
    ckpt = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt, strict=True)
    model.eval()

    scheduler = PixelFlowScheduler(
        config.scheduler.num_train_timesteps,
        num_stages=config.scheduler.num_stages,
        gamma=-1 / 3,
    )
    scheduler_copy = copy.deepcopy(scheduler)

    if measurement_mode == "call":
        y = active_operator(gt).detach()
    else:
        y = active_operator.measure(gt).detach()

    mask = active_operator.get_mask(gt)

    uncond_label = int(model.num_classes)
    prompt_class_label = uncond_label if class_label is None else int(class_label)
    prompt_embeds = torch.tensor([prompt_class_label] * num_examples, dtype=torch.int32, device=device)
    negative_prompt_embeds = uncond_label * torch.ones_like(prompt_embeds)
    do_classifier_free_guidance = guidance_scale > 0
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

    init_factor = 2 ** (num_stages - 1)
    height = resolution // init_factor
    width = resolution // init_factor
    shape = (num_examples, 3, height, width)
    latents_v = randn_tensor(shape, device=device, dtype=torch.float32)
    latents_stage = latents_v.clone()

    x1_vpred_final = None
    x1_exppred_final = None

    for stage_idx in range(num_stages):
        stage_inference_steps = get_stage_inference_steps(inference_each_step, stage_idx, num_stages)
        scheduler_copy.set_timesteps(stage_inference_steps, stage_idx, device=device, shift=shift)
        Timesteps_k = scheduler_copy.Timesteps
        start_t = scheduler_copy.start_t[stage_idx]
        end_t = scheduler_copy.end_t[stage_idx]

        if stage_idx > 0:
            height *= 2
            width *= 2
            latents_v = F.interpolate(latents_v, size=(height, width), mode="nearest")
            latents_stage = F.interpolate(latents_stage, size=(height, width), mode="nearest")

            original_start_t = scheduler_copy.original_start_t[stage_idx]
            gamma = scheduler_copy.gamma
            alpha = 1 / (math.sqrt(1 - (1 / gamma)) * (1 - original_start_t) + original_start_t)
            beta = alpha * (1 - original_start_t) / math.sqrt(-gamma)

            noise = sample_block_noise(scheduler_copy, *latents_v.shape).to(device=device, dtype=latents_v.dtype)
            latents_v = alpha * latents_v + beta * noise
            latents_stage = alpha * latents_stage + beta * noise

        size_tensor = torch.tensor([latents_v.shape[-1] // model.patch_size], dtype=torch.int32, device=device)
        pos_embed = get_2d_rotary_pos_embed(
            embed_dim=model.attention_head_dim,
            crops_coords=((0, 0), (latents_v.shape[-1] // model.patch_size, latents_v.shape[-1] // model.patch_size)),
            grid_size=(latents_v.shape[-1] // model.patch_size, latents_v.shape[-1] // model.patch_size),
            device=device, output_type="pt",
        )
        rope_pos = torch.stack(pos_embed, -1)

        for step_idx, T in enumerate(Timesteps_k):
            latent_model_input_v = torch.cat([latents_v] * 2) if do_classifier_free_guidance else latents_v
            latent_model_input_stage = torch.cat([latents_stage] * 2) if do_classifier_free_guidance else latents_stage

            timestep_v = T.expand(latent_model_input_v.shape[0]).to(latent_model_input_v.dtype)
            timestep_stage = T.expand(latent_model_input_stage.shape[0]).to(latent_model_input_stage.dtype)

            with torch.no_grad():
                noise_pred_v = model(latent_model_input_v, timestep=timestep_v, class_labels=prompt_embeds,
                                     latent_size=size_tensor, pos_embed=rope_pos)
                noise_pred_stage = model(latent_model_input_stage, timestep=timestep_stage, class_labels=prompt_embeds,
                                         latent_size=size_tensor, pos_embed=rope_pos)

            if do_classifier_free_guidance:
                scale = class_guidance_scale(guidance_scale, stage_idx)
                noise_pred_uncond_v, noise_pred_text_v = noise_pred_v.chunk(2)
                noise_pred_v = noise_pred_uncond_v + scale * (noise_pred_text_v - noise_pred_uncond_v)
                noise_pred_uncond_stage, noise_pred_text_stage = noise_pred_stage.chunk(2)
                noise_pred_stage = noise_pred_uncond_stage + scale * (noise_pred_text_stage - noise_pred_uncond_stage)

            t_curr = scheduler_copy.t[step_idx].to(device=latents_v.device, dtype=latents_v.dtype)
            t_next = scheduler_copy.t[step_idx + 1].to(device=latents_v.device, dtype=latents_v.dtype)

            pixel_values_start_v = latents_v - t_curr * noise_pred_v
            pixel_values_end_v = latents_v + (1.0 - t_curr) * noise_pred_v
            pixel_values_start_stage = latents_stage - t_curr * noise_pred_stage
            pixel_values_end_stage = latents_stage + (1.0 - t_curr) * noise_pred_stage

            x1_stagePred, x0_stagePred, _, _ = get_x1_and_noise_with_stage_start_and_end2(
                pixel_values_end_stage, pixel_values_start_stage, scheduler_copy, stage_idx)
            x1_vPred, x0_vPred = pred_x1_x0_with_vt(latents_v, noise_pred_v, T, start_t, end_t)

            ratio = float(T.detach().item()) / 1000.0
            lr = float(get_lr(ratio, lr_base=lr_base, lr_min_ratio=lr_min_ratio))

            x1_vPred_after_langevin, _, _ = langevin_sample_split_prior(
                x_init=x1_vPred, y=y, operator=active_operator,
                stage_t_end=end_t, stage_t_start=start_t,
                stage_x_end=pixel_values_end_v.detach(), stage_x_start=pixel_values_start_v.detach(),
                num_Langevin=num_langevin, sigma_n=sigma_n, lambda_prior=lambda_prior,
                step_size=lr, device=device, proj=proj, return_traj=True, record_every=record_every,
            )
            x1_stagePred_after_langevin, _, _ = langevin_sample_split_prior(
                x_init=x1_stagePred, y=y, operator=active_operator,
                stage_t_end=end_t, stage_t_start=start_t,
                stage_x_end=pixel_values_end_stage.detach(), stage_x_start=pixel_values_start_stage.detach(),
                num_Langevin=num_langevin, sigma_n=sigma_n, lambda_prior=lambda_prior,
                step_size=lr, device=device, proj=proj, return_traj=True, record_every=record_every,
            )

            x1_vpred_final = x1_vPred_after_langevin.clone()
            x1_exppred_final = x1_stagePred_after_langevin.clone()

            if latent_update_mode == "intp":
                x0_stagePred = math.sqrt(t_next) * x0_stagePred + math.sqrt(1 - t_next) * torch.randn_like(x0_stagePred)
                x0_vPred = math.sqrt(t_next) * x0_vPred + math.sqrt(1 - t_next) * torch.randn_like(x0_vPred)

            latents_stage, latents_v = update_latents_from_mode(
                latent_update_mode=latent_update_mode, scheduler=scheduler_copy,
                x1_stage=x1_stagePred_after_langevin, x1_v=x1_vPred_after_langevin,
                stage_idx=stage_idx, step_idx=step_idx + 1, device=device,
                x0_stage=x0_stagePred, x0_v=x0_vPred,
                pixel_values_start_stage=pixel_values_start_stage, pixel_values_start_v=pixel_values_start_v,
            )

        print(f"  Stage {stage_idx}/{num_stages-1} done, resolution={height}x{width}")

    return x1_vpred_final, x1_exppred_final, gt, y, mask


def main():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ms_posterior_sampling.json")
    with open(config_path, "r") as f:
        base_cfg = json.load(f)

    device = torch.device(base_cfg["device"])

    for operator_name in ["box", "random"]:
        print("=" * 70)
        print(f"  Testing: {operator_name} inpainting")
        print("=" * 70)

        cfg = copy.deepcopy(base_cfg)
        cfg["active_operator"] = operator_name
        cfg["save_videos"] = False
        cfg["save_dict_to_pt"] = False

        t0 = time.time()
        x1_vpred, x1_exppred, gt, y, mask = run_posterior_sampling(cfg, device)
        elapsed = time.time() - t0

        # PSNR: full image
        psnr_vpred_all = compute_psnr(x1_vpred, gt)
        psnr_exppred_all = compute_psnr(x1_exppred, gt)

        # PSNR: observed region only (where mask == 1)
        psnr_vpred_obs = compute_psnr(x1_vpred * mask, gt * mask)
        psnr_exppred_obs = compute_psnr(x1_exppred * mask, gt * mask)

        # PSNR: masked/inpainted region only (where mask == 0)
        inv_mask = 1.0 - mask
        if inv_mask.sum() > 0:
            mse_vpred_masked = ((x1_vpred * inv_mask - gt * inv_mask) ** 2).sum(dim=(1,2,3)) / (inv_mask.sum() * 3 / inv_mask.shape[1] + 1e-8)
            psnr_vpred_masked = 10 * torch.log10(4.0 / mse_vpred_masked)
            mse_exppred_masked = ((x1_exppred * inv_mask - gt * inv_mask) ** 2).sum(dim=(1,2,3)) / (inv_mask.sum() * 3 / inv_mask.shape[1] + 1e-8)
            psnr_exppred_masked = 10 * torch.log10(4.0 / mse_exppred_masked)
        else:
            psnr_vpred_masked = psnr_vpred_all
            psnr_exppred_masked = psnr_exppred_all

        print(f"\n  Results for {operator_name} inpainting ({elapsed:.1f}s):")
        print(f"  {'':30s} {'PSNR_all':>10s} {'PSNR_obs':>10s} {'PSNR_masked':>12s}")
        print(f"  {'-'*65}")
        for i in range(gt.shape[0]):
            print(f"  Sample {i} (vpred):            {psnr_vpred_all[i].item():10.2f} {psnr_vpred_obs[i].item():10.2f} {psnr_vpred_masked[i].item():12.2f}")
            print(f"  Sample {i} (exppred):          {psnr_exppred_all[i].item():10.2f} {psnr_exppred_obs[i].item():10.2f} {psnr_exppred_masked[i].item():12.2f}")
        print(f"  {'-'*65}")
        print(f"  Mean (vpred):                {psnr_vpred_all.mean().item():10.2f} {psnr_vpred_obs.mean().item():10.2f} {psnr_vpred_masked.mean().item():12.2f}")
        print(f"  Mean (exppred):              {psnr_exppred_all.mean().item():10.2f} {psnr_exppred_obs.mean().item():10.2f} {psnr_exppred_masked.mean().item():12.2f}")
        print()


if __name__ == "__main__":
    main()
