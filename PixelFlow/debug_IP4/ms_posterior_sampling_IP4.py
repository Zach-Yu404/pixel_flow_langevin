#!/usr/bin/env python
# coding: utf-8
"""
IP4 final posterior sampler — extends ms_posterior_sampling_article_version_final.py.

New knobs vs IP3 version (all backward-compatible — defaults reproduce IP3 baseline):
    guidance_scale > 0                  : CFG on (never tested in IP3)
    num_langevin / h_x / h_epsilon /
    lambda_reg / noise_scale            : scalar OR 4-list (per stage)
    inference_each_step                 : scalar OR 4-list
    dps_kick_zeta                       : scalar OR 4-list, 0 = off
    terminal_replace_weight             : float in [0,1], final x1 <- mask*y + (1-mask)*x1 blend
    soft_replace_weight                 : float in [0,1], per-step blend at stage 3
    h_x_obs_ratio                       : scalar OR 4-list, mask-aware step scaling

Chosen default profile ("IP4_winner") is set in the JSON config beside this file.
"""

import argparse, copy, math, os, shutil
import torch, torch.nn.functional as F
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.datasets import ImageFolder
from diffusers.models.embeddings import get_2d_rotary_pos_embed
from diffusers.utils.torch_utils import randn_tensor

from inpaintingStart import get_operator
from ms_posterior_sampling_article_version_final_utils import (
    apply_G, apply_H_tau, build_experiment_paths, center_crop_arr,
    compute_sigma_tau, direct_estimate_x1, get_stage_inference_steps,
    load_run_config, make_Ak_fns, make_velocity_fn, resolve_path,
    sample_block_noise, save_langevin_logs_csv, save_posterior_sampling_videos,
)
from pixelflow.scheduling_pixelflow import PixelFlowScheduler
from pixelflow.utils import config as config_utils
from pixelflow.utils.misc import seed_everything

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from langevin_v5 import principle_langevin_v5, dps_gradient_kick, terminal_replacement


def _ps(val, k, default):
    if val is None:
        return default
    if isinstance(val, (list, tuple)):
        return val[k]
    return val


def main(config_path):
    config_path = os.path.abspath(config_path)
    config_dir = os.path.dirname(config_path)
    cfg = load_run_config(config_path)

    exp_name = str(cfg["exp_name"])
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
    device_str = cfg["device"]
    device = torch.device(device_str) if torch.cuda.is_available() or not device_str.startswith("cuda") else torch.device("cpu")

    data_dir = resolve_path(config_dir, cfg["data_dir"])
    model_dir = resolve_path(config_dir, cfg["model_dir"])
    dict_root = resolve_path(config_dir, cfg["dict_path"])
    record_every = int(cfg.get("record_every", 1))
    active_operator_name = cfg["active_operator"]
    measurement_mode = cfg["measurement_mode"]
    latent_update_mode = str(cfg.get("latent_update_mode", "principle"))

    # IP3 params (scalars, unchanged defaults)
    lambda_x = float(cfg.get("lambda_x", 0.01))
    rho_s_cfg = cfg.get("rho_s", 1.0)
    rho_e_cfg = cfg.get("rho_e", 1.0)
    cg_tol = float(cfg.get("cg_tol", 1e-5))
    cg_max_iter = int(cfg.get("cg_max_iter", 50))
    warm_restart = bool(cfg.get("warm_restart", True))
    g_bypass_stage3 = bool(cfg.get("g_bypass_stage3", True))
    x1_init_mode = str(cfg.get("x1_init_mode", "model"))

    # IP4 params — scalar or length-num_stages list
    num_langevin = cfg.get("num_langevin", 10)
    h_x = cfg.get("h_x", 0.1)
    h_epsilon = cfg.get("h_epsilon", 0.01)
    lambda_reg = cfg.get("lambda_reg", 50.0)
    noise_scale = cfg.get("noise_scale", 0.0)

    # IP4 hybrid knobs
    dps_kick_zeta = cfg.get("dps_kick_zeta", 0.0)
    terminal_replace_weight = float(cfg.get("terminal_replace_weight", 0.0))
    soft_replace_weight = float(cfg.get("soft_replace_weight", 0.0))
    h_x_obs_ratio = cfg.get("h_x_obs_ratio", 1.0)

    if dict_root.endswith(".pt"):
        dict_root = os.path.dirname(dict_root)
    dict_path, config_copy_path, _ = build_experiment_paths(dict_root, exp_name, latent_update_mode)
    save_videos = bool(cfg.get("save_videos", True))
    combined_video_fps = int(cfg.get("combined_video_fps", 6))
    langevin_video_fps = int(cfg.get("langevin_video_fps", 8))
    video_dpi = int(cfg.get("video_dpi", 220))
    video_batch_size = int(cfg.get("video_batch_size", num_examples))
    save_inner_latents = bool(cfg.get("save_inner_latents", False))
    save_dict_to_pt = bool(cfg.get("save_dict_to_pt", False))

    seed_everything(seed)

    transform = transforms.Compose([
        transforms.Lambda(lambda pil: center_crop_arr(pil, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3, inplace=True),
    ])
    dataset = ImageFolder(data_dir, transform=transform)

    if class_label is None or class_label == 1000:
        selected_indices = torch.randperm(len(dataset))[:num_examples].tolist()
    else:
        class_indices = [i for i, (_, y) in enumerate(dataset.samples) if int(y) == int(class_label)]
        if len(class_indices) < num_examples:
            raise ValueError(f"class_label={class_label} has only {len(class_indices)} samples")
        selected_indices = [class_indices[i] for i in torch.randperm(len(class_indices))[:num_examples].tolist()]

    gt = torch.stack([dataset[i][0] for i in selected_indices], dim=0).to(device)
    labels = [int(dataset[i][1]) for i in selected_indices]

    box_cfg = cfg["box_operator"]; random_cfg = cfg["random_operator"]
    ops = {}
    for k, op_cfg in [("box", box_cfg), ("random", random_cfg)]:
        ops[k] = get_operator(
            "inpainting", mask_type=op_cfg["mask_type"],
            mask_len_range=None if op_cfg.get("mask_len_range") is None else tuple(op_cfg["mask_len_range"]),
            mask_prob_range=None if op_cfg.get("mask_prob_range") is None else tuple(op_cfg["mask_prob_range"]),
            resolution=resolution, device=device, sigma=sigma_n,
        )
    active_operator = ops[active_operator_name]

    config = OmegaConf.load(os.path.join(model_dir, "config.yaml"))
    model = config_utils.instantiate_from_config(config.model).to(device)
    print(f"Num of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    ckpt = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt, strict=True); model.eval()

    scheduler = PixelFlowScheduler(config.scheduler.num_train_timesteps,
                                    num_stages=config.scheduler.num_stages, gamma=-1/3)
    if num_stages != int(config.scheduler.num_stages):
        raise ValueError(f"num_stages={num_stages} != config {config.scheduler.num_stages}")
    scheduler_copy = copy.deepcopy(scheduler)

    if measurement_mode == "call":
        y = active_operator(gt).detach()
    else:
        y = active_operator.measure(gt).detach()

    # CFG setup
    uncond_label = int(model.num_classes)
    prompt_class_label = uncond_label if class_label is None else int(class_label)
    prompt_embeds = torch.tensor([prompt_class_label]*num_examples, dtype=torch.int32, device=device)
    do_cfg = guidance_scale > 0
    if do_cfg:
        neg = uncond_label * torch.ones_like(prompt_embeds)
        pe_combined = torch.cat([neg, prompt_embeds], dim=0)
    else:
        pe_combined = prompt_embeds

    mask_full = active_operator.get_mask(x=gt).float().to(device)
    if mask_full.shape[0] == 1 and num_examples > 1:
        mask_full = mask_full.expand(num_examples, -1, -1, -1)

    init_factor = 2 ** (num_stages - 1)
    h = resolution // init_factor
    w = resolution // init_factor
    shape = (num_examples, 3, h, w)

    x1_k = torch.randn(shape, device=device, dtype=torch.float32)
    eps_k = randn_tensor(shape, device=device, dtype=torch.float32)
    latent_tau = eps_k.clone()

    xts_traj, xt_next_traj, x1_traj_before, x1_traj_after, eps_traj = [], [], [], [], []
    langevin_inner_traj, langevin_inner_logs = [], []

    for stage_idx in range(num_stages):
        stage_inference_steps = get_stage_inference_steps(inference_each_step, stage_idx, num_stages)
        scheduler_copy.set_timesteps(stage_inference_steps, stage_idx, device=device, shift=shift)
        Timesteps_k = scheduler_copy.Timesteps
        start_t = scheduler_copy.start_t[stage_idx]; end_t = scheduler_copy.end_t[stage_idx]
        eff_si = stage_idx if g_bypass_stage3 else None

        if stage_idx > 0:
            h *= 2; w *= 2
            latent_tau = F.interpolate(latent_tau, size=(h, w), mode="nearest")
            ost = scheduler_copy.original_start_t[stage_idx]; gam = scheduler_copy.gamma
            al = 1/(math.sqrt(1-(1/gam))*(1-ost)+ost); be = al*(1-ost)/math.sqrt(-gam)
            noise = sample_block_noise(scheduler_copy, num_examples, 3, h, w).to(device=device, dtype=latent_tau.dtype)
            latent_tau = al*latent_tau + be*noise
            x1_k = F.interpolate(x1_k, size=(h, w), mode="nearest")
            s0 = float(start_t)
            eps_k = (latent_tau - s0*apply_G(x1_k, stage_idx=eff_si))/max(1-s0, 1e-8)

        stage_shape = (num_examples, 3, h, w)
        A_k_fn, AT_k_fn = make_Ak_fns(active_operator, y, stage_shape, device)

        if h == resolution:
            mask_k = mask_full
        else:
            mask_k = F.interpolate(mask_full, size=(h, w), mode="nearest")

        size_tensor = torch.tensor([h // model.patch_size], dtype=torch.int32, device=device)
        pos_embed = get_2d_rotary_pos_embed(
            embed_dim=model.attention_head_dim,
            crops_coords=((0, 0), (h // model.patch_size, w // model.patch_size)),
            grid_size=(h // model.patch_size, w // model.patch_size),
            device=device, output_type="pt",
        )
        rope_pos = torch.stack(pos_embed, -1)

        # Per-stage resolved parameters
        n_lang = int(_ps(num_langevin, stage_idx, 10))
        hx_k = float(_ps(h_x, stage_idx, 0.1))
        he_k = float(_ps(h_epsilon, stage_idx, 0.01))
        lam_k = float(_ps(lambda_reg, stage_idx, 50.0))
        noi_k = float(_ps(noise_scale, stage_idx, 0.0))
        zeta_k = float(_ps(dps_kick_zeta, stage_idx, 0.0))
        hr_k = float(_ps(h_x_obs_ratio, stage_idx, 1.0))
        rho_s_k = float(_ps(rho_s_cfg, stage_idx, 1.0))
        rho_e_k = float(_ps(rho_e_cfg, stage_idx, 1.0))

        for step_idx, T in enumerate(Timesteps_k):
            t_curr = scheduler_copy.t[step_idx].to(device=device, dtype=torch.float32)
            tau = float(t_curr); s_k_val = float(start_t); e_k_val = float(end_t)
            sigma_t = compute_sigma_tau(tau, s_k_val, e_k_val)
            x_tau_k = apply_H_tau(x1_k, tau, s_k_val, e_k_val, stage_idx=eff_si) + sigma_t * eps_k

            xts_traj.append(x_tau_k.detach().clone())
            x1_traj_before.append(x1_k.detach().clone())

            velocity_fn = make_velocity_fn(
                model, T, pe_combined, size_tensor, rope_pos,
                do_cfg, guidance_scale, stage_idx,
            )

            if warm_restart:
                with torch.no_grad():
                    mu = velocity_fn(x_tau_k)
                xs_h = x_tau_k - tau*mu
                xe_h = x_tau_k + (1-tau)*mu
                x1_k = direct_estimate_x1(xs_h, xe_h, s_k_val, e_k_val).detach().clone()
                if sigma_t > 1e-8:
                    eps_k = (x_tau_k - apply_H_tau(x1_k, tau, s_k_val, e_k_val, stage_idx=eff_si))/sigma_t
                else:
                    eps_k = torch.randn_like(x1_k)

            if zeta_k > 0:
                x1_k = dps_gradient_kick(x1_k, A_k_fn, AT_k_fn, y, zeta_k)
                if sigma_t > 1e-8:
                    eps_k = (x_tau_k - apply_H_tau(x1_k, tau, s_k_val, e_k_val, stage_idx=eff_si))/sigma_t

            if sigma_t < 0.01 or n_lang == 0:
                inner_traj, inner_logs = [], []
            else:
                x1_k, eps_k = principle_langevin_v5(
                    x1_init=x1_k, eps_init=eps_k,
                    tau=tau, s_k=s_k_val, e_k=e_k_val,
                    velocity_fn=velocity_fn, A_k_fn=A_k_fn, AT_k_fn=AT_k_fn,
                    y=y, sigma_n=sigma_n, h_x=hx_k, h_epsilon=he_k,
                    lambda_x=lambda_x, lambda_reg=lam_k,
                    rho_s=rho_s_k, rho_e=rho_e_k,
                    cg_tol=cg_tol, cg_max_iter=cg_max_iter,
                    num_Langevin=n_lang, device=device,
                    stage_idx=eff_si, x1_init_mode=x1_init_mode,
                    noise_scale=noi_k,
                    mask_k=mask_k, h_x_obs_ratio=hr_k,
                )
                inner_traj, inner_logs = [], []

            if soft_replace_weight > 0 and h == resolution:
                x1_k = soft_replace_weight*(mask_k*y + (1-mask_k)*x1_k) + (1-soft_replace_weight)*x1_k
                if sigma_t > 1e-8:
                    eps_k = (x_tau_k - apply_H_tau(x1_k, tau, s_k_val, e_k_val, stage_idx=eff_si))/sigma_t

            x1_traj_after.append(x1_k.detach().clone())
            eps_traj.append(eps_k.detach().clone())
            latent_tau = apply_H_tau(x1_k, tau, s_k_val, e_k_val, stage_idx=eff_si) + sigma_t * eps_k
            xt_next_traj.append(latent_tau.detach().clone())
            langevin_inner_traj.append(inner_traj)
            langevin_inner_logs.append(inner_logs)

    # Terminal replacement
    if terminal_replace_weight > 0:
        x1_k = terminal_replacement(x1_k, y, mask_full, terminal_replace_weight)

    # Saving (same as IP3 version)
    save_dir = os.path.dirname(dict_path)
    os.makedirs(save_dir, exist_ok=True)
    shutil.copy2(config_path, config_copy_path)

    if save_dict_to_pt:
        torch.save({
            "config_path": config_path, "run_config": cfg,
            "exp_name": exp_name, "selected_indices": selected_indices,
            "labels": labels, "gt": gt.detach().cpu(), "y": y.detach().cpu(),
            "xts_traj": xts_traj, "xt_next_traj": xt_next_traj,
            "x1_traj_before": x1_traj_before, "x1_traj_after": x1_traj_after,
            "eps_traj": eps_traj, "langevin_inner_traj": langevin_inner_traj,
            "x1_final": x1_k.detach().cpu(),
        }, dict_path)
        print(f"saved to {dict_path}")

    if save_videos:
        save_langevin_logs_csv(
            output_dir=save_dir, exp_name=exp_name,
            latent_update_mode=latent_update_mode,
            langevin_inner_logs_vpred=langevin_inner_logs,
            langevin_inner_logs_exppred=langevin_inner_logs,
        )
        save_posterior_sampling_videos(
            output_dir=save_dir, exp_name=exp_name,
            latent_update_mode=latent_update_mode,
            y_tensor=y.detach().cpu(),
            xts_vpred_traj=xts_traj, xt_next_vpred_traj=xt_next_traj,
            x1_vpred_traj_before_langevin=x1_traj_before,
            x1_vpred_traj_after_langevin=x1_traj_after,
            xs_vpred_traj=xts_traj, xe_vpred_traj=xt_next_traj,
            xts_exppred_traj=xts_traj, xt_next_exppred_traj=xt_next_traj,
            x1_exppred_traj_before_langevin=x1_traj_before,
            x1_exppred_traj_after_langevin=x1_traj_after,
            xs_exppred_traj=eps_traj, xe_exppred_traj=xt_next_traj,
            langevin_inner_traj_vpred=langevin_inner_traj,
            langevin_inner_traj_exppred=langevin_inner_traj,
            inference_each_step=inference_each_step, num_stages=num_stages,
            batch_size=video_batch_size, combined_fps=combined_video_fps,
            langevin_fps=langevin_video_fps, dpi=video_dpi,
            save_inner_latents=save_inner_latents,
        )


if __name__ == "__main__":
    default_cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "ms_posterior_sampling_IP4.json")
    parser = argparse.ArgumentParser(description="IP4 posterior sampling")
    parser.add_argument("--config", type=str, default=default_cfg)
    args = parser.parse_args()
    main(args.config)
