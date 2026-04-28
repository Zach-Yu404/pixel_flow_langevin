#!/usr/bin/env python
# coding: utf-8

import argparse
import copy
import math
import os
import shutil

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.datasets import ImageFolder

from diffusers.models.embeddings import get_2d_rotary_pos_embed
from diffusers.utils.torch_utils import randn_tensor

from inpaintingStart import get_operator
from ms_posterior_sampling_article_version_utils import (
    apply_G,
    apply_H_tau,
    build_experiment_paths,
    center_crop_arr,
    compute_sigma_tau,
    get_stage_inference_steps,
    load_run_config,
    make_Ak_fns,
    make_velocity_fn,
    principle_langevin_sample,
    resolve_path,
    sample_block_noise,
    save_langevin_logs_csv,
    save_posterior_sampling_videos,
)
from pixelflow.scheduling_pixelflow import PixelFlowScheduler
from pixelflow.utils import config as config_utils
from pixelflow.utils.misc import seed_everything


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
    num_langevin = int(cfg["num_langevin"])
    device_str = cfg["device"]
    if isinstance(device_str, str) and device_str.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(device_str)
    data_dir = resolve_path(config_dir, cfg["data_dir"])
    model_dir = resolve_path(config_dir, cfg["model_dir"])
    dict_root = resolve_path(config_dir, cfg["dict_path"])
    record_every = int(cfg.get("record_every", 1))
    active_operator_name = cfg["active_operator"]
    measurement_mode = cfg["measurement_mode"]
    latent_update_mode = str(cfg.get("latent_update_mode", "principle"))

    # PRINCIPLE parameters
    h_x = float(cfg["h_x"])
    h_epsilon = float(cfg["h_epsilon"])
    lambda_x = float(cfg["lambda_x"])
    lambda_reg = float(cfg["lambda_reg"])
    rho_s_cfg = cfg["rho_s"]
    rho_e_cfg = cfg["rho_e"]
    cg_tol = float(cfg["cg_tol"])
    cg_max_iter = int(cfg["cg_max_iter"])

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

    # ── data ───────────────────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3, inplace=True),
    ])
    dataset = ImageFolder(data_dir, transform=transform)

    if class_label is None or class_label == 1000:
        selected_indices = torch.randperm(len(dataset))[:num_examples].tolist()
    else:
        class_indices = [i for i, (_, y_label) in enumerate(dataset.samples) if int(y_label) == int(class_label)]
        if len(class_indices) < num_examples:
            raise ValueError(f"class_label={class_label} has only {len(class_indices)} samples, need {num_examples}")
        selected_indices = [class_indices[i] for i in torch.randperm(len(class_indices))[:num_examples].tolist()]

    gt = torch.stack([dataset[i][0] for i in selected_indices], dim=0).to(device)
    labels = [int(dataset[i][1]) for i in selected_indices]

    # ── operator ───���───────────────────────────────────────────────────
    box_cfg = cfg["box_operator"]
    random_cfg = cfg["random_operator"]
    box_operator = get_operator(
        "inpainting", mask_type=box_cfg["mask_type"],
        mask_len_range=None if box_cfg.get("mask_len_range") is None else tuple(box_cfg["mask_len_range"]),
        mask_prob_range=None if box_cfg.get("mask_prob_range") is None else tuple(box_cfg["mask_prob_range"]),
        resolution=resolution, device=device, sigma=sigma_n,
    )
    random_operator = get_operator(
        "inpainting", mask_type=random_cfg["mask_type"],
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

    # ── model ────────────────────────────────────────��─────────────────
    config = OmegaConf.load(os.path.join(model_dir, "config.yaml"))
    model = config_utils.instantiate_from_config(config.model).to(device)
    print(f"Num of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    ckpt = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt, strict=True)
    model.eval()

    scheduler = PixelFlowScheduler(
        config.scheduler.num_train_timesteps,
        num_stages=config.scheduler.num_stages, gamma=-1 / 3,
    )
    if num_stages != int(config.scheduler.num_stages):
        raise ValueError(f"num_stages={num_stages} != config {config.scheduler.num_stages}")
    scheduler_copy = copy.deepcopy(scheduler)

    # ── measurement ────────────────────────────────────────────────────
    if measurement_mode == "call":
        y = active_operator(gt).detach()
    elif measurement_mode == "measure":
        y = active_operator.measure(gt).detach()
    else:
        raise ValueError(f"measurement_mode must be 'call' or 'measure', got {measurement_mode!r}")

    # ── trajectory tracking ────────────────────────────────────────────
    xts_traj = []
    xt_next_traj = []
    x1_traj_before = []
    x1_traj_after = []
    eps_traj = []
    langevin_inner_traj = []
    langevin_inner_logs = []

    # ── CFG setup ─────���────────────────────────────────────────────────
    # unconditional token = num_classes (matches LabelEmbedding training convention)
    uncond_label = int(model.num_classes)
    prompt_class_label = uncond_label if class_label is None else int(class_label)
    prompt_embeds = torch.tensor([prompt_class_label] * num_examples, dtype=torch.int32, device=device)
    negative_prompt_embeds = uncond_label * torch.ones_like(prompt_embeds)
    do_cfg = guidance_scale > 0
    if do_cfg:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

    # ── PRINCIPLE init ─────────────────────────────────────────────────
    init_factor = 2 ** (num_stages - 1)
    height = resolution // init_factor
    width = resolution // init_factor
    shape = (num_examples, 3, height, width)

    x1_k = torch.randn(shape, device=device, dtype=torch.float32)
    eps_k = randn_tensor(shape, device=device, dtype=torch.float32)
    # latent_tau = x_tau, tracks the interpolated state across stages (for renoise)
    latent_tau = eps_k.clone()  # stage 0, tau=0, start_t=0 → x_tau = eps

    # ── main loop ──────────────────────────────────────────────────────
    for stage_idx in range(num_stages):
        stage_inference_steps = get_stage_inference_steps(inference_each_step, stage_idx, num_stages)
        scheduler_copy.set_timesteps(stage_inference_steps, stage_idx, device=device, shift=shift)
        Timesteps_k = scheduler_copy.Timesteps
        start_t = scheduler_copy.start_t[stage_idx]
        end_t = scheduler_copy.end_t[stage_idx]

        if stage_idx > 0:
            height *= 2
            width *= 2

            # PixelFlow renoise: upsample latent_tau, mix with block noise
            latent_tau = F.interpolate(latent_tau, size=(height, width), mode="nearest")
            original_start_t = scheduler_copy.original_start_t[stage_idx]
            gamma = scheduler_copy.gamma
            alpha = 1 / (math.sqrt(1 - (1 / gamma)) * (1 - original_start_t) + original_start_t)
            beta = alpha * (1 - original_start_t) / math.sqrt(-gamma)
            noise = sample_block_noise(scheduler_copy, num_examples, 3, height, width).to(device=device, dtype=latent_tau.dtype)
            latent_tau = alpha * latent_tau + beta * noise

            # Decompose renoised latent_tau → (x1, eps) at tau=0 of new stage
            # x_tau = start_t * G(x1) + (1-start_t) * eps
            x1_k = F.interpolate(x1_k, size=(height, width), mode="nearest")
            s0 = float(start_t)
            eps_k = (latent_tau - s0 * apply_G(x1_k)) / max(1.0 - s0, 1e-8)

        stage_shape = (num_examples, 3, height, width)
        A_k_fn, AT_k_fn = make_Ak_fns(active_operator, y, stage_shape, device)
        rho_s_k = float(rho_s_cfg[stage_idx]) if isinstance(rho_s_cfg, list) else float(rho_s_cfg)
        rho_e_k = float(rho_e_cfg[stage_idx]) if isinstance(rho_e_cfg, list) else float(rho_e_cfg)

        size_tensor = torch.tensor([height // model.patch_size], dtype=torch.int32, device=device)
        pos_embed = get_2d_rotary_pos_embed(
            embed_dim=model.attention_head_dim,
            crops_coords=((0, 0), (height // model.patch_size, width // model.patch_size)),
            grid_size=(height // model.patch_size, width // model.patch_size),
            device=device, output_type="pt",
        )
        rope_pos = torch.stack(pos_embed, -1)

        for step_idx, T in enumerate(Timesteps_k):
            t_curr = scheduler_copy.t[step_idx].to(device=device, dtype=torch.float32)
            sigma_t = compute_sigma_tau(float(t_curr), float(start_t), float(end_t))
            x_tau_k = apply_H_tau(x1_k, float(t_curr), float(start_t), float(end_t)) + sigma_t * eps_k

            xts_traj.append(x_tau_k.detach().clone())
            x1_traj_before.append(x1_k.detach().clone())

            velocity_fn = make_velocity_fn(
                model, T, prompt_embeds, size_tensor, rope_pos,
                do_cfg, guidance_scale, stage_idx,
            )

            # Skip Langevin when sigma_tau < 0.01: score 1/sigma² explodes,
            # and at tau≈1 the flow model already gives a good x1 estimate.
            if sigma_t < 0.01:
                inner_traj, inner_logs = [], []
            else:
                x1_k, eps_k, inner_traj, inner_logs = principle_langevin_sample(
                    x1_init=x1_k, eps_init=eps_k,
                    tau=float(t_curr), s_k=float(start_t), e_k=float(end_t),
                    velocity_fn=velocity_fn, A_k_fn=A_k_fn, AT_k_fn=AT_k_fn,
                    y=y, sigma_n=sigma_n, h_x=h_x, h_epsilon=h_epsilon,
                    lambda_x=lambda_x, lambda_reg=lambda_reg,
                    rho_s=rho_s_k, rho_e=rho_e_k,
                    cg_tol=cg_tol, cg_max_iter=cg_max_iter,
                    num_Langevin=num_langevin, device=device,
                    return_traj=True, record_every=record_every,
                )

            x1_traj_after.append(x1_k.detach().clone())
            eps_traj.append(eps_k.detach().clone())
            latent_tau = apply_H_tau(x1_k, float(t_curr), float(start_t), float(end_t)) + sigma_t * eps_k
            xt_next_traj.append(latent_tau.detach().clone())
            langevin_inner_traj.append(inner_traj)
            langevin_inner_logs.append(inner_logs)

    # ── saving ────────────────���────────────────────────────────────────
    save_dir = os.path.dirname(dict_path)
    os.makedirs(save_dir, exist_ok=True)
    shutil.copy2(config_path, config_copy_path)

    if save_dict_to_pt:
        torch.save({
            "config_path": config_path,
            "run_config": cfg,
            "exp_name": exp_name,
            "selected_indices": selected_indices,
            "labels": labels,
            "gt": gt.detach().cpu(),
            "y": y.detach().cpu(),
            "xts_traj": xts_traj,
            "xt_next_traj": xt_next_traj,
            "x1_traj_before": x1_traj_before,
            "x1_traj_after": x1_traj_after,
            "eps_traj": eps_traj,
            "langevin_inner_traj": langevin_inner_traj,
        }, dict_path)
        print(f"saved to {dict_path}")

    logs_csv_path = save_langevin_logs_csv(
        output_dir=save_dir, exp_name=exp_name,
        latent_update_mode=latent_update_mode,
        langevin_inner_logs_vpred=langevin_inner_logs,
        langevin_inner_logs_exppred=langevin_inner_logs,
    )
    print(f"saved langevin logs csv to {logs_csv_path}")

    if save_videos:
        # PRINCIPLE column titles (13 columns to match video layout)
        # Slots: y | x_tau | x_tau_next | x1_before | x1_after | xs_hat | xe_hat
        #        | (dup x_tau) | (dup x_tau_next) | (dup x1_before) | (dup x1_after) | eps_k | (dup x_tau_next)
        principle_col_titles = [
            r"$y$",
            r"$x_\tau^k$",
            r"$x_{\tau,\mathrm{next}}^k$",
            r"$\hat{x}_1^k\ \mathrm{before}$",
            r"$\hat{x}_1^k\ \mathrm{after}$",
            r"$\hat{x}_s^k$",
            r"$\hat{x}_e^k$",
            r"$x_\tau^k$",
            r"$x_{\tau,\mathrm{next}}^k$",
            r"$\hat{x}_1^k\ \mathrm{before}$",
            r"$\hat{x}_1^k\ \mathrm{after}$",
            r"$\epsilon^k$",
            r"$x_{\tau,\mathrm{next}}^k$",
        ]
        combined_video_path, langevin_video_path = save_posterior_sampling_videos(
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
            col_titles_override=principle_col_titles,
        )
        print(f"saved combined video to {combined_video_path}")
        print(f"saved langevin inner video to {langevin_video_path}")


if __name__ == "__main__":
    default_config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ms_posterior_sampling_article_version.json",
    )
    parser = argparse.ArgumentParser(description="PRINCIPLE posterior sampling")
    parser.add_argument("--config", type=str, default=default_config_path)
    args = parser.parse_args()
    main(args.config)
