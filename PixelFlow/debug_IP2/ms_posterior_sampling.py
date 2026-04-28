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
from ms_posterior_sampling_utils import (
    build_experiment_paths,
    center_crop_arr,
    class_guidance_scale,
    get_lr,
    get_stage_inference_steps,
    get_x1_and_noise_with_stage_start_and_end2,
    load_run_config,
    langevin_sample_split_prior,
    pred_x1_x0_with_vt,
    resolve_path,
    save_langevin_logs_csv,
    save_posterior_sampling_videos,
    sample_block_noise,
    update_latents_from_mode,
)
from pixelflow.scheduling_pixelflow import PixelFlowScheduler
from pixelflow.utils import config as config_utils
from pixelflow.utils.misc import seed_everything


def main(config_path):
    config_path = os.path.abspath(config_path)
    script_dir = os.path.dirname(os.path.abspath(__file__))
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
    proj = bool(cfg["proj"])
    lr_base = float(cfg["lr_base"])
    lr_min_ratio = float(cfg["lr_min_ratio"])
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
    latent_update_mode = str(cfg["latent_update_mode"])
    if latent_update_mode not in {"n", "x0", "f_x1", "intp"}:
        raise ValueError(
            f"latent_update_mode must be one of ['n', 'x0', 'f_x1', 'intp'], got {latent_update_mode!r}"
        )
    if dict_root.endswith(".pt"):
        dict_root = os.path.dirname(dict_root)
    dict_path, config_copy_path, exp_tag = build_experiment_paths(dict_root, exp_name, latent_update_mode)
    save_videos = bool(cfg.get("save_videos", True))
    combined_video_fps = int(cfg.get("combined_video_fps", 6))
    langevin_video_fps = int(cfg.get("langevin_video_fps", 8))
    video_dpi = int(cfg.get("video_dpi", 220))
    video_batch_size = int(cfg.get("video_batch_size", num_examples))
    save_inner_latents = bool(cfg.get("save_inner_latents", False))
    save_dict_to_pt = bool(cfg.get("save_dict_to_pt", False))
    lambda_prior = float(cfg.get("lambda_prior", 1.0))

    seed_everything(seed)

    transform = transforms.Compose(
        [
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, resolution)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )
    dataset = ImageFolder(data_dir, transform=transform)

    if class_label is None:
        selected_indices = torch.randperm(len(dataset))[:num_examples].tolist()
    else:
        class_indices = [i for i, (_, y_label) in enumerate(dataset.samples) if int(y_label) == int(class_label)]
        if len(class_indices) < num_examples:
            raise ValueError(
                f"class_label={class_label} has only {len(class_indices)} samples, need {num_examples}"
            )
        perm = torch.randperm(len(class_indices))[:num_examples]
        selected_indices = [class_indices[i] for i in perm.tolist()]

    gt = torch.stack([dataset[i][0] for i in selected_indices], dim=0).to(device)
    labels = [int(dataset[i][1]) for i in selected_indices]

    box_cfg = cfg["box_operator"]
    random_cfg = cfg["random_operator"]
    box_operator = get_operator(
        "inpainting",
        mask_type=box_cfg["mask_type"],
        mask_len_range=None if box_cfg.get("mask_len_range") is None else tuple(box_cfg["mask_len_range"]),
        mask_prob_range=None if box_cfg.get("mask_prob_range") is None else tuple(box_cfg["mask_prob_range"]),
        resolution=resolution,
        device=device,
        sigma=sigma_n,
    )
    random_operator = get_operator(
        "inpainting",
        mask_type=random_cfg["mask_type"],
        mask_len_range=None if random_cfg.get("mask_len_range") is None else tuple(random_cfg["mask_len_range"]),
        mask_prob_range=None if random_cfg.get("mask_prob_range") is None else tuple(random_cfg["mask_prob_range"]),
        resolution=resolution,
        device=device,
        sigma=sigma_n,
    )
    if active_operator_name == "random":
        active_operator = random_operator
    elif active_operator_name == "box":
        active_operator = box_operator
    else:
        raise ValueError(f"active_operator must be 'random' or 'box', got {active_operator_name!r}")

    config = OmegaConf.load(os.path.join(model_dir, "config.yaml"))
    model = config_utils.instantiate_from_config(config.model).to(device)
    print(f"Num of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    ckpt = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt, strict=True)
    model.eval()

    scheduler = PixelFlowScheduler(
        config.scheduler.num_train_timesteps,
        num_stages=config.scheduler.num_stages,
        gamma=-1 / 3,
    )
    if num_stages != int(config.scheduler.num_stages):
        raise ValueError(
            f"num_stages={num_stages} does not match config.scheduler.num_stages={config.scheduler.num_stages}"
        )
    scheduler_copy = copy.deepcopy(scheduler)

    if measurement_mode == "call":
        y = active_operator(gt).detach()
    elif measurement_mode == "measure":
        y = active_operator.measure(gt).detach()
    else:
        raise ValueError(f"measurement_mode must be 'call' or 'measure', got {measurement_mode!r}")

    xts_vpred_traj = []
    xt_next_vpred_traj = []
    xs_vpred_traj = []
    xe_vpred_traj = []
    x0_vpred_traj = []
    x1_vpred_traj_before_langevin = []
    x1_vpred_traj_after_langevin = []

    xts_exppred_traj = []
    xt_next_exppred_traj = []
    xs_exppred_traj = []
    xe_exppred_traj = []
    x0_exppred_traj = []
    x1_exppred_traj_before_langevin = []
    x1_exppred_traj_after_langevin = []

    langevin_inner_traj_vpred = []
    langevin_inner_traj_exppred = []
    langevin_inner_logs_vpred = []
    langevin_inner_logs_exppred = []

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
            crops_coords=(
                (0, 0),
                (latents_v.shape[-1] // model.patch_size, latents_v.shape[-1] // model.patch_size),
            ),
            grid_size=(latents_v.shape[-1] // model.patch_size, latents_v.shape[-1] // model.patch_size),
            device=device,
            output_type="pt",
        )
        rope_pos = torch.stack(pos_embed, -1)
        ####### Adjust projection stage:
        # if stage_idx < 1:
        #     proj = proj
        # else:
        #     proj = False
        # print("proj:", proj)

        for step_idx, T in enumerate(Timesteps_k):
            xts_vpred_traj.append(latents_v.clone())
            xts_exppred_traj.append(latents_stage.clone())

            latent_model_input_v = torch.cat([latents_v] * 2) if do_classifier_free_guidance else latents_v
            latent_model_input_stage = torch.cat([latents_stage] * 2) if do_classifier_free_guidance else latents_stage

            timestep_v = T.expand(latent_model_input_v.shape[0]).to(latent_model_input_v.dtype)
            timestep_stage = T.expand(latent_model_input_stage.shape[0]).to(latent_model_input_stage.dtype)

            with torch.no_grad():
                noise_pred_v = model(
                    latent_model_input_v,
                    timestep=timestep_v,
                    class_labels=prompt_embeds,
                    latent_size=size_tensor,
                    pos_embed=rope_pos,
                )
                noise_pred_stage = model(
                    latent_model_input_stage,
                    timestep=timestep_stage,
                    class_labels=prompt_embeds,
                    latent_size=size_tensor,
                    pos_embed=rope_pos,
                )

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

            xs_vpred_traj.append(pixel_values_start_v.clone())
            xe_vpred_traj.append(pixel_values_end_v.clone())
            xs_exppred_traj.append(pixel_values_start_stage.clone())
            xe_exppred_traj.append(pixel_values_end_stage.clone())

            x1_stagePred, x0_stagePred, _, _ = get_x1_and_noise_with_stage_start_and_end2(
                pixel_values_end_stage,
                pixel_values_start_stage,
                scheduler_copy,
                stage_idx,
            )
            x1_vPred, x0_vPred = pred_x1_x0_with_vt(latents_v, noise_pred_v, T, start_t, end_t)

            x1_vpred_traj_before_langevin.append(x1_vPred.clone())
            x0_vpred_traj.append(x0_vPred.clone())
            x1_exppred_traj_before_langevin.append(x1_stagePred.clone())
            x0_exppred_traj.append(x0_stagePred.clone())

            ratio = float(T.detach().item()) / 1000.0
            lr = float(get_lr(ratio, lr_base=lr_base, lr_min_ratio=lr_min_ratio))
            # if t_curr ==0:
            #     x1_vPred = x1_vPred; x1_stagePred = x1_stagePred
            # else:
            #     x1_vPred = x1_vPred_after_langevin.clone(); x1_stagePred = x1_stagePred_after_langevin.clone()
            x1_vPred_after_langevin, inner_traj_v, inner_logs_v = langevin_sample_split_prior(
                x_init=x1_vPred,
                y=y,
                operator=active_operator,
                stage_t_end=end_t,
                stage_t_start=start_t,
                stage_x_end=pixel_values_end_v.detach(),
                stage_x_start=pixel_values_start_v.detach(),
                num_Langevin=num_langevin,
                sigma_n=sigma_n,
                lambda_prior=lambda_prior,
                step_size=lr,
                device=device,
                proj=proj,
                return_traj=True,
                record_every=record_every,
            )

            x1_stagePred_after_langevin, inner_traj_exp, inner_logs_exp = langevin_sample_split_prior(
                x_init=x1_stagePred,
                y=y,
                operator=active_operator,
                stage_t_end=end_t,
                stage_t_start=start_t,
                stage_x_end=pixel_values_end_stage.detach(),
                stage_x_start=pixel_values_start_stage.detach(),
                num_Langevin=num_langevin,
                sigma_n=sigma_n,
                lambda_prior=lambda_prior,
                step_size=lr,
                device=device,
                proj=proj,
                return_traj=True,
                record_every=record_every,
            )

            x1_vpred_traj_after_langevin.append(x1_vPred_after_langevin.clone())
            x1_exppred_traj_after_langevin.append(x1_stagePred_after_langevin.clone())
            langevin_inner_traj_vpred.append(inner_traj_v)
            langevin_inner_logs_vpred.append(inner_logs_v)
            langevin_inner_traj_exppred.append(inner_traj_exp)
            langevin_inner_logs_exppred.append(inner_logs_exp)

            if latent_update_mode == "intp":
                x0_stagePred = math.sqrt(t_next)* x0_stagePred + math.sqrt(1 - t_next) * torch.randn_like(x0_stagePred)
                x0_vPred = math.sqrt(t_next)* x0_vPred + math.sqrt(1 - t_next) * torch.randn_like(x0_vPred)

            latents_stage, latents_v = update_latents_from_mode(
                latent_update_mode=latent_update_mode,
                scheduler=scheduler_copy,
                x1_stage=x1_stagePred_after_langevin,
                x1_v=x1_vPred_after_langevin,
                stage_idx=stage_idx,
                step_idx=step_idx + 1,
                device=device,
                x0_stage=x0_stagePred,
                x0_v=x0_vPred,
                pixel_values_start_stage=pixel_values_start_stage,
                pixel_values_start_v=pixel_values_start_v,
            )

            xt_next_vpred_traj.append(latents_v.clone())
            xt_next_exppred_traj.append(latents_stage.clone())

    save_dir = os.path.dirname(dict_path)
    os.makedirs(save_dir, exist_ok=True)
    shutil.copy2(config_path, config_copy_path)

    if save_dict_to_pt:
        save_dict = {
            "config_path": config_path,
            "config_copy_path": config_copy_path,
            "run_config": cfg,
            "exp_name": exp_name,
            "exp_tag": exp_tag,
            "latent_update_mode": latent_update_mode,
            "selected_indices": selected_indices,
            "labels": labels,
            "gt": gt.detach().cpu(),
            "y": y.detach().cpu(),
            "xts_vpred_traj": xts_vpred_traj,
            "xt_next_vpred_traj": xt_next_vpred_traj,
            "xs_vpred_traj": xs_vpred_traj,
            "xe_vpred_traj": xe_vpred_traj,
            "x0_vpred_traj": x0_vpred_traj,
            "x1_vpred_traj_before_langevin": x1_vpred_traj_before_langevin,
            "x1_vpred_traj_after_langevin": x1_vpred_traj_after_langevin,
            "xts_exppred_traj": xts_exppred_traj,
            "xt_next_exppred_traj": xt_next_exppred_traj,
            "xs_exppred_traj": xs_exppred_traj,
            "xe_exppred_traj": xe_exppred_traj,
            "x0_exppred_traj": x0_exppred_traj,
            "x1_exppred_traj_before_langevin": x1_exppred_traj_before_langevin,
            "x1_exppred_traj_after_langevin": x1_exppred_traj_after_langevin,
            "langevin_inner_traj_vpred": langevin_inner_traj_vpred,
            "langevin_inner_traj_exppred": langevin_inner_traj_exppred,
        }
        torch.save(save_dict, dict_path)
        print(f"saved to {dict_path}")
        print(f"saved run config copy to {config_copy_path}")

    logs_csv_path = save_langevin_logs_csv(
        output_dir=save_dir,
        exp_name=exp_name,
        latent_update_mode=latent_update_mode,
        langevin_inner_logs_vpred=langevin_inner_logs_vpred,
        langevin_inner_logs_exppred=langevin_inner_logs_exppred,
    )
    print(f"saved langevin logs csv to {logs_csv_path}")

    if save_videos:
        combined_video_path, langevin_video_path = save_posterior_sampling_videos(
            output_dir=save_dir,
            exp_name=exp_name,
            latent_update_mode=latent_update_mode,
            y_tensor=y.detach().cpu(),
            xts_vpred_traj=xts_vpred_traj,
            xt_next_vpred_traj=xt_next_vpred_traj,
            x1_vpred_traj_before_langevin=x1_vpred_traj_before_langevin,
            x1_vpred_traj_after_langevin=x1_vpred_traj_after_langevin,
            xs_vpred_traj=xs_vpred_traj,
            xe_vpred_traj=xe_vpred_traj,
            xts_exppred_traj=xts_exppred_traj,
            xt_next_exppred_traj=xt_next_exppred_traj,
            x1_exppred_traj_before_langevin=x1_exppred_traj_before_langevin,
            x1_exppred_traj_after_langevin=x1_exppred_traj_after_langevin,
            xs_exppred_traj=xs_exppred_traj,
            xe_exppred_traj=xe_exppred_traj,
            langevin_inner_traj_vpred=langevin_inner_traj_vpred,
            langevin_inner_traj_exppred=langevin_inner_traj_exppred,
            inference_each_step=inference_each_step,
            num_stages=num_stages,
            batch_size=video_batch_size,
            combined_fps=combined_video_fps,
            langevin_fps=langevin_video_fps,
            dpi=video_dpi,
            save_inner_latents = save_inner_latents
        )
        print(f"saved combined video to {combined_video_path}")
        print(f"saved langevin inner video to {langevin_video_path}")


if __name__ == "__main__":
    default_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ms_posteior_sampling.json")
    parser = argparse.ArgumentParser(description="Run MS posterior sampling from JSON config")
    parser.add_argument(
        "--config",
        type=str,
        default=default_config_path,
        help="Path to ms_posteior_sampling JSON config",
    )
    args = parser.parse_args()
    main(args.config)
