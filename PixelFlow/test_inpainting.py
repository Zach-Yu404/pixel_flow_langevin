#!/usr/bin/env python
# coding: utf-8

import argparse
import copy
import json
import math
import os
import shutil
import time
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from einops import rearrange
from matplotlib.animation import FFMpegWriter
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm

from inpaintingStart import get_operator
from pixelflow.scheduling_pixelflow import PixelFlowScheduler
from pixelflow.utils import config as config_utils
from pixelflow.utils.misc import seed_everything
from sampler2 import PixelFlowPipeline3


def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def sample_block_noise(scheduler, bs, ch, height, width, eps=1e-6):
    gamma = scheduler.gamma
    cov = torch.eye(4) * (1 - gamma) + torch.ones(4, 4) * gamma + eps * torch.eye(4)
    dist = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(4), cov)
    block_number = bs * ch * (height // 2) * (width // 2)
    noise = torch.stack([dist.sample() for _ in range(block_number)])
    noise = rearrange(
        noise,
        "(b c h w) (p q) -> b c (h p) (w q)",
        b=bs,
        c=ch,
        h=height // 2,
        w=width // 2,
        p=2,
        q=2,
    )
    return noise


def T_to_t_linear(Timesteps, t, T):
    T_end = Timesteps[-1]
    T_start = Timesteps[0]
    t_end = t[-2]
    t_start = t[0]
    k = (T_end - T_start) / (t_end - t_start)
    b = T_start - t_start * k
    return (T - b) / k


def get_xt_from_x0(
    scheduler,
    Timesteps,
    x0,
    stage_idx,
    T,
    num_stages,
    device,
    stage_noise=None,
    last_stage_xt=None,
    renoise=False,
):
    # last_stage_xt = None -> interpolation only.
    # last_stage_xt = previous stage endpoint -> stage-start renoise.
    if stage_idx > 0 and T == scheduler.Timesteps_per_stage[stage_idx][0] and renoise:
        print(f"New stage renoising! T = {T}, stage = {stage_idx}.")
        assert last_stage_xt is not None, f"You need x_e_{stage_idx} to get x_s_{stage_idx + 1}"

        xt = F.interpolate(last_stage_xt, scale_factor=2, mode="nearest")
        original_start_t = scheduler.original_start_t[stage_idx]
        gamma = scheduler.gamma
        alpha = 1 / (math.sqrt(1 - (1 / gamma)) * (1 - original_start_t) + original_start_t)
        beta = alpha * (1 - original_start_t) / math.sqrt(-gamma)

        noise = sample_block_noise(scheduler, *xt.shape).to(device=device, dtype=xt.dtype)
        xt = alpha * xt + beta * noise
        return xt

    start_t, end_t = scheduler.start_t[stage_idx], scheduler.end_t[stage_idx]
    print(f"Start T: {start_t}")

    resolution = x0.shape[-1]
    pixel_values_end = x0
    pixel_values_start = x0

    if stage_idx < num_stages - 1:
        for downsample_idx in range(1, num_stages - stage_idx):
            size = resolution // (2 ** downsample_idx)
            pixel_values_end = F.interpolate(pixel_values_end, (size, size), mode="bilinear")

    for downsample_idx in range(1, num_stages - stage_idx + 1):
        size = resolution // (2 ** downsample_idx)
        pixel_values_start = F.interpolate(pixel_values_start, (size, size), mode="bilinear")

    end_height, end_width = pixel_values_end.shape[-2], pixel_values_end.shape[-1]
    pixel_values_start = F.interpolate(pixel_values_start, (end_height, end_width), mode="nearest")

    if stage_noise is None:
        noise = torch.randn_like(pixel_values_end)
    else:
        noise = stage_noise

    pixel_values_end = end_t * pixel_values_end + (1.0 - end_t) * noise
    pixel_values_start = start_t * pixel_values_start + (1.0 - start_t) * noise

    t_select = T_to_t_linear(Timesteps, scheduler.t, T).to(device)
    xt = t_select.float() * pixel_values_end.to(device) + (1.0 - t_select.float()) * pixel_values_start.to(device)
    return xt


def get_stage_noise(coarse_noise):
    coarse = F.interpolate(coarse_noise, scale_factor=2, mode="nearest") / 2.0
    addition = torch.randn_like(coarse)

    block_mean = F.avg_pool2d(addition, kernel_size=2, stride=2)
    block_mean_up = F.interpolate(block_mean, scale_factor=2, mode="nearest")
    noise_orth = addition - block_mean_up

    stage_noise = coarse + noise_orth
    return stage_noise


@torch.no_grad()
def langevin_update(
    x,
    operator,
    y,
    x0hat,
    sigma_n,
    T,
    step_size,
    device,
):
    x = x.to(device)
    y = y.to(device)

    def nu_t(T_cur):
        nt = T_cur / 1000
        return (1 - nt) / math.sqrt(nt ** 2 + (1 - nt) ** 2)

    with torch.enable_grad():
        x = x.detach().requires_grad_(True)
        grad_data_sq = operator.gradient(x, y)
        grad_loss = (x - x0hat) ** 2 / (2 * nu_t(T) ** 2 + 1e-4)
        grad_prior = torch.autograd.grad(grad_loss.sum(), x)[0]

    grad_data = grad_data_sq / (2.0 * (sigma_n ** 2))
    grad_L = grad_data + grad_prior

    eps = torch.randn_like(x)
    x_new = x - step_size * grad_L + math.sqrt(2.0 * step_size) * eps
    return x_new


def langevin_sample(
    x_init,
    operator,
    y,
    x0hat,
    sigma_n,
    T,
    step_size=1e-4,
    num_steps=50,
    device="cpu",
    proj=False,
):
    if proj:
        o_M = operator.mask.to(device)
    else:
        o_M = torch.zeros_like(y, device=device)

    x = x_init.clone().detach()
    for _ in range(num_steps):
        x = (1 - o_M) * langevin_update(x, operator, y, x0hat, sigma_n, T, step_size, device) + o_M * y
        if torch.isnan(x).any():
            break
    return x.detach()


def get_lr(ratio, lr_base=1e-4, lr_min_ratio=0.01):
    p = 1
    multiplier = (1 ** (1 / p) + ratio * (lr_min_ratio ** (1 / p) - 1 ** (1 / p))) ** p
    return multiplier * lr_base


def inverseSample(
    y,
    operator,
    num_stages,
    inference_each_step,
    ode_steps_per_stage,
    langevin_steps,
    Langevin_proj,
    resolution,
    device,
    scheduler,
    sampler_pipeline,
    class_label,
    num_examples,
    lr_base,
    lr_min_ratio,
    sigma_n,
    shift,
    return_traj=True,
    fix_stage_noise=False,
    renoise=False,
    guidance=0.0,
):
    x0_hats_before_Langevin = []
    x0_hats_after_Langevin = []
    xts = []

    for stage_idx in range(num_stages):
        initial_factor = 2 ** (num_stages - 1)
        h, w = resolution // initial_factor, resolution // initial_factor

        if isinstance(inference_each_step, list):
            scheduler.set_timesteps(inference_each_step[stage_idx], stage_idx, device=device, shift=shift)
        else:
            scheduler.set_timesteps(inference_each_step, stage_idx, device=device, shift=shift)

        Timesteps = scheduler.Timesteps

        if fix_stage_noise:
            if stage_idx == 0:
                stage_noise = torch.randn(num_examples, 3, h, w).float().to(device)
                x0_hat = y.detach().clone()
            else:
                stage_noise = get_stage_noise(stage_noise).to(device)
        else:
            if stage_idx == 0:
                x0_hat = y.detach().clone()
            stage_noise = None

        pbar = tqdm(Timesteps.float(), desc=f"Stage: {stage_idx}")
        for T in pbar:
            if renoise and stage_idx > 0:
                xt = get_xt_from_x0(
                    scheduler,
                    Timesteps,
                    x0_hat,
                    stage_idx,
                    T,
                    num_stages,
                    device,
                    stage_noise=stage_noise,
                    last_stage_xt=xt,
                    renoise=renoise,
                )
            else:
                xt = get_xt_from_x0(
                    scheduler,
                    Timesteps,
                    x0_hat,
                    stage_idx,
                    T,
                    num_stages,
                    device,
                    stage_noise=stage_noise,
                )
            xts.append(xt.cpu().float().numpy())

            with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
                x0_hat = sampler_pipeline(
                    prompt=[class_label] * num_examples,
                    height=resolution,
                    width=resolution,
                    num_inference_steps=ode_steps_per_stage,
                    guidance_scale=guidance,
                    num_images_per_prompt=1,
                    device=device,
                    shift=shift,
                    use_ode_dopri5=False,
                    xt=xt,
                    start_stage=stage_idx,
                    start_T=T,
                    noises=None,
                    normalized_x0_hat=False,
                    verbose=False,
                )

            x0_hats_before_Langevin.append(x0_hat.cpu().float().numpy())
            ratio = T / 1000
            lr = get_lr(ratio, lr_base, lr_min_ratio)
            x0_hat_last = copy.deepcopy(x0_hat)
            x0_hat = langevin_sample(
                x0_hat_last,
                operator,
                y,
                x0_hat,
                sigma_n,
                T,
                step_size=lr,
                num_steps=langevin_steps,
                device=device,
                proj=Langevin_proj,
            )
            x0_hats_after_Langevin.append(x0_hat.cpu().float().numpy())

    if return_traj:
        return xts, x0_hats_before_Langevin, x0_hats_after_Langevin
    return None


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().float().cpu().numpy()
    return np.asarray(x)


def to_display_image(x, batch_idx=0):
    x = to_numpy(x)

    if x.ndim == 4:
        if not (0 <= batch_idx < x.shape[0]):
            raise IndexError(f"batch_idx={batch_idx} out of range for batch size {x.shape[0]}")
        x = x[batch_idx]

    if x.ndim == 3 and x.shape[0] in (1, 3):
        x = np.transpose(x, (1, 2, 0))

    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]

    return x


def normalize_for_display(img):
    samples = (img / 2 + 0.5).clip(0, 1)
    return samples


def to_scalar(x):
    if isinstance(x, torch.Tensor):
        return float(x.detach().item())
    return float(x)


def format_T(x, int_digits=4, frac_digits=3):
    width = int_digits + 1 + frac_digits
    return f"{float(x):0{width}.{frac_digits}f}"


def mse_mae(x_img, y_img):
    x = x_img.astype(np.float32)
    y = y_img.astype(np.float32)
    d = x - y
    mae = float(np.mean(np.abs(d)))
    mse = float(np.mean(d * d))
    return mse, mae


def build_T_and_stage_list(scheduler, num_stages, inference_each_step, shift, expected_len=None):
    T_list, stage_list = [], []
    for s in range(num_stages):
        if isinstance(inference_each_step, list):
            scheduler.set_timesteps(inference_each_step[s], s, shift=shift)
        else:
            scheduler.set_timesteps(inference_each_step, s, shift=shift)

        Timesteps = scheduler.Timesteps
        for T in Timesteps:
            T_list.append(float(T.item() if isinstance(T, torch.Tensor) else T))
            stage_list.append(int(s))

    if expected_len is not None:
        L = min(expected_len, len(T_list))
        return T_list[:L], stage_list[:L]
    return T_list, stage_list


def save_traj_video(
    traj_before,
    traj_after,
    traj_xt,
    gt,
    T_list,
    plot_ids,
    mp4_path,
    fps=6,
):
    traj_list = [traj_before, traj_after, traj_xt]
    row_names = ["Before Langevin", "After Langevin", "xt", "GT"]

    min_len = min(len(t) for t in traj_list)
    min_len = min(min_len, len(T_list))
    traj_list = [t[:min_len] for t in traj_list]
    T_list = T_list[:min_len]

    gt_imgs = [to_display_image(gt, batch_idx=pid) for pid in plot_ids]
    gt_disps = [normalize_for_display(im) for im in gt_imgs]

    ncols = len(plot_ids)
    nrows = 4
    fig_w = 4.2 * ncols
    fig_h = 9.5
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h))

    if ncols == 1:
        axes = axes.reshape(nrows, 1)

    plt.subplots_adjust(wspace=0.02, hspace=0.06, top=0.93)
    writer = FFMpegWriter(fps=fps)

    with writer.saving(fig, mp4_path, dpi=150):
        for i in range(min_len):
            T_val = to_scalar(T_list[i])
            T_str = format_T(T_val, int_digits=4, frac_digits=3)
            fig.suptitle(f"Frame {i} | T = {T_str}", fontsize=12)

            for c, pid in enumerate(plot_ids):
                target_img = gt_imgs[c]
                target_disp = gt_disps[c]

                for r in range(3):
                    ax = axes[r, c]
                    ax.clear()

                    frame = traj_list[r][i]
                    x_img = to_display_image(frame, batch_idx=pid)
                    x_disp = normalize_for_display(x_img)

                    if x_disp.ndim == 2:
                        ax.imshow(x_disp, cmap="gray", vmin=0, vmax=1)
                    else:
                        ax.imshow(x_disp)

                    try:
                        mse_v, mae_v = mse_mae(x_img, target_img)
                        ax.set_title(f"pid={pid} | {row_names[r]}\nMSE={mse_v:.5f} MAE={mae_v:.5f}", fontsize=9)
                    except Exception:
                        ax.set_title(f"pid={pid} | {row_names[r]}", fontsize=9)
                    ax.axis("off")

                ax = axes[3, c]
                ax.clear()
                if target_disp.ndim == 2:
                    ax.imshow(target_disp, cmap="gray", vmin=0, vmax=1)
                else:
                    ax.imshow(target_disp)
                ax.set_title(f"pid={pid} | GT", fontsize=9)
                ax.axis("off")

            writer.grab_frame()

    plt.close(fig)


def load_run_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    required_scalar_keys = [
        "data_dir",
        "model_dir",
        "output_root_dir",
        "device",
        "resolution",
        "num_stages",
        "class_label",
        "num_examples",
        "sigma_n",
        "shift",
        "fix_stage_noise",
        "guidance",
        "seed",
        "fps",
        "operator_name",
        "operator_mask_type",
        "scheduler_gamma",
        "max_token_length",
    ]
    required_list_keys = [
        "plot_ids",
        "operator_mask_len_range",
        "num_inference_steps",
        "num_ode_steps",
        "renoise",
        "langevin_proj",
        "num_langevin_steps",
        "lr_base",
        "lr_min_ratio",
    ]
    missing_keys = [k for k in required_scalar_keys + required_list_keys if k not in cfg]
    if missing_keys:
        raise KeyError(f"Missing keys in experiment config: {missing_keys}")

    for key in required_list_keys:
        if not isinstance(cfg[key], list) or len(cfg[key]) == 0:
            raise ValueError(f"'{key}' must be a non-empty list in {config_path}")

    if not all(isinstance(v, bool) for v in cfg["renoise"]):
        raise TypeError("'renoise' must be a list of booleans")
    if not all(isinstance(v, bool) for v in cfg["langevin_proj"]):
        raise TypeError("'langevin_proj' must be a list of booleans")

    if not isinstance(cfg["plot_ids"], list) or len(cfg["plot_ids"]) == 0:
        raise ValueError("'plot_ids' must be a non-empty list")
    if len(cfg["operator_mask_len_range"]) != 2:
        raise ValueError("'operator_mask_len_range' must have length 2")

    return cfg


def make_experiment_folder(output_root_dir, experiment_name):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_name = f"{experiment_name}_{timestamp}" if experiment_name else f"experiment_{timestamp}"
    experiment_dir = os.path.join(output_root_dir, base_name)
    suffix = 1
    while os.path.exists(experiment_dir):
        experiment_dir = os.path.join(output_root_dir, f"{base_name}_{suffix:02d}")
        suffix += 1
    os.makedirs(experiment_dir, exist_ok=False)
    os.makedirs(os.path.join(experiment_dir, "videos"), exist_ok=False)
    return experiment_dir


def main(run_config_path="./test_inpainting_experiments.sample.json"):
    run_config_path = os.path.abspath(run_config_path)
    exp_cfg = load_run_config(run_config_path)

    data_dir = os.path.expanduser(exp_cfg["data_dir"])
    model_dir = os.path.expanduser(exp_cfg["model_dir"])
    output_root_dir = os.path.expanduser(exp_cfg["output_root_dir"])
    device = torch.device(exp_cfg["device"])

    resolution = int(exp_cfg["resolution"])
    num_stages = int(exp_cfg["num_stages"])
    raw_class_label = exp_cfg["class_label"]
    class_label = None if raw_class_label is None else int(raw_class_label)
    num_examples = int(exp_cfg["num_examples"])
    sigma_n = float(exp_cfg["sigma_n"])
    shift = float(exp_cfg["shift"])
    fix_stage_noise = bool(exp_cfg["fix_stage_noise"])
    guidance = float(exp_cfg["guidance"])
    seed = int(exp_cfg["seed"])
    fps = int(exp_cfg["fps"])
    plot_ids = [int(v) for v in exp_cfg["plot_ids"]]
    operator_name = exp_cfg["operator_name"]
    operator_mask_type = exp_cfg["operator_mask_type"]
    operator_mask_len_range = tuple(int(v) for v in exp_cfg["operator_mask_len_range"])
    operator_mask_prob_range = exp_cfg.get("operator_mask_prob_range", None)
    scheduler_gamma = float(exp_cfg["scheduler_gamma"])
    max_token_length = int(exp_cfg["max_token_length"])
    experiment_name = exp_cfg.get("experiment_name", "experiment")

    num_inference_steps_grid = exp_cfg["num_inference_steps"]
    num_ode_steps_grid = exp_cfg["num_ode_steps"]
    renoise_grid = exp_cfg["renoise"]
    langevin_proj_grid = exp_cfg["langevin_proj"]
    num_langevin_steps_grid = exp_cfg["num_langevin_steps"]
    lr_base_grid = exp_cfg["lr_base"]
    lr_min_ratio_grid = exp_cfg["lr_min_ratio"]

    experiment_dir = make_experiment_folder(output_root_dir, experiment_name)
    videos_dir = os.path.join(experiment_dir, "videos")
    config_copy_path = os.path.join(experiment_dir, "run_config.json")
    shutil.copy2(run_config_path, config_copy_path)
    summary_json_path = os.path.join(experiment_dir, "summary.json")

    transform = transforms.Compose(
        [
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, resolution)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )
    dataset = ImageFolder(data_dir, transform=transform)

    seed_everything(seed)
    if class_label is None or class_label == 1000:
        idx = torch.randperm(len(dataset))[:num_examples]
    else:
        candidate_indices = [i for i, y_label in enumerate(dataset.targets) if int(y_label) == class_label]
        if len(candidate_indices) < num_examples:
            raise ValueError(
                f"class_label={class_label} has only {len(candidate_indices)} samples in dataset, "
                f"but num_examples={num_examples}"
            )
        chosen = torch.randperm(len(candidate_indices))[:num_examples].tolist()
        idx = torch.tensor([candidate_indices[i] for i in chosen], dtype=torch.long)
    gt = torch.stack([dataset[i][0] for i in idx], dim=0).to(device)

    A_opr = get_operator(
        operator_name,
        mask_type=operator_mask_type,
        mask_len_range=operator_mask_len_range,
        mask_prob_range=operator_mask_prob_range,
        resolution=resolution,
        device=device,
        sigma=sigma_n,
    )
    y = A_opr.measure(gt).to(device)

    config = OmegaConf.load(f"{model_dir}/config.yaml")
    model = config_utils.instantiate_from_config(config.model).to(device)
    print(f"Num of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    ckpt = torch.load(f"{model_dir}/model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt, strict=True)
    model.eval()

    if num_stages != int(config.scheduler.num_stages):
        raise ValueError(
            f"num_stages in json ({num_stages}) != model config scheduler.num_stages ({config.scheduler.num_stages})"
        )

    scheduler = PixelFlowScheduler(
        config.scheduler.num_train_timesteps,
        num_stages=config.scheduler.num_stages,
        gamma=scheduler_gamma,
    )
    scheduler_copy = copy.deepcopy(scheduler)
    sampler_pipeline = PixelFlowPipeline3(
        scheduler_copy,
        model,
        text_encoder=None,
        tokenizer=None,
        max_token_length=max_token_length,
    )

    def normalize_stage_steps(steps, name):
        if isinstance(steps, int):
            return steps
        if isinstance(steps, (list, tuple)):
            out = [int(v) for v in steps]
            if len(out) != num_stages:
                raise ValueError(
                    f"{name} list length must equal num_stages={num_stages}, got {len(out)}"
                )
            return out
        raise TypeError(f"{name} must be int or list/tuple[int], got {type(steps)}")

    def tag_steps(steps):
        if isinstance(steps, list):
            return "x".join(str(v) for v in steps)
        return str(steps)

    experiments = list(
        product(
            num_inference_steps_grid,
            num_ode_steps_grid,
            renoise_grid,
            langevin_proj_grid,
            num_langevin_steps_grid,
            lr_base_grid,
            lr_min_ratio_grid,
        )
    )
    print(f"Total experiment combos: {len(experiments)}")

    results = []
    for exp_idx, (
        num_inference_steps,
        num_ode_steps,
        renoise,
        langevin_proj,
        num_langevin_steps,
        lr_base,
        lr_min_ratio,
    ) in enumerate(experiments, start=1):
        inference_each_step = normalize_stage_steps(num_inference_steps, "num_inference_steps")
        ode_steps_per_stage = normalize_stage_steps(num_ode_steps, "num_ode_steps")

        seed_everything(seed)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        xts, x0s_woL, x0s_L = inverseSample(
            y=y,
            operator=A_opr,
            num_stages=num_stages,
            inference_each_step=inference_each_step,
            ode_steps_per_stage=ode_steps_per_stage,
            langevin_steps=num_langevin_steps,
            Langevin_proj=langevin_proj,
            resolution=resolution,
            device=device,
            scheduler=scheduler,
            sampler_pipeline=sampler_pipeline,
            class_label=class_label,
            num_examples=num_examples,
            lr_base=lr_base,
            lr_min_ratio=lr_min_ratio,
            sigma_n=sigma_n,
            shift=shift,
            return_traj=True,
            fix_stage_noise=fix_stage_noise,
            renoise=renoise,
            guidance=guidance,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dt = time.perf_counter() - t0

        T_list, _ = build_T_and_stage_list(
            scheduler=scheduler,
            num_stages=num_stages,
            inference_each_step=inference_each_step,
            shift=shift,
            expected_len=len(xts),
        )

        exp_tag = (
            f"exp{exp_idx:03d}"
            f"_inf{tag_steps(inference_each_step)}"
            f"_ode{tag_steps(ode_steps_per_stage)}"
            f"_renoise{int(renoise)}"
            f"_lproj{int(langevin_proj)}"
            f"_lang{num_langevin_steps}"
            f"_lrb{lr_base:g}"
            f"_lrr{lr_min_ratio:g}"
        )
        mp4_path = os.path.join(videos_dir, f"{exp_tag}_Time{dt:.2f}s.mp4")

        save_traj_video(
            traj_before=x0s_woL,
            traj_after=x0s_L,
            traj_xt=xts,
            gt=gt,
            T_list=T_list,
            plot_ids=plot_ids,
            mp4_path=mp4_path,
            fps=fps,
        )

        results.append(
            {
                "exp_id": exp_idx,
                "num_inference_steps": inference_each_step,
                "num_ode_steps": ode_steps_per_stage,
                "renoise": renoise,
                "langevin_proj": langevin_proj,
                "num_langevin_steps": num_langevin_steps,
                "lr_base": lr_base,
                "lr_min_ratio": lr_min_ratio,
                "time_sec": round(dt, 2),
                "video": mp4_path,
            }
        )
        print(
            f"[{exp_idx}/{len(experiments)}] done | "
            f"inf={inference_each_step}, ode={ode_steps_per_stage}, renoise={renoise}, "
            f"lproj={langevin_proj}, langevin={num_langevin_steps}, "
            f"lr_base={lr_base}, lr_min_ratio={lr_min_ratio}, "
            f"time={dt:.2f}s"
        )

    print("\nExperiment summary:")
    for r in results:
        print(
            f"exp{r['exp_id']:03d} | inf={r['num_inference_steps']} | ode={r['num_ode_steps']} | "
            f"renoise={r['renoise']} | lproj={r['langevin_proj']} | "
            f"lang={r['num_langevin_steps']} | "
            f"lr_base={r['lr_base']} | lr_min_ratio={r['lr_min_ratio']} | "
            f"time={r['time_sec']:.2f}s | video={r['video']}"
        )

    summary_payload = {
        "run_config_path": run_config_path,
        "run_config_copy": config_copy_path,
        "experiment_dir": experiment_dir,
        "videos_dir": videos_dir,
        "num_experiments": len(experiments),
        "global_config": exp_cfg,
        "results": results,
    }
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)
    print(f"Experiment folder: {experiment_dir}")
    print(f"Copied run config to: {config_copy_path}")
    print(f"Saved experiment summary JSON to: {summary_json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inpainting experiment grid from JSON config")
    parser.add_argument(
        "--config",
        type=str,
        default="./test_inpainting_experiments.sample.json",
        help="Path to run JSON config",
    )
    args = parser.parse_args()
    main(run_config_path=args.config)
