import csv
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from einops import rearrange
from matplotlib.animation import FFMpegWriter


def resolve_path(base_dir, path_value):
    if path_value is None:
        return None
    expanded = os.path.expanduser(path_value)
    if os.path.isabs(expanded):
        return expanded
    return os.path.abspath(os.path.join(base_dir, expanded))


def load_run_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    required_keys = [
        "exp_name",
        "seed",
        "num_stages",
        "resolution",
        "num_examples",
        "sigma_n",
        "class_label",
        "shift",
        "inference_each_step",
        "guidance_scale",
        "num_langevin",
        "proj",
        "lr_base",
        "lr_min_ratio",
        "device",
        "data_dir",
        "model_dir",
        "dict_path",
        "box_operator",
        "random_operator",
        "active_operator",
        "measurement_mode",
        "latent_update_mode",
    ]
    missing = [key for key in required_keys if key not in cfg]
    if missing:
        raise KeyError(f"Missing keys in config: {missing}")

    for operator_key in ("box_operator", "random_operator"):
        operator_cfg = cfg[operator_key]
        if not isinstance(operator_cfg, dict):
            raise TypeError(f"'{operator_key}' must be a dict")
        if "mask_type" not in operator_cfg:
            raise KeyError(f"'{operator_key}.mask_type' is required")

    return cfg


def get_stage_inference_steps(inference_each_step, stage_idx, num_stages):
    if isinstance(inference_each_step, int):
        return inference_each_step
    if isinstance(inference_each_step, list):
        if len(inference_each_step) != num_stages:
            raise ValueError(
                f"inference_each_step list length must equal num_stages={num_stages}, got {len(inference_each_step)}"
            )
        return int(inference_each_step[stage_idx])
    raise TypeError(f"inference_each_step must be int or list[int], got {type(inference_each_step)}")


def build_output_paths(output_root, latent_update_mode):
    output_root = os.path.abspath(output_root)
    result_path = os.path.join(output_root, f"posterior_sampling_mode-{latent_update_mode}.pt")
    config_copy_path = os.path.join(output_root, f"posterior_sampling_mode-{latent_update_mode}.run_config.json")
    return result_path, config_copy_path


def normalize_experiment_name(exp_name):
    exp_name = str(exp_name).strip()
    if not exp_name:
        return "posterior_sampling"
    safe = []
    for ch in exp_name:
        if ch.isalnum() or ch in ("-", "_"):
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)


def build_experiment_paths(output_root, exp_name, latent_update_mode):
    output_root = os.path.abspath(output_root)
    exp_tag = normalize_experiment_name(exp_name)
    exp_dir = os.path.join(output_root, exp_tag)
    result_path = os.path.join(exp_dir, f"{exp_tag}_mode-{latent_update_mode}.pt")
    config_copy_path = os.path.join(exp_dir, f"{exp_tag}_mode-{latent_update_mode}.run_config.json")
    return result_path, config_copy_path, exp_tag


def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size),
            resample=Image.BOX,
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size),
        resample=Image.BICUBIC,
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


def Up_k(x, y):
    target_h, target_w = y.shape[-2:]
    if x.shape[-2:] == (target_h, target_w):
        return x

    out = x
    h, w = x.shape[-2:]
    while h < target_h or w < target_w:
        next_h = min(h * 2, target_h)
        next_w = min(w * 2, target_w)
        out = F.interpolate(out, size=(next_h, next_w), mode="bilinear", align_corners=False)
        h, w = next_h, next_w

    if (h, w) != (target_h, target_w):
        out = F.interpolate(out, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return out

def Dn_k(x, y, mode = "bilinear"):
    target_h, target_w = y.shape[-2:]
    if x.shape[-2:] == (target_h, target_w):
        return x

    out = x
    h, w = x.shape[-2:]

    while h > target_h or w > target_w:
        next_h = max(h // 2, target_h)
        next_w = max(w // 2, target_w)
        out = F.interpolate(out, size=(next_h, next_w), mode=mode)
        h, w = next_h, next_w

    if (h, w) != (target_h, target_w):
        out = F.interpolate(out, size=(target_h, target_w), mode=mode)
    return out


def DownUp_operation(z, scale_factor=2):
    _, _, H, W = z.shape
    h_small = H // scale_factor
    w_small = W // scale_factor
    z_down = F.interpolate(z, size=(h_small, w_small), mode="bilinear", align_corners=False)
    z_up = F.interpolate(z_down, size=(H, W), mode="nearest")
    return z_up


def get_xt_from_x1_sameStage(scheduler, D_x1, stage_idx, step_idx, device, DU_x0=None, noise=None):
    start_t = scheduler.start_t[stage_idx]
    end_t = scheduler.end_t[stage_idx]
    if DU_x0 is None:
        if noise is None:
            noise = torch.randn_like(D_x1)
        pixel_values_end = end_t * D_x1 + (1.0 - end_t) * noise
        pixel_values_start = start_t * DownUp_operation(D_x1) + (1.0 - start_t) * noise
    else:
        pixel_values_start = DU_x0.clone()
        pixel_values_end = end_t * D_x1 + (1.0 - end_t) / (1.0 - start_t) * (
            pixel_values_start - start_t * DownUp_operation(D_x1)
        )

    t_step = scheduler.t[step_idx].float().to(device=device, dtype=D_x1.dtype)
    xt = t_step * pixel_values_end.to(device) + (1.0 - t_step) * pixel_values_start.to(device)
    return xt


def update_latents_from_mode(
    latent_update_mode,
    scheduler,
    x1_stage,
    x1_v,
    stage_idx,
    step_idx,
    device,
    x0_stage=None,
    x0_v=None,
    pixel_values_start_stage=None,
    pixel_values_start_v=None,
):
    if latent_update_mode == "n":
        latents_stage = get_xt_from_x1_sameStage(
            scheduler,
            x1_stage,
            stage_idx,
            step_idx,
            device=device,
            noise=None,
            DU_x0=None,
        )
        latents_v = get_xt_from_x1_sameStage(
            scheduler,
            x1_v,
            stage_idx,
            step_idx,
            device=device,
            noise=None,
            DU_x0=None,
        )
        return latents_stage, latents_v

    if latent_update_mode == "x0":
        latents_stage = get_xt_from_x1_sameStage(
            scheduler,
            x1_stage,
            stage_idx,
            step_idx,
            device=device,
            noise=x0_stage,
            DU_x0=None,
        )
        latents_v = get_xt_from_x1_sameStage(
            scheduler,
            x1_v,
            stage_idx,
            step_idx,
            device=device,
            noise=x0_v,
            DU_x0=None,
        )
        return latents_stage, latents_v

    if latent_update_mode == "f_x1":
        latents_stage = get_xt_from_x1_sameStage(
            scheduler,
            x1_stage,
            stage_idx,
            step_idx,
            device=device,
            noise=None,
            DU_x0=pixel_values_start_stage,
        )
        latents_v = get_xt_from_x1_sameStage(
            scheduler,
            x1_v,
            stage_idx,
            step_idx,
            device=device,
            noise=None,
            DU_x0=pixel_values_start_v,
        )
        return latents_stage, latents_v
    if latent_update_mode == "intp":
        latents_stage = get_xt_from_x1_sameStage(
            scheduler,
            x1_stage,
            stage_idx,
            step_idx,
            device=device,
            noise=x0_stage,
            DU_x0=None,
        )
        latents_v = get_xt_from_x1_sameStage(
            scheduler,
            x1_v,
            stage_idx,
            step_idx,
            device=device,
            noise=x0_v,
            DU_x0=None,
        )
        return latents_stage, latents_v

    raise ValueError(
        f"latent_update_mode must be one of ['n', 'x0', 'f_x1', 'intp'], got {latent_update_mode!r}"
    )


def pred_x1_x0_with_vt(xt, vt, T, start_t, end_t):
    x1_normal = xt + (1 - T / 1000) / (end_t - start_t) * vt
    x0_normal = xt - T / 1000 * (end_t - start_t) * vt
    return x1_normal, x0_normal


def get_x1_and_noise_with_stage_start_and_end2(xe, xs, scheduler, stage_idx):
    s = scheduler.start_t[stage_idx]
    e = scheduler.end_t[stage_idx]
    x1 = ((1 - s) * xe - (1 - e) * xs) / (e - s)
    x0 = (e * xs - s * xe) / (e - s)
    return x1, x0, e, s


def get_lr(ratio, lr_base=1e-4, lr_min_ratio=1e-2):
    p = 1
    multiplier = (1 ** (1 / p) + ratio * (lr_min_ratio ** (1 / p) - 1 ** (1 / p))) ** p
    return multiplier * lr_base


def class_guidance_scale(base_scale, stage_idx):
    scale_dict = {0: 0.0, 1: 1.0 / 6.0, 2: 2.0 / 3.0, 3: 1.0}
    return (base_scale - 1.0) * scale_dict.get(stage_idx, 1.0) + 1.0


def objective_split_prior(
    x0hat,
    operator,
    stage_t_end,
    stage_t_start,
    stage_x_end,
    stage_x_start,
    y,
    sigma_n,
    o_M,
    lambda_prior=1.0,
):
    stage_x_end = stage_x_end.detach().requires_grad_(False)
    stage_x_start = stage_x_start.detach().requires_grad_(False)
    

    x_up_proj = operator(Up_k(x0hat, y))
    l3 = ((x_up_proj - y) ** 2).sum()

    var_eps = 1e-2
    denom_end = max(float((1 - stage_t_end) ** 2), var_eps)
    denom_start = max(float((1 - stage_t_start) ** 2), var_eps)

    l1 = (sigma_n**2 / (denom_end) * (stage_x_end - stage_t_end * x0hat) ** 2).sum()
    l2 = (sigma_n**2 / (denom_start) * (stage_x_start - stage_t_start * DownUp_operation(x0hat)) ** 2).sum()
    return lambda_prior*(l1 + l2) + l3, l1, l2, l3
    # l3 = (1/(2*sigma_n**2)*(x_up_proj - y) ** 2).sum()

    # var_eps = 1e-2
    # denom_end = max(float((1 - stage_t_end) ** 2), var_eps)
    # denom_start = max(float((1 - stage_t_start) ** 2), var_eps)

    # l1 = (1 / (2*denom_end) * (stage_x_end - stage_t_end * x0hat) ** 2).sum()
    # l2 = (1 / (2*denom_start) * (stage_x_start - stage_t_start * DownUp_operation(x0hat)) ** 2).sum()
    # return l1 + l2 + l3, l1, l2, l3


@torch.no_grad()
def langevin_update_split_prior(
    x,
    y,
    operator,
    stage_t_end,
    stage_t_start,
    stage_x_end,
    stage_x_start,
    sigma_n,
    step_size,
    proj,
    o_M,
    lambda_prior,
    device,
):
    x = x.to(device)
    y = y.to(device)

    with torch.enable_grad():
        x = x.detach().requires_grad_(True)
        loss, l1, l2, l3 = objective_split_prior(
            x,
            operator,
            stage_t_end,
            stage_t_start,
            stage_x_end,
            stage_x_start,
            y,
            sigma_n,
            o_M,
            lambda_prior,
        )
        grad = torch.autograd.grad(loss, x)[0]

    grad_norm = grad.norm().item()
    delta_norm = (step_size * grad).norm().item()

    if proj:
        o_M = Dn_k(o_M, x, "nearest")
        y = Dn_k(y, x)
        x_new = x - step_size * grad + math.sqrt(2.0 * step_size) * torch.randn_like(x)
        # x_new = x - step_size * grad 
        x_new = (1-o_M)*x_new + o_M*y
    else:
        x_new = x - step_size * grad + math.sqrt(2.0 * step_size) * torch.randn_like(x)
        # x_new = x - step_size * grad 

    return x_new, {
        "loss": float(loss.detach().cpu()),
        "l1": float(l1.detach().cpu()),
        "l2": float(l2.detach().cpu()),
        "l3": float(l3.detach().cpu()),
        "grad_norm": float(grad_norm),
        "delta_norm": float(delta_norm),
        "z_mean": float(x_new.mean().detach().cpu()),
        "z_std": float(x_new.std().detach().cpu()),
    }


def langevin_sample_split_prior(
    x_init,
    y,
    operator,
    stage_t_end,
    stage_t_start,
    stage_x_end,
    stage_x_start,
    num_Langevin,
    sigma_n,
    lambda_prior=1.0,
    step_size=1e-4,
    device="cpu",
    proj=False,
    return_traj=False,
    record_every=1,
):
    if proj:
        if hasattr(operator, "get_mask"):
            o_M = operator.get_mask(x=y).to(device)
        else:
            o_M = operator.mask.to(device)
    else:
        o_M = torch.zeros_like(y, device=device)

    x = x_init.clone().detach().to(device)
    traj = []
    logs = []
    record_every = max(1, int(record_every))

    for i in range(int(num_Langevin)):
        x, record = langevin_update_split_prior(
            x=x,
            y=y,
            operator=operator,
            stage_t_end=stage_t_end,
            stage_t_start=stage_t_start,
            stage_x_end=stage_x_end,
            stage_x_start=stage_x_start,
            sigma_n=sigma_n,
            step_size=step_size,
            proj=proj,
            o_M=o_M,
            lambda_prior=lambda_prior,
            device=device,
        )
        if return_traj and i % record_every == 0:
            traj.append(x.detach().clone())
            logs.append(record)

    if return_traj:
        return x.detach(), traj, logs
    return x.detach()


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().float().cpu().numpy()
    return np.asarray(x)


def tensor_to_display_image(x, batch_idx=0):
    x = to_numpy(x)

    if x.ndim == 4:
        x = x[batch_idx]

    if x.ndim == 3 and x.shape[0] in (1, 3):
        x = np.transpose(x, (1, 2, 0))

    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]

    x = x.astype(np.float32)
    x = x / 2.0 + 0.5
    return np.clip(x, 0, 1)


def infer_stage_and_step(frame_idx, inference_each_step, num_stages):
    if isinstance(inference_each_step, int):
        stage_idx = frame_idx // inference_each_step
        step_idx = frame_idx % inference_each_step
        if stage_idx >= num_stages:
            stage_idx = num_stages - 1
        return stage_idx, step_idx

    if isinstance(inference_each_step, list):
        if len(inference_each_step) != num_stages:
            raise ValueError(
                f"inference_each_step list length must equal num_stages={num_stages}, got {len(inference_each_step)}"
            )
        running = 0
        for stage_idx, stage_steps in enumerate(inference_each_step):
            next_running = running + int(stage_steps)
            if frame_idx < next_running:
                return stage_idx, frame_idx - running
            running = next_running
        return num_stages - 1, int(inference_each_step[-1]) - 1

    raise TypeError(f"inference_each_step must be int or list[int], got {type(inference_each_step)}")


def repeat_y_as_trajectory(y_tensor, target_len):
    return [y_tensor.clone() for _ in range(target_len)]


def make_combined_video_with_y(
    save_path,
    y_tensor,
    xts_vpred_traj,
    xt_next_vpred_traj,
    x1_vpred_traj_before_langevin,
    x1_vpred_traj_after_langevin,
    xs_vpred_traj,
    xe_vpred_traj,
    xts_exppred_traj,
    xt_next_exppred_traj,
    x1_exppred_traj_before_langevin,
    x1_exppred_traj_after_langevin,
    xs_exppred_traj,
    xe_exppred_traj,
    inference_each_step,
    num_stages,
    latent_update_mode,
    batch_size=4,
    fps=6,
    dpi=220,
):
    target_len = min(
        len(xts_vpred_traj),
        len(xt_next_vpred_traj),
        len(x1_vpred_traj_before_langevin),
        len(x1_vpred_traj_after_langevin),
        len(xs_vpred_traj),
        len(xe_vpred_traj),
        len(xts_exppred_traj),
        len(xt_next_exppred_traj),
        len(x1_exppred_traj_before_langevin),
        len(x1_exppred_traj_after_langevin),
        len(xs_exppred_traj),
        len(xe_exppred_traj),
    )

    y_traj = repeat_y_as_trajectory(y_tensor, target_len)
    trajectories = [
        y_traj,
        xts_vpred_traj[:target_len],
        xt_next_vpred_traj[:target_len],
        x1_vpred_traj_before_langevin[:target_len],
        x1_vpred_traj_after_langevin[:target_len],
        xs_vpred_traj[:target_len],
        xe_vpred_traj[:target_len],
        xts_exppred_traj[:target_len],
        xt_next_exppred_traj[:target_len],
        x1_exppred_traj_before_langevin[:target_len],
        x1_exppred_traj_after_langevin[:target_len],
        xs_exppred_traj[:target_len],
        xe_exppred_traj[:target_len],
    ]

    col_titles = [
        r"$y$",
        r"$x_t^k\ \mathrm{v_{pred}}$",
        r"$\hat{x}_{t+\Delta}^k\ \mathrm{v_{pred}}$",
        r"$\hat{x}_1^k\ \mathrm{v_{pred}}\ \mathrm{before\ Langevin}$",
        r"$\hat{x}_1^k\ \mathrm{v_{pred}}\ \mathrm{after\ Langevin}$",
        r"$\hat{x}_s^k\ \mathrm{v_{pred}}$",
        r"$\hat{x}_e^k\ \mathrm{v_{pred}}$",
        r"$x_t^k\ \mathrm{exp_{pred}}$",
        r"$\hat{x}_{t+\Delta}^k\ \mathrm{exp_{pred}}$",
        r"$\hat{x}_1^k\ \mathrm{exp_{pred}}\ \mathrm{before\ Langevin}$",
        r"$\hat{x}_1^k\ \mathrm{exp_{pred}}\ \mathrm{after\ Langevin}$",
        r"$\hat{x}_s^k\ \mathrm{exp_{pred}}$",
        r"$\hat{x}_e^k\ \mathrm{exp_{pred}}$",
    ]

    num_frames = target_len
    num_rows = min(batch_size, int(to_numpy(y_tensor).shape[0]))
    num_cols = len(trajectories)

    fig_w = max(26, 2.1 * num_cols)
    fig_h = max(10, 2.15 * num_rows)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_w, fig_h), squeeze=False)
    plt.subplots_adjust(left=0.005, right=0.995, top=0.90, bottom=0.02, wspace=0.015, hspace=0.015)

    writer = FFMpegWriter(
        fps=fps,
        metadata={"title": os.path.basename(save_path), "artist": "Codex"},
    )

    for r in range(num_rows):
        for c in range(num_cols):
            axes[r, c].axis("off")

    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontsize=10.8, pad=6)

    with writer.saving(fig, save_path, dpi=dpi):
        for frame_idx in range(num_frames):
            stage_idx, step_idx = infer_stage_and_step(frame_idx, inference_each_step, num_stages)

            for r in range(num_rows):
                for c in range(num_cols):
                    ax = axes[r, c]
                    ax.clear()
                    ax.axis("off")

                    img = tensor_to_display_image(trajectories[c][frame_idx], batch_idx=r)
                    if img.ndim == 2:
                        ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
                    else:
                        ax.imshow(img, vmin=0.0, vmax=1.0, interpolation="nearest")

                    if r == 0:
                        ax.set_title(col_titles[c], fontsize=10.8, pad=6)

            fig.suptitle(
                f"latent_update_mode = {latent_update_mode}   |   Frame {frame_idx + 1}/{num_frames}"
                f"   |   stage = {stage_idx}   |   step = {step_idx}",
                fontsize=16,
                y=0.955,
            )

            for rr in range(num_rows):
                axes[rr, 0].spines["right"].set_visible(True)
                axes[rr, 0].spines["right"].set_linewidth(1.2)
                axes[rr, 6].spines["right"].set_visible(True)
                axes[rr, 6].spines["right"].set_linewidth(1.5)

            writer.grab_frame()

    plt.close(fig)


def flatten_inner_traj_pair(vpred_inner, exppred_inner):
    frames_v = []
    frames_e = []
    meta = []

    outer_len = min(len(vpred_inner), len(exppred_inner))
    for outer_idx in range(outer_len):
        inner_len = min(len(vpred_inner[outer_idx]), len(exppred_inner[outer_idx]))
        for inner_idx in range(inner_len):
            frames_v.append(vpred_inner[outer_idx][inner_idx])
            frames_e.append(exppred_inner[outer_idx][inner_idx])
            meta.append((outer_idx, inner_idx))
    return frames_v, frames_e, meta


def make_langevin_inner_video(
    save_path,
    langevin_inner_traj_vpred,
    langevin_inner_traj_exppred,
    latent_update_mode,
    batch_size=4,
    fps=8,
    dpi=220,
):
    frames_v, frames_e, meta = flatten_inner_traj_pair(
        langevin_inner_traj_vpred,
        langevin_inner_traj_exppred,
    )

    num_frames = min(len(frames_v), len(frames_e))
    if num_frames == 0:
        return

    num_rows = min(batch_size, int(to_numpy(frames_v[0]).shape[0]))
    num_cols = 2

    fig_w = 7.5
    fig_h = max(10, 2.3 * num_rows)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_w, fig_h), squeeze=False)
    plt.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.02, wspace=0.02, hspace=0.02)

    col_titles = [
        r"$\hat{x}_1^k\ \mathrm{v_{pred}}\ \mathrm{Langevin\ inner}$",
        r"$\hat{x}_1^k\ \mathrm{exp_{pred}}\ \mathrm{Langevin\ inner}$",
    ]

    writer = FFMpegWriter(
        fps=fps,
        metadata={"title": os.path.basename(save_path), "artist": "Codex"},
    )

    for r in range(num_rows):
        for c in range(num_cols):
            axes[r, c].axis("off")

    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontsize=11.0, pad=6)

    with writer.saving(fig, save_path, dpi=dpi):
        for frame_idx in range(num_frames):
            outer_idx, inner_idx = meta[frame_idx]

            for r in range(num_rows):
                ax = axes[r, 0]
                ax.clear()
                ax.axis("off")
                img_v = tensor_to_display_image(frames_v[frame_idx], batch_idx=r)
                if img_v.ndim == 2:
                    ax.imshow(img_v, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
                else:
                    ax.imshow(img_v, vmin=0.0, vmax=1.0, interpolation="nearest")
                if r == 0:
                    ax.set_title(col_titles[0], fontsize=11.0, pad=6)

                ax = axes[r, 1]
                ax.clear()
                ax.axis("off")
                img_e = tensor_to_display_image(frames_e[frame_idx], batch_idx=r)
                if img_e.ndim == 2:
                    ax.imshow(img_e, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
                else:
                    ax.imshow(img_e, vmin=0.0, vmax=1.0, interpolation="nearest")
                if r == 0:
                    ax.set_title(col_titles[1], fontsize=11.0, pad=6)

            fig.suptitle(
                f"latent_update_mode = {latent_update_mode}   |   Langevin inner frame {frame_idx + 1}/{num_frames}"
                f"   |   outer step = {outer_idx}   |   inner step = {inner_idx}",
                fontsize=16,
                y=0.955,
            )
            writer.grab_frame()

    plt.close(fig)


def save_posterior_sampling_videos(
    output_dir,
    exp_name,
    latent_update_mode,
    y_tensor,
    xts_vpred_traj,
    xt_next_vpred_traj,
    x1_vpred_traj_before_langevin,
    x1_vpred_traj_after_langevin,
    xs_vpred_traj,
    xe_vpred_traj,
    xts_exppred_traj,
    xt_next_exppred_traj,
    x1_exppred_traj_before_langevin,
    x1_exppred_traj_after_langevin,
    xs_exppred_traj,
    xe_exppred_traj,
    langevin_inner_traj_vpred,
    langevin_inner_traj_exppred,
    inference_each_step,
    num_stages,
    batch_size=4,
    combined_fps=6,
    langevin_fps=8,
    dpi=220,
    save_inner_latents = False
):
    os.makedirs(output_dir, exist_ok=True)
    exp_tag = normalize_experiment_name(exp_name)
    combined_video_path = os.path.join(output_dir, f"{exp_tag}_mode-{latent_update_mode}_with_y.mp4")
    langevin_video_path = os.path.join(output_dir, f"{exp_tag}_mode-{latent_update_mode}_langevin_inner.mp4")

    make_combined_video_with_y(
        save_path=combined_video_path,
        y_tensor=y_tensor,
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
        inference_each_step=inference_each_step,
        num_stages=num_stages,
        latent_update_mode=latent_update_mode,
        batch_size=batch_size,
        fps=combined_fps,
        dpi=dpi,
    )
    if save_inner_latents:
        make_langevin_inner_video(
            save_path=langevin_video_path,
            langevin_inner_traj_vpred=langevin_inner_traj_vpred,
            langevin_inner_traj_exppred=langevin_inner_traj_exppred,
            latent_update_mode=latent_update_mode,
            batch_size=batch_size,
            fps=langevin_fps,
            dpi=dpi,
        )

    return combined_video_path, langevin_video_path


def save_langevin_logs_csv(
    output_dir,
    exp_name,
    latent_update_mode,
    langevin_inner_logs_vpred,
    langevin_inner_logs_exppred,
):
    os.makedirs(output_dir, exist_ok=True)
    exp_tag = normalize_experiment_name(exp_name)
    csv_path = os.path.join(output_dir, f"{exp_tag}_mode-{latent_update_mode}_langevin_logs.csv")
    fieldnames = [
        "source",
        "outer_step",
        "inner_step",
        "loss",
        "l1",
        "l2",
        "l3",
        "grad_norm",
        "delta_norm",
        "z_mean",
        "z_std",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for source_name, logs in (
            ("vpred", langevin_inner_logs_vpred),
            ("exppred", langevin_inner_logs_exppred),
        ):
            for outer_idx, inner_logs in enumerate(logs):
                for inner_idx, record in enumerate(inner_logs):
                    row = {
                        "source": source_name,
                        "outer_step": outer_idx,
                        "inner_step": inner_idx,
                    }
                    row.update(record)
                    writer.writerow(row)

    return csv_path
