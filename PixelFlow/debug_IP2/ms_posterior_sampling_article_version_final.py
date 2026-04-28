#!/usr/bin/env python
"""
FINAL version of the article-style PRINCIPLE inpainting sampler (debug_IP2).

Key differences vs. main `ms_posterior_sampling_article_version.py`:

  1) At each outer time step, x1_k is warm-restarted from the model's v-pred on
     the current x_tau. This mirrors the old sampler's behaviour (the source
     of stability), and limits the drift that plagues the main article version.

  2) Langevin step size h_x and CG ridge lambda_reg are chosen so that the
     per-step correction in observed regions is ~5% of the residual.
     With h_x=5e-2 and sigma_n=0.05, inner steps converge the measurement
     residual to its noise floor in 5–10 iterations. (Main article's h_x=1e-3
     under-damps by a factor of 50.)

  3) G is the stage-invariant DownUp at scale 2 (matches PixelFlow training
     for pixel_values_start). The main article already used this; we keep it.
     NB: the repo's old `ms_posterior_sampling_utils.DownUp_operation` was
     hand-patched to be "identity at stage 3" — that deviates from training
     and is not used here.

  4) Stage boundary: after renoising latent_tau with PixelFlow alpha/beta +
     block noise, we RE-SEED x1_k from one forward pass of the flow model
     at the new resolution (v-pred on renoised latent_tau). This avoids the
     nearest-upsample-from-prev-stage initialization that corrupts eps in
     the main article version.

  5) sigma_floor is only applied to 1/σ² in the Tweedie score; raw sigma_tau
     is kept in g_eps (multiplicative) and x_tau reconstruction.

The overall control flow and outputs (save_posterior_sampling_videos etc.)
match the main article version so downstream viewers still work.
"""
import argparse, copy, math, os, shutil, sys

# Order matters: put debug_IP2/ FIRST in sys.path so this script's sibling utils
# (debug_IP2 edition) wins over any top-level file of the same name.
_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_THIS))  # repo root — for inpaintingStart, pixelflow, ms_posterior_sampling_utils
sys.path.insert(0, _THIS)                   # debug_IP2 — precedence for our own utils

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.datasets import ImageFolder

from diffusers.models.embeddings import get_2d_rotary_pos_embed
from diffusers.utils.torch_utils import randn_tensor

from inpaintingStart import get_operator

# Local utils (debug_IP2 edition) — imported explicitly by file path to avoid
# any collision with the top-level ms_posterior_sampling_article_version_final_utils.py
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "_debug_ip2_final_utils",
    os.path.join(_THIS, "ms_posterior_sampling_article_version_final_utils.py"),
)
_utils = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_utils)

(apply_G, apply_H_tau, apply_HT_tau,
 build_experiment_paths, center_crop_arr,
 compute_sigma_tau, get_stage_inference_steps,
 load_run_config, make_Ak_fns, make_velocity_fn,
 pred_x1_from_vpred, principle_langevin_final,
 resolve_path, sample_block_noise, save_langevin_logs_csv,
 save_posterior_sampling_videos, wls_estimate_x1) = (
    _utils.apply_G, _utils.apply_H_tau, _utils.apply_HT_tau,
    _utils.build_experiment_paths, _utils.center_crop_arr,
    _utils.compute_sigma_tau, _utils.get_stage_inference_steps,
    _utils.load_run_config, _utils.make_Ak_fns, _utils.make_velocity_fn,
    _utils.pred_x1_from_vpred, _utils.principle_langevin_final,
    _utils.resolve_path, _utils.sample_block_noise, _utils.save_langevin_logs_csv,
    _utils.save_posterior_sampling_videos, _utils.wls_estimate_x1,
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
    device = torch.device(device_str if (not isinstance(device_str, str) or not device_str.startswith("cuda") or torch.cuda.is_available()) else "cpu")
    data_dir = resolve_path(config_dir, cfg["data_dir"])
    model_dir = resolve_path(config_dir, cfg["model_dir"])
    dict_root = resolve_path(config_dir, cfg["dict_path"])
    record_every = int(cfg.get("record_every", 1))
    active_operator_name = cfg["active_operator"]
    measurement_mode = cfg["measurement_mode"]
    latent_update_mode = str(cfg.get("latent_update_mode", "principle_final"))

    h_x = float(cfg["h_x"])
    h_epsilon = float(cfg["h_epsilon"])
    lambda_x = float(cfg["lambda_x"])
    lambda_reg = float(cfg["lambda_reg"])
    rho_s_cfg = cfg["rho_s"]
    rho_e_cfg = cfg["rho_e"]
    cg_tol = float(cfg["cg_tol"])
    cg_max_iter = int(cfg["cg_max_iter"])
    warm_x1_from_vpred = bool(cfg.get("warm_x1_from_vpred", True))
    warm_x1_from_wls = bool(cfg.get("warm_x1_from_wls", False))
    sigma_floor = float(cfg.get("sigma_floor", 0.02))
    skip_sigma = float(cfg.get("skip_sigma", 0.02))
    reseed_x1_at_stage_start = bool(cfg.get("reseed_x1_at_stage_start", True))
    joint_eps = bool(cfg.get("joint_eps", True))

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

    # ── operator ────────────────────────────────────────────────────
    box_cfg = cfg["box_operator"]
    random_cfg = cfg["random_operator"]
    def _build(op_cfg):
        return get_operator(
            "inpainting", mask_type=op_cfg["mask_type"],
            mask_len_range=None if op_cfg.get("mask_len_range") is None else tuple(op_cfg["mask_len_range"]),
            mask_prob_range=None if op_cfg.get("mask_prob_range") is None else tuple(op_cfg["mask_prob_range"]),
            resolution=resolution, device=device, sigma=sigma_n,
        )
    box_operator = _build(box_cfg)
    random_operator = _build(random_cfg)
    active_operator = {"random": random_operator, "box": box_operator}[active_operator_name]

    # ── model ───────────────────────────────────────────────────
    config = OmegaConf.load(os.path.join(model_dir, "config.yaml"))
    model = config_utils.instantiate_from_config(config.model).to(device)
    print(f"Num of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    ckpt = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt, strict=True)
    model.eval()

    scheduler = PixelFlowScheduler(
        config.scheduler.num_train_timesteps,
        num_stages=config.scheduler.num_stages, gamma=-1/3,
    )
    if num_stages != int(config.scheduler.num_stages):
        raise ValueError(f"num_stages={num_stages} != config {config.scheduler.num_stages}")
    scheduler_copy = copy.deepcopy(scheduler)

    # ── measurement ────────────────────────────────────────────────────
    y = (active_operator(gt).detach() if measurement_mode == "call"
         else active_operator.measure(gt).detach())

    # ── trajectory tracking ────────────────────────────────────────────
    xts_traj, xt_next_traj = [], []
    x1_traj_before, x1_traj_after = [], []
    eps_traj = []
    langevin_inner_traj = []
    langevin_inner_logs = []

    # ── CFG setup ───────────────────────────────────────────────────
    uncond_label = int(model.num_classes)
    prompt_class_label = uncond_label if class_label is None else int(class_label)
    prompt_embeds = torch.tensor([prompt_class_label] * num_examples, dtype=torch.int32, device=device)
    negative_prompt_embeds = uncond_label * torch.ones_like(prompt_embeds)
    do_cfg = guidance_scale > 0
    if do_cfg:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

    # ── init ───────────────────────────────────────────────────
    init_factor = 2 ** (num_stages - 1)
    height = resolution // init_factor
    width = resolution // init_factor
    shape = (num_examples, 3, height, width)

    x1_k = torch.randn(shape, device=device, dtype=torch.float32)
    eps_k = randn_tensor(shape, device=device, dtype=torch.float32)
    latent_tau = eps_k.clone()  # stage 0, tau=0, s=0 → x_tau = eps

    # ── main loop ──────────────────────────────────────────────────────
    for stage_idx in range(num_stages):
        stage_inference_steps = get_stage_inference_steps(inference_each_step, stage_idx, num_stages)
        scheduler_copy.set_timesteps(stage_inference_steps, stage_idx, device=device, shift=shift)
        Timesteps_k = scheduler_copy.Timesteps
        start_t = scheduler_copy.start_t[stage_idx]
        end_t = scheduler_copy.end_t[stage_idx]
        s_k = float(start_t)
        e_k = float(end_t)

        if stage_idx > 0:
            height *= 2
            width *= 2

            # PixelFlow renoise on latent_tau
            latent_tau = F.interpolate(latent_tau, size=(height, width), mode="nearest")
            original_start_t = scheduler_copy.original_start_t[stage_idx]
            gamma = scheduler_copy.gamma
            alpha = 1 / (math.sqrt(1 - (1 / gamma)) * (1 - original_start_t) + original_start_t)
            beta = alpha * (1 - original_start_t) / math.sqrt(-gamma)
            noise = sample_block_noise(scheduler_copy, num_examples, 3, height, width).to(device=device, dtype=latent_tau.dtype)
            latent_tau = alpha * latent_tau + beta * noise

            # Re-seed x1 at new resolution from the flow model's v-pred at tau=0.
            # This avoids propagating a stale nearest-upsample of the previous x1.
            if reseed_x1_at_stage_start:
                size_tensor = torch.tensor([height // model.patch_size], dtype=torch.int32, device=device)
                pos_embed = get_2d_rotary_pos_embed(
                    embed_dim=model.attention_head_dim,
                    crops_coords=((0, 0), (height // model.patch_size, width // model.patch_size)),
                    grid_size=(height // model.patch_size, width // model.patch_size),
                    device=device, output_type="pt",
                )
                rope_pos = torch.stack(pos_embed, -1)
                T0 = Timesteps_k[0]
                with torch.no_grad():
                    noise_pred = model(
                        latent_tau, timestep=T0.expand(num_examples).to(latent_tau.dtype),
                        class_labels=prompt_embeds[-num_examples:] if do_cfg else prompt_embeds,
                        latent_size=size_tensor, pos_embed=rope_pos,
                    )
                x1_k = pred_x1_from_vpred(latent_tau, noise_pred, T0, start_t, end_t).detach()
            else:
                x1_k = F.interpolate(x1_k, size=(height, width), mode="nearest")

            # Derive eps_k from latent_tau and new x1_k assuming tau=0:
            # x_tau = s_k * G(x1_k) + (1 - s_k) * eps_k
            eps_k = (latent_tau - s_k * apply_G(x1_k)) / max(1.0 - s_k, 1e-8)

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
            tau = float(t_curr)
            sigma_t = compute_sigma_tau(tau, s_k, e_k)

            # Construct x_tau from current x1_k, eps_k
            x_tau_k = apply_H_tau(x1_k, tau, s_k, e_k) + sigma_t * eps_k

            xts_traj.append(x_tau_k.detach().clone())
            x1_traj_before.append(x1_k.detach().clone())

            # --- Warm start x1 from v-pred at current x_tau ---
            if warm_x1_from_vpred:
                with torch.no_grad():
                    noise_pred = model(
                        x_tau_k,
                        timestep=T.expand(num_examples).to(x_tau_k.dtype),
                        class_labels=prompt_embeds[-num_examples:] if do_cfg else prompt_embeds,
                        latent_size=size_tensor, pos_embed=rope_pos,
                    )
                x1_k = pred_x1_from_vpred(x_tau_k, noise_pred, T, start_t, end_t).detach()

            velocity_fn = make_velocity_fn(
                model, T, prompt_embeds, size_tensor, rope_pos,
                do_cfg, guidance_scale, stage_idx,
            )

            if sigma_t < skip_sigma:
                inner_traj, inner_logs = [], []
            else:
                x1_k, eps_k, inner_traj, inner_logs = principle_langevin_final(
                    x1_init=x1_k, eps_init=eps_k,
                    tau=tau, s_k=s_k, e_k=e_k,
                    velocity_fn=velocity_fn, A_k_fn=A_k_fn, AT_k_fn=AT_k_fn,
                    y=y, sigma_n=sigma_n, h_x=h_x, h_epsilon=h_epsilon,
                    lambda_x=lambda_x, lambda_reg=lambda_reg,
                    rho_s=rho_s_k, rho_e=rho_e_k,
                    cg_tol=cg_tol, cg_max_iter=cg_max_iter,
                    num_Langevin=num_langevin, device=device,
                    sigma_floor=sigma_floor,
                    warm_x1_from_wls=warm_x1_from_wls,
                    joint_eps=joint_eps,
                    return_traj=True, record_every=record_every,
                )

            x1_traj_after.append(x1_k.detach().clone())
            eps_traj.append(eps_k.detach().clone())
            latent_tau = apply_H_tau(x1_k, tau, s_k, e_k) + sigma_t * eps_k
            xt_next_traj.append(latent_tau.detach().clone())
            langevin_inner_traj.append(inner_traj)
            langevin_inner_logs.append(inner_logs)

    # ── saving ──────────────────────────────────────────────────────
    save_dir = os.path.dirname(dict_path)
    os.makedirs(save_dir, exist_ok=True)
    shutil.copy2(config_path, config_copy_path)

    if save_dict_to_pt:
        torch.save({
            "config_path": config_path, "run_config": cfg,
            "exp_name": exp_name, "selected_indices": selected_indices,
            "labels": labels,
            "gt": gt.detach().cpu(), "y": y.detach().cpu(),
            "xts_traj": xts_traj, "xt_next_traj": xt_next_traj,
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
        principle_col_titles = [
            r"$y$", r"$x_\tau^k$", r"$x_{\tau,\mathrm{next}}^k$",
            r"$\hat{x}_1^k\ \mathrm{before}$", r"$\hat{x}_1^k\ \mathrm{after}$",
            r"$\hat{x}_s^k$", r"$\hat{x}_e^k$",
            r"$x_\tau^k$", r"$x_{\tau,\mathrm{next}}^k$",
            r"$\hat{x}_1^k\ \mathrm{before}$", r"$\hat{x}_1^k\ \mathrm{after}$",
            r"$\epsilon^k$", r"$x_{\tau,\mathrm{next}}^k$",
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
        "ms_posterior_sampling_article_version_final.json",
    )
    parser = argparse.ArgumentParser(description="PRINCIPLE sampler — FINAL (debug_IP2)")
    parser.add_argument("--config", type=str, default=default_config_path)
    args = parser.parse_args()
    main(args.config)
