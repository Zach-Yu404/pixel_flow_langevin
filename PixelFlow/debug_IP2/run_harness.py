#!/usr/bin/env python
"""
Core harness: runs a single PRINCIPLE-style inpainting pipeline with configurable
hooks for every evolution step. Every variant is selected by string flags so we
can layer changes without editing code.

Flags (all bool unless stated):
  use_old_langevin           old autograd ULA over x1 only (baseline)
  g_identity_stage_last      apply_G is identity on the last stage (match old DownUp)
  warm_x1_from_wls           re-init x1_k with wls x1_hat each Langevin step
  warm_x1_from_vpred         re-init x1_k from model velocity each outer step
  sigma_floor                Tweedie sigma floor (float); default 0.01
  skip_sigma                 skip Langevin when sigma_tau below this (float)
  h_x, h_eps, lambda_x, lambda_reg, rho_s, rho_e
  num_langevin
  stage_steps                int or list
  eps_init_from_vpred        at stage boundary, use model-predicted x0 for eps init
  joint_eps                  True → update eps; False → freeze eps = randn each step
  reset_eps_each_step        re-sample eps at each outer step

Everything is one file so it is easy to diff and reason about.
"""
import sys, os, math, copy, json, time, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.datasets import ImageFolder

from diffusers.models.embeddings import get_2d_rotary_pos_embed
from diffusers.utils.torch_utils import randn_tensor

from inpaintingStart import get_operator

# NOTE: the repo's current ms_posterior_sampling_utils.py has a signature bug
# (DownUp_operation passed stage_idx as positional scale_factor). We bypass it
# by using the git-original DownUp semantics — always scale=2 low-pass.
import ms_posterior_sampling_utils as _mps_utils

def DownUp_git(z, scale_factor=2, stage_idx=None):
    """Git-original DownUp: always low-pass at scale 2, no stage special-case."""
    _, _, H, W = z.shape
    h_small = max(H // scale_factor, 1)
    w_small = max(W // scale_factor, 1)
    z_down = F.interpolate(z, size=(h_small, w_small), mode="bilinear", align_corners=False)
    z_up = F.interpolate(z_down, size=(H, W), mode="nearest")
    return z_up

# Monkey-patch to fix the stage_0 crash in the repo's old utils
_mps_utils.DownUp_operation = DownUp_git
DownUp_operation = DownUp_git

from ms_posterior_sampling_utils import (
    center_crop_arr, sample_block_noise,
    langevin_update_split_prior, pred_x1_x0_with_vt,
    get_xt_from_x1_sameStage, get_lr, class_guidance_scale,
    get_x1_and_noise_with_stage_start_and_end2,
)
from ms_posterior_sampling_article_version_utils import (
    apply_G as _apply_G_lowpass,
    compute_sigma_tau, wls_estimate_x1,
    make_Ak_fns, cg_solve, make_velocity_fn,
)
from pixelflow.scheduling_pixelflow import PixelFlowScheduler
from pixelflow.utils import config as config_utils
from pixelflow.utils.misc import seed_everything


# --------------------------------------------------------------------------
# Patched G / H_tau — stage-aware version
# --------------------------------------------------------------------------

def apply_G_stage(x, stage_idx=None, num_stages=4, g_identity_stage_last=False):
    """If g_identity_stage_last and stage is the last one, G = I. Else low-pass."""
    if g_identity_stage_last and stage_idx is not None and stage_idx == num_stages - 1:
        return x
    return _apply_G_lowpass(x)


def apply_H_tau_stage(x, tau, s_k, e_k, stage_idx=None, num_stages=4, g_identity_stage_last=False):
    g = apply_G_stage(x, stage_idx, num_stages, g_identity_stage_last)
    return (1.0 - tau) * s_k * g + tau * e_k * x


def apply_HT_tau_stage(x, tau, s_k, e_k, stage_idx=None, num_stages=4, g_identity_stage_last=False):
    # G self-adjoint (and identity at last stage if flagged)
    g = apply_G_stage(x, stage_idx, num_stages, g_identity_stage_last)
    return (1.0 - tau) * s_k * g + tau * e_k * x


def wls_estimate_x1_stage(x_start_hat, x_end_hat, s_k, e_k, rho_s, rho_e,
                          lambda_x, cg_tol, cg_max_iter,
                          stage_idx=None, num_stages=4, g_identity_stage_last=False):
    var_eps = 1e-2
    denom_s = max((1.0 - s_k) ** 2, var_eps)
    denom_e = max((1.0 - e_k) ** 2, var_eps)
    coeff_GTG = rho_s * s_k ** 2 / denom_s
    coeff_I = rho_e * e_k ** 2 / denom_e + lambda_x

    def G(x):
        return apply_G_stage(x, stage_idx, num_stages, g_identity_stage_last)

    def M_fn(x):
        return coeff_GTG * G(G(x)) + coeff_I * x

    r = (rho_s * s_k / denom_s) * G(x_start_hat) + \
        (rho_e * e_k / denom_e) * x_end_hat
    return cg_solve(M_fn, r, tol=cg_tol, max_iter=cg_max_iter)


# --------------------------------------------------------------------------
# Modified Langevin step with all the toggles we care about
# --------------------------------------------------------------------------

@torch.no_grad()
def _langevin_step_v2(x1_k, eps_k, x_tau_k, tau, s_k, e_k, sigma_tau,
                      velocity_fn, A_k_fn, AT_k_fn, b, eta,
                      h_x, h_eps, lambda_x, lambda_reg,
                      rho_s, rho_e, cg_tol, cg_max_iter,
                      stage_idx, num_stages, g_identity_stage_last,
                      sigma_floor=0.01,
                      warm_x1_from_wls=False,
                      joint_eps=True):
    sigma_safe = max(float(sigma_tau), sigma_floor)

    mu = velocity_fn(x_tau_k)
    xs_hat = x_tau_k - tau * mu
    xe_hat = x_tau_k + (1.0 - tau) * mu

    x1_hat = wls_estimate_x1_stage(
        xs_hat, xe_hat, s_k, e_k, rho_s, rho_e, lambda_x, cg_tol, cg_max_iter,
        stage_idx=stage_idx, num_stages=num_stages,
        g_identity_stage_last=g_identity_stage_last,
    )

    if warm_x1_from_wls:
        x1_k = x1_hat.clone()

    H_x1_hat = apply_H_tau_stage(x1_hat, tau, s_k, e_k, stage_idx, num_stages, g_identity_stage_last)
    s_flow = (1.0 / sigma_safe ** 2) * (H_x1_hat - x_tau_k)

    residual = b - A_k_fn(x1_k)
    g_x1 = (1.0 / eta ** 2) * AT_k_fn(residual) + apply_HT_tau_stage(
        s_flow, tau, s_k, e_k, stage_idx, num_stages, g_identity_stage_last)

    # x1 update
    xi_1, xi_2 = torch.randn_like(b), torch.randn_like(x1_k)
    def system(x):
        return (1.0 / eta ** 2) * AT_k_fn(A_k_fn(x)) + lambda_reg * x
    rhs = (h_x / 2) * g_x1 + math.sqrt(h_x) * (
        (1.0 / eta) * AT_k_fn(xi_1) + math.sqrt(lambda_reg) * xi_2)
    delta = cg_solve(system, rhs, tol=cg_tol, max_iter=cg_max_iter)
    x1_k = x1_k + delta

    # eps update
    if joint_eps:
        g_eps = sigma_tau * s_flow - eps_k
        x3 = torch.randn_like(eps_k)
        eps_k = eps_k + (h_eps / 2) * g_eps + math.sqrt(h_eps) * x3

    x_tau_k = apply_H_tau_stage(x1_k, tau, s_k, e_k, stage_idx, num_stages, g_identity_stage_last) + sigma_tau * eps_k

    log = {
        "residual_obs_sq": float((residual ** 2).sum()),
        "grad_norm": float(g_x1.norm()),
        "delta_norm": float(delta.norm()),
        "x1_mean": float(x1_k.mean()),
        "x1_std": float(x1_k.std()),
        "eps_std": float(eps_k.std()),
        "sigma_tau": float(sigma_tau),
    }
    return x1_k, eps_k, x_tau_k, log


def principle_langevin_v2(x1_init, eps_init, tau, s_k, e_k,
                          velocity_fn, A_k_fn, AT_k_fn,
                          y, sigma_n, h_x, h_epsilon,
                          lambda_x, lambda_reg, rho_s, rho_e,
                          cg_tol, cg_max_iter, num_Langevin, device,
                          stage_idx, num_stages, g_identity_stage_last,
                          sigma_floor=0.01, warm_x1_from_wls=False, joint_eps=True):
    sigma_tau = compute_sigma_tau(tau, s_k, e_k)
    x1_k = x1_init.clone().detach().to(device)
    eps_k = eps_init.clone().detach().to(device)
    x_tau_k = apply_H_tau_stage(x1_k, tau, s_k, e_k, stage_idx, num_stages, g_identity_stage_last) + sigma_tau * eps_k

    logs = []
    for _ in range(int(num_Langevin)):
        x1_k, eps_k, x_tau_k, log = _langevin_step_v2(
            x1_k, eps_k, x_tau_k, tau, s_k, e_k, sigma_tau,
            velocity_fn, A_k_fn, AT_k_fn, y, sigma_n,
            h_x, h_epsilon, lambda_x, lambda_reg,
            rho_s, rho_e, cg_tol, cg_max_iter,
            stage_idx, num_stages, g_identity_stage_last,
            sigma_floor=sigma_floor,
            warm_x1_from_wls=warm_x1_from_wls,
            joint_eps=joint_eps,
        )
        logs.append(log)
    return x1_k.detach(), eps_k.detach(), logs


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------

def run_pipeline(flags):
    """flags: dict with experiment configuration."""
    device = torch.device(flags.get("device", "cuda:0"))
    seed = int(flags.get("seed", 20000120))
    num_stages = int(flags.get("num_stages", 4))
    resolution = int(flags.get("resolution", 256))
    num_examples = int(flags.get("num_examples", 2))
    sigma_n = float(flags.get("sigma_n", 0.05))
    class_label = int(flags.get("class_label", 10))
    shift = float(flags.get("shift", 1.0))
    stage_steps = flags.get("stage_steps", 5)
    num_langevin = int(flags.get("num_langevin", 10))
    h_x = float(flags.get("h_x", 1e-3))
    h_eps = float(flags.get("h_eps", 1e-3))
    lambda_x = float(flags.get("lambda_x", 0.01))
    lambda_reg = float(flags.get("lambda_reg", 0.01))
    rho_s_cfg = flags.get("rho_s", 1.0)
    rho_e_cfg = flags.get("rho_e", 1.0)
    cg_tol = float(flags.get("cg_tol", 1e-5))
    cg_max_iter = int(flags.get("cg_max_iter", 50))
    lr_base = float(flags.get("lr_base", 2e-2))
    lr_min_ratio = float(flags.get("lr_min_ratio", 1e-2))
    lambda_prior = float(flags.get("lambda_prior", 1e-5))
    measurement_mode = flags.get("measurement_mode", "measure")
    active_operator_name = flags.get("active_operator", "box")
    data_dir = flags.get("data_dir", "/data/Zach_dataset/imagenet256/train/")
    model_dir = flags.get("model_dir", "./pretrained_models/c2img")

    # Algorithm toggles
    use_old_langevin = bool(flags.get("use_old_langevin", False))
    g_identity_stage_last = bool(flags.get("g_identity_stage_last", False))
    warm_x1_from_wls = bool(flags.get("warm_x1_from_wls", False))
    warm_x1_from_vpred = bool(flags.get("warm_x1_from_vpred", False))
    sigma_floor = float(flags.get("sigma_floor", 0.01))
    skip_sigma = float(flags.get("skip_sigma", 0.01))
    joint_eps = bool(flags.get("joint_eps", True))
    eps_init_from_vpred = bool(flags.get("eps_init_from_vpred", False))
    reset_eps_each_step = bool(flags.get("reset_eps_each_step", False))

    seed_everything(seed)

    # Data
    transform = transforms.Compose([
        transforms.Lambda(lambda pil: center_crop_arr(pil, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3, inplace=True),
    ])
    dataset = ImageFolder(data_dir, transform=transform)
    class_indices = [i for i, (_, y_lbl) in enumerate(dataset.samples) if int(y_lbl) == class_label]
    perm = torch.randperm(len(class_indices))[:num_examples]
    selected = [class_indices[i] for i in perm.tolist()]
    gt = torch.stack([dataset[i][0] for i in selected], dim=0).to(device)

    # Operator
    box_cfg = flags.get("box_operator", {
        "mask_type": "box", "mask_len_range": [80, 160], "mask_prob_range": None,
    })
    random_cfg = flags.get("random_operator", {
        "mask_type": "random", "mask_len_range": None, "mask_prob_range": [0.8, 0.8],
    })
    op_cfg = box_cfg if active_operator_name == "box" else random_cfg
    operator = get_operator(
        "inpainting", mask_type=op_cfg["mask_type"],
        mask_len_range=None if op_cfg.get("mask_len_range") is None else tuple(op_cfg["mask_len_range"]),
        mask_prob_range=None if op_cfg.get("mask_prob_range") is None else tuple(op_cfg["mask_prob_range"]),
        resolution=resolution, device=device, sigma=sigma_n,
    )

    # Model
    config = OmegaConf.load(os.path.join(model_dir, "config.yaml"))
    model = config_utils.instantiate_from_config(config.model).to(device)
    ckpt = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt, strict=True)
    model.eval()

    scheduler = PixelFlowScheduler(
        config.scheduler.num_train_timesteps,
        num_stages=config.scheduler.num_stages, gamma=-1/3,
    )

    # Measurement
    if measurement_mode == "call":
        y = operator(gt).detach()
    else:
        y = operator.measure(gt).detach()

    # Init
    init_factor = 2 ** (num_stages - 1)
    H = resolution // init_factor
    W = resolution // init_factor
    x1_k = torch.randn(num_examples, 3, H, W, device=device)
    eps_k = randn_tensor((num_examples, 3, H, W), device=device, dtype=torch.float32)
    latent_tau = eps_k.clone()
    latents_v = eps_k.clone()  # for old path

    uncond_label = int(model.num_classes)
    prompt_embeds = torch.tensor([class_label] * num_examples, dtype=torch.int32, device=device)

    # Result containers
    residuals = []   # ||A(x1_at_full_res) - y||² per outer step
    x1_snapshots = []   # x1 per stage end (upsampled to full if needed)
    per_step_logs = []

    def _get_stage_steps(k):
        if isinstance(stage_steps, list):
            return int(stage_steps[k])
        return int(stage_steps)

    A_full_fn, _ = make_Ak_fns(operator, y, (num_examples, 3, resolution, resolution), device)

    for stage_idx in range(num_stages):
        sched = copy.deepcopy(scheduler)
        steps_k = _get_stage_steps(stage_idx)
        sched.set_timesteps(steps_k, stage_idx, device=device, shift=shift)
        Timesteps_k = sched.Timesteps
        start_t = sched.start_t[stage_idx]
        end_t = sched.end_t[stage_idx]
        s_k = float(start_t)
        e_k = float(end_t)

        if stage_idx > 0:
            H *= 2
            W *= 2
            latent_tau = F.interpolate(latent_tau, size=(H, W), mode="nearest")
            latents_v = F.interpolate(latents_v, size=(H, W), mode="nearest")
            original_start_t = sched.original_start_t[stage_idx]
            gamma = sched.gamma
            alpha = 1 / (math.sqrt(1 - (1 / gamma)) * (1 - original_start_t) + original_start_t)
            beta = alpha * (1 - original_start_t) / math.sqrt(-gamma)
            noise = sample_block_noise(sched, num_examples, 3, H, W).to(device=device, dtype=latent_tau.dtype)
            latent_tau = alpha * latent_tau + beta * noise
            latents_v = alpha * latents_v + beta * noise

            x1_k = F.interpolate(x1_k, size=(H, W), mode="nearest")
            s0 = float(start_t)
            eps_k = (latent_tau - s0 * apply_G_stage(x1_k, stage_idx, num_stages, g_identity_stage_last)) / max(1.0 - s0, 1e-8)

        stage_shape = (num_examples, 3, H, W)
        A_k_fn, AT_k_fn = make_Ak_fns(operator, y, stage_shape, device)
        rho_s_k = float(rho_s_cfg[stage_idx]) if isinstance(rho_s_cfg, list) else float(rho_s_cfg)
        rho_e_k = float(rho_e_cfg[stage_idx]) if isinstance(rho_e_cfg, list) else float(rho_e_cfg)

        size_tensor = torch.tensor([H // model.patch_size], dtype=torch.int32, device=device)
        pos_embed = get_2d_rotary_pos_embed(
            embed_dim=model.attention_head_dim,
            crops_coords=((0, 0), (H // model.patch_size, W // model.patch_size)),
            grid_size=(H // model.patch_size, W // model.patch_size),
            device=device, output_type="pt",
        )
        rope_pos = torch.stack(pos_embed, -1)

        for step_idx, T in enumerate(Timesteps_k):
            t_curr = sched.t[step_idx].to(device=device, dtype=torch.float32)
            tau = float(t_curr)
            sigma_tau_val = compute_sigma_tau(tau, s_k, e_k)

            if use_old_langevin:
                # Reconstruct x_tau from current (x1, eps) via old-style
                inp = latents_v
                ts = T.expand(inp.shape[0]).to(inp.dtype)
                with torch.no_grad():
                    noise_pred = model(inp, timestep=ts, class_labels=prompt_embeds,
                                       latent_size=size_tensor, pos_embed=rope_pos)

                pixel_values_start = latents_v - tau * noise_pred
                pixel_values_end = latents_v + (1.0 - tau) * noise_pred
                x1_vPred, x0_vPred = pred_x1_x0_with_vt(latents_v, noise_pred, T, start_t, end_t)

                ratio = float(T.detach().item()) / 1000.0
                lr = get_lr(ratio, lr_base=lr_base, lr_min_ratio=lr_min_ratio)

                x1_refined = x1_vPred.clone()
                for _ in range(num_langevin):
                    x1_refined, _ = langevin_update_split_prior(
                        x1_refined, y, operator, end_t, start_t,
                        pixel_values_end.detach(), pixel_values_start.detach(),
                        sigma_n, step_size=lr, proj=False,
                        o_M=None, lambda_prior=lambda_prior, device=device,
                    )
                x1_k = x1_refined.detach()

                # f_x1 mode update to next latent — inlined to bypass repo bug.
                # Reconstruct pixel_values_end at next step from updated x1:
                #   xs = DU_x0 (observed start)
                #   xe = e*x1 + (1-e)/(1-s) * (xs - s*DU(x1))
                #   xt_next = t_next * xe + (1 - t_next) * xs
                t_next = sched.t[step_idx + 1].to(device=device, dtype=torch.float32)
                DU_x1 = DownUp_git(x1_k)
                xe_recon = e_k * x1_k + (1 - e_k) / max(1 - s_k, 1e-8) * (
                    pixel_values_start - s_k * DU_x1
                )
                latents_v = float(t_next) * xe_recon + (1 - float(t_next)) * pixel_values_start

                if H == resolution:
                    res = (y - A_full_fn(x1_k)).pow(2).sum().item()
                else:
                    x1_up = F.interpolate(x1_k, size=(resolution, resolution), mode="bilinear", align_corners=False)
                    res = (y - A_full_fn(x1_up)).pow(2).sum().item()
                residuals.append(res)
                per_step_logs.append({"residual": res, "stage": stage_idx, "step": step_idx})
                continue

            # --- New (article-style) path with all toggles ---
            if warm_x1_from_vpred:
                # Re-init x1_k from model velocity using current x_tau
                x_tau_k = apply_H_tau_stage(x1_k, tau, s_k, e_k, stage_idx, num_stages, g_identity_stage_last) + sigma_tau_val * eps_k
                inp = x_tau_k
                ts = T.expand(inp.shape[0]).to(inp.dtype)
                with torch.no_grad():
                    noise_pred = model(inp, timestep=ts, class_labels=prompt_embeds,
                                       latent_size=size_tensor, pos_embed=rope_pos)
                x1_vpred, _ = pred_x1_x0_with_vt(x_tau_k, noise_pred, T, start_t, end_t)
                x1_k = x1_vpred.detach().clone()

            if reset_eps_each_step:
                eps_k = torch.randn_like(eps_k)

            velocity_fn = make_velocity_fn(
                model, T, prompt_embeds, size_tensor, rope_pos,
                False, 0.0, stage_idx,
            )

            if sigma_tau_val >= skip_sigma:
                x1_k, eps_k, logs = principle_langevin_v2(
                    x1_init=x1_k, eps_init=eps_k,
                    tau=tau, s_k=s_k, e_k=e_k,
                    velocity_fn=velocity_fn, A_k_fn=A_k_fn, AT_k_fn=AT_k_fn,
                    y=y, sigma_n=sigma_n, h_x=h_x, h_epsilon=h_eps,
                    lambda_x=lambda_x, lambda_reg=lambda_reg,
                    rho_s=rho_s_k, rho_e=rho_e_k,
                    cg_tol=cg_tol, cg_max_iter=cg_max_iter,
                    num_Langevin=num_langevin, device=device,
                    stage_idx=stage_idx, num_stages=num_stages,
                    g_identity_stage_last=g_identity_stage_last,
                    sigma_floor=sigma_floor,
                    warm_x1_from_wls=warm_x1_from_wls,
                    joint_eps=joint_eps,
                )

            # Guard against divergence: re-seed eps if it's drifted
            if torch.isnan(x1_k).any() or torch.isinf(x1_k).any() or x1_k.abs().max() > 100:
                print(f"  !! divergence detected stage {stage_idx} step {step_idx} — re-seeding", flush=True)
                x1_k = torch.clamp(torch.nan_to_num(x1_k), -5, 5)
                eps_k = torch.randn_like(eps_k)

            latent_tau = apply_H_tau_stage(x1_k, tau, s_k, e_k, stage_idx, num_stages, g_identity_stage_last) + sigma_tau_val * eps_k

            if H == resolution:
                res = (y - A_full_fn(x1_k)).pow(2).sum().item()
            else:
                x1_up = F.interpolate(x1_k, size=(resolution, resolution), mode="bilinear", align_corners=False)
                res = (y - A_full_fn(x1_up)).pow(2).sum().item()
            residuals.append(res)
            per_step_logs.append({"residual": res, "stage": stage_idx, "step": step_idx,
                                  "sigma_tau": sigma_tau_val})
            print(f"    s{stage_idx} step{step_idx}: res={res:.0f}  σ_τ={sigma_tau_val:.3f}", flush=True)

        x1_snapshots.append(x1_k.detach().cpu())

    return {
        "gt": gt.detach().cpu(),
        "y": y.detach().cpu(),
        "x1_final": x1_k.detach().cpu(),
        "x1_snapshots": x1_snapshots,
        "residuals": residuals,
        "per_step_logs": per_step_logs,
    }


def save_figures(results_dict, save_dir, title):
    """Dump GT, y, and per-experiment final x1 reconstructions as PNG grids."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)
    names = list(results_dict.keys())
    first = results_dict[names[0]]
    gt = first["gt"]
    y = first["y"]
    B = gt.shape[0]

    ncols = 2 + len(names)
    nrows = B
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 2.5 * nrows), squeeze=False)

    def to_img(x):
        x = x.detach().float().cpu()
        if x.dim() == 4:
            x = x[0] if x.shape[0] == 1 else x
        x = x / 2.0 + 0.5
        x = x.clamp(0, 1)
        return x.permute(1, 2, 0).numpy()

    for b in range(B):
        axes[b, 0].imshow(to_img(gt[b:b+1]))
        axes[b, 0].set_title("GT" if b == 0 else "")
        axes[b, 0].axis("off")
        axes[b, 1].imshow(to_img(y[b:b+1]))
        axes[b, 1].set_title("y" if b == 0 else "")
        axes[b, 1].axis("off")
        for i, name in enumerate(names):
            r = results_dict[name]
            x1 = r["x1_final"]
            axes[b, 2 + i].imshow(to_img(x1[b:b+1]))
            if b == 0:
                final_res = r["residuals"][-1] if r["residuals"] else float("nan")
                axes[b, 2 + i].set_title(f"{name}\n res={final_res:.0f}", fontsize=8)
            axes[b, 2 + i].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    out = os.path.join(save_dir, f"{title}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)

    # Residual curves
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for name, r in results_dict.items():
        ax.plot(r["residuals"], label=name)
    ax.set_xlabel("outer step")
    ax.set_ylabel("||A(x1) - y||²")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{title}_residuals.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--flags", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    with open(args.flags) as f:
        flags = json.load(f)
    t0 = time.time()
    result = run_pipeline(flags)
    print(f"Pipeline done in {time.time()-t0:.1f}s, final residual = {result['residuals'][-1]:.2f}")
    torch.save(result, args.out)
    print(f"saved → {args.out}")
