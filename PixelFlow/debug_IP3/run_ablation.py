#!/usr/bin/env python
"""
Ablation study for PRINCIPLE v2 improvements.
Tests each improvement independently and combined.
Logs per-step prior/data gradient ratio for analysis.
"""
import sys, os, math, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from omegaconf import OmegaConf
from diffusers.models.embeddings import get_2d_rotary_pos_embed
from diffusers.utils.torch_utils import randn_tensor

from inpaintingStart import get_operator
from ms_posterior_sampling_article_version_final_utils import (
    apply_G, apply_H_tau, compute_sigma_tau, direct_estimate_x1,
    make_Ak_fns, make_velocity_fn, sample_block_noise,
)
from pixelflow.scheduling_pixelflow import PixelFlowScheduler
from pixelflow.utils import config as config_utils
from pixelflow.utils.misc import seed_everything

# Import v2 Langevin
from debug_IP3.langevin_v2 import langevin_sample_v2

DEVICE = "cuda:0"
RESULT_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULT_DIR, exist_ok=True)


def run_pipeline(model, config, gt, y, operator, sigma_n, label="", **kw):
    B = gt.shape[0]
    seed_everything(20000120)
    scheduler = PixelFlowScheduler(config.scheduler.num_train_timesteps,
                                   num_stages=config.scheduler.num_stages, gamma=-1/3)
    prompt_embeds = torch.tensor([10]*B, dtype=torch.int32, device=DEVICE)
    mask = operator.get_mask(x=gt).float().to(DEVICE)

    num_langevin = kw.get("num_langevin", 10)
    h_x = kw.get("h_x", 0.1)
    h_eps = kw.get("h_eps", 0.01)
    lambda_reg = kw.get("lambda_reg", 50.0)
    sigma_ref = kw.get("sigma_ref", 0.1)
    max_score_norm = kw.get("max_score_norm", 100.0)
    rho_s_tau_scaling = kw.get("rho_s_tau_scaling", True)
    x1_init_mode = kw.get("x1_init_mode", "model")

    h, w = 32, 32
    x1_k = torch.randn(B, 3, h, w, device=DEVICE)
    eps_k = randn_tensor((B, 3, h, w), device=DEVICE)
    lat = eps_k.clone()

    all_residuals = []
    all_logs = []  # per-step v2 logs

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

        for step_idx, T in enumerate(sc.Timesteps):
            tau = float(sc.t[step_idx].to(DEVICE))
            sig = compute_sigma_tau(tau, sk, ek)
            xtau = apply_H_tau(x1_k, tau, sk, ek, stage_idx=si) + sig*eps_k
            vfn = make_velocity_fn(model, T, prompt_embeds, st, rope, False, 0.0, si)

            # Warm restart
            with torch.no_grad(): mu = vfn(xtau)
            x1_k = direct_estimate_x1(xtau - tau*mu, xtau + (1-tau)*mu, sk, ek).detach()
            if sig > 1e-8:
                eps_k = (xtau - apply_H_tau(x1_k, tau, sk, ek, stage_idx=si))/sig

            # v2 Langevin
            step_logs = []
            if sig >= 0.01:
                x1_k, eps_k, _, step_logs = langevin_sample_v2(
                    x1_init=x1_k, eps_init=eps_k, tau=tau, s_k=sk, e_k=ek,
                    velocity_fn=vfn, A_k_fn=Ak, AT_k_fn=ATk,
                    y=y, sigma_n=sigma_n, h_x=h_x, h_epsilon=h_eps,
                    lambda_x=0.01, lambda_reg=lambda_reg, rho_s=1.0, rho_e=1.0,
                    cg_tol=1e-5, cg_max_iter=50, num_Langevin=num_langevin,
                    device=DEVICE, stage_idx=si, x1_init_mode=x1_init_mode,
                    noise_scale=0.0,
                    sigma_ref=sigma_ref,
                    max_score_norm=max_score_norm,
                    rho_s_tau_scaling=rho_s_tau_scaling,
                    return_traj=True, record_every=5)

            lat = apply_H_tau(x1_k, tau, sk, ek, stage_idx=si) + sig*eps_k

            if h == 256:
                res = (y - Ak(x1_k)).pow(2).sum().item()
            else:
                xu = F.interpolate(x1_k, size=(256,256), mode="bilinear", align_corners=False)
                Af, _ = make_Ak_fns(operator, y, (B, 3, 256, 256), DEVICE)
                res = (y - Af(xu)).pow(2).sum().item()
            all_residuals.append(res)

            # Save last inner log for this step
            if step_logs:
                step_logs[-1]["stage"] = si
                step_logs[-1]["step"] = step_idx
                step_logs[-1]["tau"] = tau
                step_logs[-1]["sigma_tau"] = sig
                all_logs.append(step_logs[-1])

    xf = x1_k.detach()
    psnr_all = -10*torch.log10(((xf - gt)**2).mean()).item()
    psnr_obs = -10*torch.log10(((mask*xf - mask*gt)**2).mean()).item()
    final_res = all_residuals[-1]

    return xf.cpu(), all_residuals, psnr_obs, psnr_all, final_res, all_logs


def main():
    print("Loading...", flush=True)
    # Load GT from available source
    pt_candidates = [
        "article_v_results/test_box_inpainting/test_box_inpainting_mode-f_x1.pt",
        "trajectory_videos/posterior_sampling/principle_final/principle_final_mode-f_x1.pt",
        "trajectory_videos/posterior_sampling/principle_final_box/principle_final_box_mode-f_x1.pt",
        "trajectory_videos/posterior_sampling/baseline_05/baseline_05_mode-f_x1.pt",
    ]
    gt = None
    for pp in pt_candidates:
        if os.path.exists(pp):
            d = torch.load(pp, map_location="cpu", weights_only=False)
            gt = d["gt"][:2].to(DEVICE)
            print(f"  Loaded GT from {pp}", flush=True)
            break
    if gt is None:
        # Load from ImageNet directly
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
        class_indices = [i for i, (_, y_label) in enumerate(dataset.samples) if int(y_label) == 10]
        perm = torch.randperm(len(class_indices))[:2]
        selected = [class_indices[i] for i in perm.tolist()]
        gt = torch.stack([dataset[i][0] for i in selected], dim=0).to(DEVICE)
        print(f"  Loaded GT from ImageNet ({data_dir})", flush=True)
    config = OmegaConf.load("pretrained_models/c2img/config.yaml")
    model = config_utils.instantiate_from_config(config.model).to(DEVICE)
    ckpt = torch.load("pretrained_models/c2img/model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt, strict=True); model.eval()
    print("Model loaded.", flush=True)

    sigma_n = 0.05

    results = {}

    for mask_type, op_kw in [
        ("box", dict(mask_type="box", mask_len_range=(80,160), mask_prob_range=None)),
        ("random", dict(mask_type="random", mask_len_range=None, mask_prob_range=(0.8,0.8))),
    ]:
        operator = get_operator("inpainting", resolution=256, device=DEVICE, sigma=sigma_n, **op_kw)
        y = operator(gt).detach()

        print(f"\n{'='*60}", flush=True)
        print(f"  {mask_type.upper()} INPAINTING", flush=True)
        print(f"{'='*60}", flush=True)

        ablations = [
            # Baseline v1 (current best): no tau scaling, hard floor via large sigma_ref=999
            ("v1_baseline", dict(sigma_ref=999.0, max_score_norm=1e6, rho_s_tau_scaling=False)),

            # Individual improvements
            ("v2a_soft_damp_0.1", dict(sigma_ref=0.1, max_score_norm=1e6, rho_s_tau_scaling=False)),
            ("v2b_tau_rho", dict(sigma_ref=999.0, max_score_norm=1e6, rho_s_tau_scaling=True)),
            ("v2c_score_clip", dict(sigma_ref=999.0, max_score_norm=100.0, rho_s_tau_scaling=False)),

            # Pairwise combinations
            ("v2ab_damp+rho", dict(sigma_ref=0.1, max_score_norm=1e6, rho_s_tau_scaling=True)),
            ("v2ac_damp+clip", dict(sigma_ref=0.1, max_score_norm=100.0, rho_s_tau_scaling=False)),

            # Full v2
            ("v2_full", dict(sigma_ref=0.1, max_score_norm=100.0, rho_s_tau_scaling=True)),

            # sigma_ref sweep
            ("v2_sref0.05", dict(sigma_ref=0.05, max_score_norm=100.0, rho_s_tau_scaling=True)),
            ("v2_sref0.2", dict(sigma_ref=0.2, max_score_norm=100.0, rho_s_tau_scaling=True)),
            ("v2_sref0.3", dict(sigma_ref=0.3, max_score_norm=100.0, rho_s_tau_scaling=True)),
        ]

        for name, kw in ablations:
            print(f"  {name}...", flush=True)
            x1, res, po, pa, fr, logs = run_pipeline(
                model, config, gt, y, operator, sigma_n, **kw)
            results[f"{mask_type}_{name}"] = (x1, res, po, pa, fr, logs)
            # Print with prior/data ratio from last stage
            avg_ratio = np.mean([l["prior_data_ratio"] for l in logs if l["stage"] == 3]) if logs else 0
            print(f"    res={fr:.0f}  PSNR_obs={po:.2f}  PSNR_all={pa:.2f}  "
                  f"avg_prior/data_S3={avg_ratio:.2f}", flush=True)

    # ══ Summary ══
    print(f"\n{'='*80}", flush=True)
    print(f"{'Config':<30} {'Residual':>8} {'PSNR_obs':>9} {'PSNR_all':>9} {'P/D S3':>7}", flush=True)
    print("-"*68, flush=True)
    for k in sorted(results.keys()):
        _, _, po, pa, fr, logs = results[k]
        avg_ratio = np.mean([l["prior_data_ratio"] for l in logs if l["stage"] == 3]) if logs else 0
        print(f"{k:<30} {fr:>8.0f} {po:>9.2f} {pa:>9.2f} {avg_ratio:>7.2f}", flush=True)

    # ══ Visualization ══
    for mask_type in ["box", "random"]:
        keys = sorted([k for k in results if k.startswith(mask_type)])
        n = len(keys)
        cols = min(6, n+1)
        rows = (n+1+cols-1)//cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols*4.5, rows*4.5))
        if rows == 1: axes = axes[np.newaxis, :]
        axes = axes.flatten()

        img_gt = (gt[0].cpu().permute(1,2,0)*0.5+0.5).clamp(0,1).numpy()
        axes[0].imshow(img_gt); axes[0].set_title("GT"); axes[0].axis("off")
        for i, k in enumerate(keys):
            x1, _, po, pa, fr, _ = results[k]
            img = (x1[0].permute(1,2,0)*0.5+0.5).clamp(0,1).numpy()
            short = k.replace(f"{mask_type}_", "")
            axes[i+1].imshow(img)
            axes[i+1].set_title(f"{short}\n{pa:.2f}dB", fontsize=8)
            axes[i+1].axis("off")
        for j in range(i+2, len(axes)): axes[j].axis("off")
        plt.suptitle(f"{mask_type} — v2 ablation", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{RESULT_DIR}/ablation_{mask_type}.png", dpi=150)
        plt.close()
        print(f"Saved ablation_{mask_type}.png", flush=True)

    # ══ Prior/Data ratio plot ══
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for idx, mask_type in enumerate(["box", "random"]):
        for k in sorted(results.keys()):
            if not k.startswith(mask_type): continue
            _, _, _, _, _, logs = results[k]
            if not logs: continue
            steps = [l["stage"]*10 + l["step"] for l in logs]
            ratios = [l["prior_data_ratio"] for l in logs]
            short = k.replace(f"{mask_type}_", "")
            axes[idx].plot(steps, ratios, "-o", markersize=2, label=short, alpha=0.7)
        axes[idx].set_yscale("log")
        axes[idx].set_xlabel("Step"); axes[idx].set_ylabel("prior/data gradient ratio")
        axes[idx].set_title(f"{mask_type}: prior vs data gradient balance")
        axes[idx].legend(fontsize=6, ncol=2); axes[idx].grid(True, alpha=0.3)
        for si in range(4): axes[idx].axvline(x=si*10, color="gray", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/prior_data_ratio.png", dpi=150)
    plt.close()
    print("Saved prior_data_ratio.png", flush=True)

    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()
