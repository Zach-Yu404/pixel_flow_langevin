#!/usr/bin/env python
"""
debug5/test_hybrid_stage3.py — Test fundamentally different stage-3 strategies.

The visual analysis shows ALL PRINCIPLE configs produce smooth box regions.
The CG-Langevin inner loop is the root cause: it smooths model predictions.

Strategy: Use PRINCIPLE for stages 0-2 (data consistency at low res),
then switch to model-trusting methods at stage 3 (texture generation):

  A) No Langevin at stage 3 (ODE flow only)
  B) Replacement at stage 3 (observed=y, unobserved=model_prediction)
  C) DPS at stage 3 (gradient correction, no inner loop)
  D) L=1 minimal Langevin at stage 3
  E) Replacement + terminal replace
  F) DPS + replacement hybrid
  G) Tikhonov one-shot at stage 3 (single CG, no iteration)
  H) Very high lambda_prox + L=1 (anchor dominates, minimal smoothing)

Runs on cuda:1 to avoid interfering with sweep on cuda:0.
Saves per-config images with box-region crops for visual analysis.
"""
import sys, os, math, copy, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "debug_IP4"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from diffusers.models.embeddings import get_2d_rotary_pos_embed
from diffusers.utils.torch_utils import randn_tensor
from inpaintingStart import get_operator
from ms_posterior_sampling_article_version_final_utils import (
    apply_G, apply_H_tau, compute_sigma_tau, direct_estimate_x1,
    make_Ak_fns, make_velocity_fn, sample_block_noise, cg_solve,
)
from ms_sampler_v5 import hf_energy
from pixelflow.scheduling_pixelflow import PixelFlowScheduler
from pixelflow.utils import config as config_utils
from pixelflow.utils.misc import seed_everything

DEVICE = "cuda:1"
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_hybrid")
os.makedirs(RESULT_DIR, exist_ok=True)


def to_img(t):
    return (t.cpu().permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1).numpy()


def find_box(mask):
    inv = (1 - mask[0, 0].cpu()).numpy()
    rows = np.any(inv > 0.5, axis=1)
    cols = np.any(inv > 0.5, axis=0)
    if not rows.any():
        return 0, 255, 0, 255
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    pad = 5
    return max(0, rmin - pad), min(255, rmax + pad), max(0, cmin - pad), min(255, cmax + pad)


def load_gt(device):
    pt = "trajectory_videos/posterior_sampling/baseline_05/baseline_05_mode-f_x1.pt"
    if os.path.exists(pt):
        d = torch.load(pt, map_location="cpu", weights_only=False)
        return d["gt"][:2].to(device)
    from torchvision import transforms, datasets
    from ms_posterior_sampling_article_version_final_utils import center_crop_arr
    data_dir = "/data/Zach_dataset/imageNet256/ILSVRC/Data/CLS-LOC/train/"
    tf = transforms.Compose([
        transforms.Lambda(lambda p: center_crop_arr(p, 256)),
        transforms.ToTensor(),
        transforms.Normalize([.5]*3, [.5]*3, inplace=True),
    ])
    ds = datasets.ImageFolder(data_dir, tf)
    torch.manual_seed(20000120)
    ci = [i for i, (_, y) in enumerate(ds.samples) if int(y) == 10]
    perm = torch.randperm(len(ci))[:2]
    return torch.stack([ds[ci[i]][0] for i in perm.tolist()]).to(device)


@torch.no_grad()
def run_hybrid(model, config, gt, y, operator, sigma_n, device,
               stage3_mode="principle",  # principle, none, replace, dps, tikhonov, l1_prox
               num_langevin_s012=10, num_langevin_s3=10,
               h_x=0.1, h_eps=0.01, lambda_reg=50.0, lambda_prox=0.0,
               cg_max_iter=20, dps_zeta=5.0, tikhonov_lambda=100.0,
               seed=20000120):
    """Run posterior sampling with configurable stage-3 strategy."""
    from langevin_v5 import principle_langevin_v5, terminal_replacement

    B = gt.shape[0]
    seed_everything(seed)
    torch.manual_seed(seed)

    num_stages = int(config.scheduler.num_stages)
    scheduler = PixelFlowScheduler(
        config.scheduler.num_train_timesteps,
        num_stages=num_stages, gamma=-1 / 3,
    )

    pe_labels = torch.tensor([10] * B, dtype=torch.int32, device=device)
    mask_full = operator.get_mask(x=gt).float().to(device)

    h = w = 32
    x1_k = torch.randn(B, 3, h, w, device=device)
    eps_k = randn_tensor((B, 3, h, w), device=device)
    lat = eps_k.clone()
    t0 = time.time()

    for si in range(num_stages):
        sc = copy.deepcopy(scheduler)
        sc.set_timesteps(10, si, device=device, shift=1.0)
        sk = float(sc.start_t[si])
        ek = float(sc.end_t[si])
        eff_si = si  # g_bypass_stage3

        if si > 0:
            h *= 2; w *= 2
            lat = F.interpolate(lat, size=(h, w), mode="nearest")
            ost = sc.original_start_t[si]; gam = sc.gamma
            al = 1 / (math.sqrt(1 - (1 / gam)) * (1 - ost) + ost)
            be = al * (1 - ost) / math.sqrt(-gam)
            nz = sample_block_noise(sc, B, 3, h, w).to(device=device, dtype=lat.dtype)
            lat = al * lat + be * nz
            x1_k = F.interpolate(x1_k, size=(h, w), mode="nearest")
            eps_k = (lat - sk * apply_G(x1_k, stage_idx=eff_si)) / max(1 - sk, 1e-8)

        Ak, ATk = make_Ak_fns(operator, y, (B, 3, h, w), device)

        if h == 256:
            mask_k = mask_full
        else:
            mask_k = F.interpolate(mask_full, size=(h, w), mode="nearest")

        st = torch.tensor([h // model.patch_size], dtype=torch.int32, device=device)
        pe = get_2d_rotary_pos_embed(
            embed_dim=model.attention_head_dim,
            crops_coords=((0, 0), (h // model.patch_size, w // model.patch_size)),
            grid_size=(h // model.patch_size, w // model.patch_size),
            device=device, output_type="pt",
        )
        rope = torch.stack(pe, -1)

        is_stage3 = (si == num_stages - 1)

        for step_idx, T in enumerate(sc.Timesteps):
            tau = float(sc.t[step_idx].to(device))
            sig = compute_sigma_tau(tau, sk, ek)
            xtau = apply_H_tau(x1_k, tau, sk, ek, stage_idx=eff_si) + sig * eps_k

            vfn = make_velocity_fn(model, T, pe_labels, st, rope, False, 0.0, si)

            # Always get model prediction (warm restart)
            mu = vfn(xtau)
            xs_h = xtau - tau * mu
            xe_h = xtau + (1 - tau) * mu
            x1_model = direct_estimate_x1(xs_h, xe_h, sk, ek).detach().clone()
            if sig > 1e-8:
                eps_k = (xtau - apply_H_tau(x1_model, tau, sk, ek, stage_idx=eff_si)) / sig
            else:
                eps_k = torch.randn_like(x1_k)

            x1_k = x1_model

            if is_stage3 and sig >= 0.01:
                # ---- Stage-3 specific strategy ----
                if stage3_mode == "none":
                    # No inner loop — just use model prediction directly
                    pass

                elif stage3_mode == "replace":
                    # Replace: observed=y, unobserved=model_prediction
                    x1_k = mask_k * y + (1 - mask_k) * x1_k

                elif stage3_mode == "dps":
                    # DPS gradient correction (no inner loop)
                    residual = y - Ak(x1_k)
                    res_norm = residual.reshape(B, -1).norm(dim=1).reshape(B, 1, 1, 1).clamp(min=1e-8)
                    x1_k = x1_k + dps_zeta * ATk(residual) / res_norm

                elif stage3_mode == "replace_dps":
                    # DPS correction then replacement
                    residual = y - Ak(x1_k)
                    res_norm = residual.reshape(B, -1).norm(dim=1).reshape(B, 1, 1, 1).clamp(min=1e-8)
                    x1_k = x1_k + dps_zeta * ATk(residual) / res_norm
                    x1_k = mask_k * y + (1 - mask_k) * x1_k

                elif stage3_mode == "tikhonov":
                    # Tikhonov one-shot: (A^TA/η² + λI)x = A^Ty/η² + λ*x1_model
                    eta2 = sigma_n ** 2
                    lam = tikhonov_lambda
                    def system(x):
                        return (1.0 / eta2) * ATk(Ak(x)) + lam * x
                    rhs = (1.0 / eta2) * ATk(y) + lam * x1_model
                    x1_k = cg_solve(system, rhs, x0=x1_model.clone(), tol=1e-5, max_iter=50)

                elif stage3_mode == "l1_prox":
                    # L=1 Langevin with high lambda_prox (minimal smoothing, strong anchor)
                    x1_k, eps_k = principle_langevin_v5(
                        x1_init=x1_k, eps_init=eps_k,
                        tau=tau, s_k=sk, e_k=ek,
                        velocity_fn=vfn, A_k_fn=Ak, AT_k_fn=ATk,
                        y=y, sigma_n=sigma_n,
                        h_x=h_x, h_epsilon=h_eps,
                        lambda_x=0.01, lambda_reg=lambda_reg,
                        rho_s=1.0, rho_e=1.0,
                        cg_tol=1e-5, cg_max_iter=cg_max_iter,
                        num_Langevin=num_langevin_s3, device=device,
                        stage_idx=eff_si, x1_init_mode="model",
                        noise_scale=0.0,
                        mask_k=mask_k, h_x_obs_ratio=1.0,
                        lambda_prox=lambda_prox,
                    )

                elif stage3_mode == "principle":
                    # Standard PRINCIPLE (baseline)
                    x1_k, eps_k = principle_langevin_v5(
                        x1_init=x1_k, eps_init=eps_k,
                        tau=tau, s_k=sk, e_k=ek,
                        velocity_fn=vfn, A_k_fn=Ak, AT_k_fn=ATk,
                        y=y, sigma_n=sigma_n,
                        h_x=0.1, h_epsilon=0.01,
                        lambda_x=0.01, lambda_reg=50.0,
                        rho_s=1.0, rho_e=1.0,
                        cg_tol=1e-5, cg_max_iter=cg_max_iter,
                        num_Langevin=num_langevin_s012, device=device,
                        stage_idx=eff_si, x1_init_mode="model",
                        noise_scale=0.0,
                        mask_k=mask_k, h_x_obs_ratio=1.0,
                        lambda_prox=0.0,
                    )

                # Update eps_k after stage-3 modification
                if sig > 1e-8:
                    eps_k = (xtau - apply_H_tau(x1_k, tau, sk, ek, stage_idx=eff_si)) / sig

            elif not is_stage3 and sig >= 0.01:
                # Stages 0-2: always use PRINCIPLE (data consistency)
                from langevin_v5 import principle_langevin_v5
                x1_k, eps_k = principle_langevin_v5(
                    x1_init=x1_k, eps_init=eps_k,
                    tau=tau, s_k=sk, e_k=ek,
                    velocity_fn=vfn, A_k_fn=Ak, AT_k_fn=ATk,
                    y=y, sigma_n=sigma_n,
                    h_x=0.1, h_epsilon=0.01,
                    lambda_x=0.01, lambda_reg=50.0,
                    rho_s=1.0, rho_e=1.0,
                    cg_tol=1e-5, cg_max_iter=20,
                    num_Langevin=num_langevin_s012, device=device,
                    stage_idx=eff_si, x1_init_mode="model",
                    noise_scale=0.0,
                )

            lat = apply_H_tau(x1_k, tau, sk, ek, stage_idx=eff_si) + sig * eps_k

    # Terminal replacement
    x1_k = mask_full * y + (1 - mask_full) * x1_k

    elapsed = time.time() - t0
    xf = x1_k.detach()
    Af, _ = make_Ak_fns(operator, y, (B, 3, 256, 256), device)
    psnr_all = -10 * torch.log10(((xf - gt) ** 2).mean()).item()
    res = (y - Af(xf)).pow(2).sum().item()
    hf_u = hf_energy(xf * (1 - mask_full[0:1]))
    return xf.cpu(), psnr_all, res, elapsed, hf_u


CONFIGS = [
    ("A_principle",   dict(stage3_mode="principle")),
    ("B_no_inner",    dict(stage3_mode="none")),
    ("C_replace",     dict(stage3_mode="replace")),
    ("D_dps_z1",      dict(stage3_mode="dps", dps_zeta=1.0)),
    ("E_dps_z5",      dict(stage3_mode="dps", dps_zeta=5.0)),
    ("F_dps_z10",     dict(stage3_mode="dps", dps_zeta=10.0)),
    ("G_rep_dps_z5",  dict(stage3_mode="replace_dps", dps_zeta=5.0)),
    ("H_tikh_l50",    dict(stage3_mode="tikhonov", tikhonov_lambda=50.0)),
    ("I_tikh_l100",   dict(stage3_mode="tikhonov", tikhonov_lambda=100.0)),
    ("J_tikh_l200",   dict(stage3_mode="tikhonov", tikhonov_lambda=200.0)),
    ("K_l1_lp50",     dict(stage3_mode="l1_prox", num_langevin_s3=1, lambda_prox=50.0, h_x=0.7)),
    ("L_l1_lp100",    dict(stage3_mode="l1_prox", num_langevin_s3=1, lambda_prox=100.0, h_x=0.7)),
    ("M_l1_lp200",    dict(stage3_mode="l1_prox", num_langevin_s3=1, lambda_prox=200.0, h_x=1.0)),
    ("N_l2_lp100",    dict(stage3_mode="l1_prox", num_langevin_s3=2, lambda_prox=100.0, h_x=0.7)),
    ("O_l3_lp50",     dict(stage3_mode="l1_prox", num_langevin_s3=3, lambda_prox=50.0, h_x=0.5)),
]


def main():
    print(f"Hybrid stage-3 test on {DEVICE}  ({len(CONFIGS)} configs)", flush=True)
    print("Loading model...", flush=True)

    gt = load_gt(DEVICE)
    config = OmegaConf.load("pretrained_models/c2img/config.yaml")
    model = config_utils.instantiate_from_config(config.model).to(DEVICE)
    ckpt = torch.load("pretrained_models/c2img/model.pt",
                      map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt, strict=True)
    model.eval()
    print("Model loaded.\n", flush=True)

    sigma_n = 0.05
    operator = get_operator(
        "inpainting", resolution=256, device=DEVICE, sigma=sigma_n,
        mask_type="box", mask_len_range=(80, 160), mask_prob_range=None,
    )
    y = operator(gt).detach()
    mask = operator.get_mask(x=gt).float().to(DEVICE)
    gt_hf = hf_energy(gt * (1 - mask[0:1]))
    rmin, rmax, cmin, cmax = find_box(mask)
    print(f"BOX [{rmin}:{rmax}, {cmin}:{cmax}]  GT HF={gt_hf:.3f}\n", flush=True)

    results = {}
    for idx, (name, kw) in enumerate(CONFIGS, 1):
        print(f"  {idx}/{len(CONFIGS)} {name}...", end=" ", flush=True)
        try:
            xf, pa, res, t, hf = run_hybrid(
                model, config, gt, y, operator, sigma_n, DEVICE, **kw)
            dhf = abs(hf - gt_hf)
            results[name] = dict(x=xf, psnr=pa, hf=hf, dhf=dhf, res=res, time=t)
            print(f"PSNR={pa:.2f}  HF={hf:.3f}  |dHF|={dhf:.3f}  res={res:.0f}  t={t:.0f}s",
                  flush=True)
        except Exception as e:
            import traceback
            print(f"ERROR: {e}", flush=True)
            traceback.print_exc()

    # ---- Box-region grid (the key visualization) ----
    n = len(results)
    fig, axes = plt.subplots(3, n + 1, figsize=((n + 1) * 2.5, 7.5))

    gt_img = to_img(gt[0])
    gt_img2 = to_img(gt[1])

    axes[0, 0].imshow(gt_img)
    axes[0, 0].set_title(f"GT\nHF={gt_hf:.3f}", fontsize=6)
    axes[0, 0].axis("off")
    axes[1, 0].imshow(gt_img[rmin:rmax+1, cmin:cmax+1])
    axes[1, 0].set_title("GT box", fontsize=6)
    axes[1, 0].axis("off")
    axes[2, 0].imshow(gt_img2[rmin:rmax+1, cmin:cmax+1])
    axes[2, 0].set_title("GT box img2", fontsize=6)
    axes[2, 0].axis("off")

    for i, (name, v) in enumerate(results.items()):
        xf_img = to_img(v["x"][0])
        xf_img2 = to_img(v["x"][1])

        axes[0, i + 1].imshow(xf_img)
        axes[0, i + 1].set_title(f"{name}\nP={v['psnr']:.1f} r={v['res']:.0f}", fontsize=5)
        axes[0, i + 1].axis("off")

        axes[1, i + 1].imshow(xf_img[rmin:rmax+1, cmin:cmax+1])
        axes[1, i + 1].set_title(f"hf={v['hf']:.3f} |d|={v['dhf']:.3f}", fontsize=5)
        axes[1, i + 1].axis("off")

        axes[2, i + 1].imshow(xf_img2[rmin:rmax+1, cmin:cmax+1])
        axes[2, i + 1].set_title(f"t={v['time']:.0f}s", fontsize=5)
        axes[2, i + 1].axis("off")

    plt.suptitle(f"Hybrid stage-3 strategies (GT HF={gt_hf:.3f})\n"
                 f"Row 1: full image | Row 2: box crop img1 | Row 3: box crop img2",
                 fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(f"{RESULT_DIR}/hybrid_grid.png", dpi=200)
    plt.close()

    # Per-config detail images
    for name, v in results.items():
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for img_idx in range(2):
            gt_i = to_img(gt[img_idx])
            xf_i = to_img(v["x"][img_idx])
            y_i = to_img((mask[0:1] * gt)[img_idx])

            axes[img_idx, 0].imshow(gt_i); axes[img_idx, 0].set_title("GT", fontsize=8); axes[img_idx, 0].axis("off")
            axes[img_idx, 1].imshow(y_i); axes[img_idx, 1].set_title("Meas", fontsize=8); axes[img_idx, 1].axis("off")
            axes[img_idx, 2].imshow(xf_i); axes[img_idx, 2].set_title(name, fontsize=8); axes[img_idx, 2].axis("off")
            crop_gt = gt_i[rmin:rmax+1, cmin:cmax+1]
            crop_xf = xf_i[rmin:rmax+1, cmin:cmax+1]
            combined = np.concatenate([crop_gt, np.ones((crop_gt.shape[0], 3, 3)), crop_xf], axis=1)
            axes[img_idx, 3].imshow(combined)
            axes[img_idx, 3].set_title("GT | Result", fontsize=8)
            axes[img_idx, 3].axis("off")

        plt.suptitle(f"{name} — PSNR={v['psnr']:.2f}  HF={v['hf']:.3f}  "
                     f"|dHF|={v['dhf']:.3f}  res={v['res']:.0f}", fontsize=10)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f"{RESULT_DIR}/{name}.png", dpi=150)
        plt.close()

    print(f"\nSaved {RESULT_DIR}/hybrid_grid.png")
    print(f"Saved {len(results)} per-config images")
    print("DONE.", flush=True)


if __name__ == "__main__":
    main()
