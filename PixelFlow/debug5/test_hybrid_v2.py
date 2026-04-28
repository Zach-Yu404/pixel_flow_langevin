#!/usr/bin/env python
"""
debug5/test_hybrid_v2.py — Hybrid stage-3 WITHOUT warm restart.

Key insight from v1: warm restart at stage 3 overwrites DPS/replace
corrections, so they never accumulate. v2 tests:
  - Skip warm restart at stage 3 → let corrections accumulate across ODE steps
  - Model velocity is still evaluated, but x1_k is NOT overwritten
  - DPS, replace, and hybrid corrections build up over 10 steps

Runs on cuda:1 (or any free GPU).
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

DEVICE = "cuda:2"
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_hybrid_v2")
os.makedirs(RESULT_DIR, exist_ok=True)


def to_img(t):
    return (t.cpu().permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1).numpy()

def find_box(mask):
    inv = (1 - mask[0, 0].cpu()).numpy()
    rows = np.any(inv > 0.5, axis=1)
    cols = np.any(inv > 0.5, axis=0)
    if not rows.any(): return 0, 255, 0, 255
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    pad = 5
    return max(0, rmin-pad), min(255, rmax+pad), max(0, cmin-pad), min(255, cmax+pad)

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
def run_hybrid_v2(model, config, gt, y, operator, sigma_n, device,
                  stage3_mode="principle",
                  warmrestart_s3=True,  # NEW: skip warm restart at stage 3
                  dps_zeta=5.0, tikhonov_lambda=100.0,
                  replace_weight=1.0,  # blending weight for replacement
                  seed=20000120):
    from langevin_v5 import principle_langevin_v5

    B = gt.shape[0]
    seed_everything(seed); torch.manual_seed(seed)

    num_stages = int(config.scheduler.num_stages)
    scheduler = PixelFlowScheduler(
        config.scheduler.num_train_timesteps, num_stages=num_stages, gamma=-1/3)

    pe_labels = torch.tensor([10]*B, dtype=torch.int32, device=device)
    mask_full = operator.get_mask(x=gt).float().to(device)

    h = w = 32
    x1_k = torch.randn(B, 3, h, w, device=device)
    eps_k = randn_tensor((B, 3, h, w), device=device)
    lat = eps_k.clone()
    t0 = time.time()

    for si in range(num_stages):
        sc = copy.deepcopy(scheduler)
        sc.set_timesteps(10, si, device=device, shift=1.0)
        sk, ek = float(sc.start_t[si]), float(sc.end_t[si])
        eff_si = si

        if si > 0:
            h *= 2; w *= 2
            lat = F.interpolate(lat, size=(h, w), mode="nearest")
            ost = sc.original_start_t[si]; gam = sc.gamma
            al = 1/(math.sqrt(1-(1/gam))*(1-ost)+ost)
            be = al*(1-ost)/math.sqrt(-gam)
            nz = sample_block_noise(sc, B, 3, h, w).to(device=device, dtype=lat.dtype)
            lat = al*lat + be*nz
            x1_k = F.interpolate(x1_k, size=(h, w), mode="nearest")
            eps_k = (lat - sk*apply_G(x1_k, stage_idx=eff_si))/max(1-sk, 1e-8)

        Ak, ATk = make_Ak_fns(operator, y, (B, 3, h, w), device)
        mask_k = mask_full if h == 256 else F.interpolate(mask_full, size=(h, w), mode="nearest")

        st = torch.tensor([h//model.patch_size], dtype=torch.int32, device=device)
        pe = get_2d_rotary_pos_embed(
            embed_dim=model.attention_head_dim,
            crops_coords=((0,0),(h//model.patch_size, w//model.patch_size)),
            grid_size=(h//model.patch_size, w//model.patch_size),
            device=device, output_type="pt")
        rope = torch.stack(pe, -1)

        is_stage3 = (si == num_stages - 1)

        for step_idx, T in enumerate(sc.Timesteps):
            tau = float(sc.t[step_idx].to(device))
            sig = compute_sigma_tau(tau, sk, ek)
            xtau = apply_H_tau(x1_k, tau, sk, ek, stage_idx=eff_si) + sig*eps_k

            vfn = make_velocity_fn(model, T, pe_labels, st, rope, False, 0.0, si)

            # Get model prediction
            mu = vfn(xtau)
            xs_h = xtau - tau*mu
            xe_h = xtau + (1-tau)*mu
            x1_model = direct_estimate_x1(xs_h, xe_h, sk, ek).detach().clone()

            if is_stage3 and not warmrestart_s3:
                # ---- NO warm restart: keep accumulated x1_k ----
                # x1_model is available as reference but doesn't overwrite x1_k
                pass
            else:
                # Standard warm restart
                x1_k = x1_model
                if sig > 1e-8:
                    eps_k = (xtau - apply_H_tau(x1_k, tau, sk, ek, stage_idx=eff_si))/sig
                else:
                    eps_k = torch.randn_like(x1_k)

            if is_stage3 and sig >= 0.01:
                if stage3_mode == "principle":
                    x1_k, eps_k = principle_langevin_v5(
                        x1_init=x1_k, eps_init=eps_k,
                        tau=tau, s_k=sk, e_k=ek,
                        velocity_fn=vfn, A_k_fn=Ak, AT_k_fn=ATk,
                        y=y, sigma_n=sigma_n,
                        h_x=0.1, h_epsilon=0.01,
                        lambda_x=0.01, lambda_reg=50.0,
                        rho_s=1.0, rho_e=1.0,
                        cg_tol=1e-5, cg_max_iter=20,
                        num_Langevin=10, device=device,
                        stage_idx=eff_si, x1_init_mode="model",
                        noise_scale=0.0)

                elif stage3_mode == "none":
                    # Use x1_model directly (with or without warm restart)
                    if not warmrestart_s3:
                        x1_k = x1_model

                elif stage3_mode == "replace":
                    # Replace observed pixels; unobserved = model or accumulated
                    if not warmrestart_s3:
                        # Blend: unobserved from model, observed from y
                        x1_k = mask_k * y + (1-mask_k) * x1_model
                    else:
                        x1_k = mask_k * y + (1-mask_k) * x1_k

                elif stage3_mode == "dps":
                    # DPS: gradient correction on ACCUMULATED x1_k
                    if not warmrestart_s3:
                        # x1_k carries corrections from previous steps
                        residual = y - Ak(x1_k)
                    else:
                        residual = y - Ak(x1_k)
                    res_norm = residual.reshape(B,-1).norm(dim=1).reshape(B,1,1,1).clamp(min=1e-8)
                    x1_k = x1_k + dps_zeta * ATk(residual) / res_norm

                elif stage3_mode == "dps_replace":
                    # DPS correction + replacement
                    residual = y - Ak(x1_k)
                    res_norm = residual.reshape(B,-1).norm(dim=1).reshape(B,1,1,1).clamp(min=1e-8)
                    x1_k = x1_k + dps_zeta * ATk(residual) / res_norm
                    x1_k = mask_k * y + (1-mask_k) * x1_k

                elif stage3_mode == "model_replace":
                    # Use model prediction for unobserved, y for observed
                    # (different from "replace" when warmrestart=False: always uses fresh model)
                    x1_k = mask_k * y + (1-mask_k) * x1_model

                elif stage3_mode == "blend":
                    # Blend model prediction with accumulated x1_k
                    w = replace_weight
                    x1_k = mask_k * y + (1-mask_k) * (w * x1_model + (1-w) * x1_k)

                elif stage3_mode == "tikhonov":
                    eta2 = sigma_n ** 2
                    lam = tikhonov_lambda
                    def system(x):
                        return (1.0/eta2)*ATk(Ak(x)) + lam*x
                    rhs = (1.0/eta2)*ATk(y) + lam*x1_model
                    x1_k = cg_solve(system, rhs, x0=x1_model.clone(), tol=1e-5, max_iter=50)

                # Update eps after modification
                if sig > 1e-8:
                    eps_k = (xtau - apply_H_tau(x1_k, tau, sk, ek, stage_idx=eff_si))/sig

            elif not is_stage3 and sig >= 0.01:
                # Stages 0-2: always PRINCIPLE
                x1_k, eps_k = principle_langevin_v5(
                    x1_init=x1_k, eps_init=eps_k,
                    tau=tau, s_k=sk, e_k=ek,
                    velocity_fn=vfn, A_k_fn=Ak, AT_k_fn=ATk,
                    y=y, sigma_n=sigma_n,
                    h_x=0.1, h_epsilon=0.01,
                    lambda_x=0.01, lambda_reg=50.0,
                    rho_s=1.0, rho_e=1.0,
                    cg_tol=1e-5, cg_max_iter=20,
                    num_Langevin=10, device=device,
                    stage_idx=eff_si, x1_init_mode="model",
                    noise_scale=0.0)

            lat = apply_H_tau(x1_k, tau, sk, ek, stage_idx=eff_si) + sig*eps_k

    # Terminal replacement
    x1_k = mask_full * y + (1-mask_full) * x1_k

    elapsed = time.time() - t0
    xf = x1_k.detach()
    Af, _ = make_Ak_fns(operator, y, (B, 3, 256, 256), device)
    psnr = -10*torch.log10(((xf-gt)**2).mean()).item()
    res = (y - Af(xf)).pow(2).sum().item()
    hf = hf_energy(xf * (1 - mask_full[0:1]))
    return xf.cpu(), psnr, res, elapsed, hf


CONFIGS = [
    # ---- References (with warm restart) ----
    ("REF_principle",       dict(stage3_mode="principle", warmrestart_s3=True)),
    ("REF_none_wr",         dict(stage3_mode="none", warmrestart_s3=True)),

    # ---- NO warm restart at stage 3 (key change) ----
    # DPS without warm restart: corrections accumulate
    ("DPS_nowr_z1",         dict(stage3_mode="dps", warmrestart_s3=False, dps_zeta=1.0)),
    ("DPS_nowr_z5",         dict(stage3_mode="dps", warmrestart_s3=False, dps_zeta=5.0)),
    ("DPS_nowr_z10",        dict(stage3_mode="dps", warmrestart_s3=False, dps_zeta=10.0)),
    ("DPS_nowr_z20",        dict(stage3_mode="dps", warmrestart_s3=False, dps_zeta=20.0)),

    # Replace without warm restart (use fresh model prediction each step)
    ("model_rep_nowr",      dict(stage3_mode="model_replace", warmrestart_s3=False)),

    # DPS + replace without warm restart
    ("DPS_rep_nowr_z5",     dict(stage3_mode="dps_replace", warmrestart_s3=False, dps_zeta=5.0)),
    ("DPS_rep_nowr_z10",    dict(stage3_mode="dps_replace", warmrestart_s3=False, dps_zeta=10.0)),

    # Tikhonov (doesn't need warm restart change since it uses x1_model directly)
    ("tikh_l50",            dict(stage3_mode="tikhonov", warmrestart_s3=True, tikhonov_lambda=50.0)),
    ("tikh_l100",           dict(stage3_mode="tikhonov", warmrestart_s3=True, tikhonov_lambda=100.0)),
    ("tikh_l200",           dict(stage3_mode="tikhonov", warmrestart_s3=True, tikhonov_lambda=200.0)),

    # Blend: weighted mix of model prediction and accumulated x1
    ("blend_nowr_07",       dict(stage3_mode="blend", warmrestart_s3=False, replace_weight=0.7)),
    ("blend_nowr_05",       dict(stage3_mode="blend", warmrestart_s3=False, replace_weight=0.5)),
    ("blend_nowr_03",       dict(stage3_mode="blend", warmrestart_s3=False, replace_weight=0.3)),
]


def main():
    print(f"Hybrid v2 on {DEVICE}  ({len(CONFIGS)} configs)", flush=True)
    print("Loading model...", flush=True)

    gt = load_gt(DEVICE)
    config = OmegaConf.load("pretrained_models/c2img/config.yaml")
    model = config_utils.instantiate_from_config(config.model).to(DEVICE)
    ckpt = torch.load("pretrained_models/c2img/model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt, strict=True); model.eval()
    print("Model loaded.\n", flush=True)

    sigma_n = 0.05
    operator = get_operator("inpainting", resolution=256, device=DEVICE, sigma=sigma_n,
                            mask_type="box", mask_len_range=(80, 160), mask_prob_range=None)
    y = operator(gt).detach()
    mask = operator.get_mask(x=gt).float().to(DEVICE)
    gt_hf = hf_energy(gt * (1 - mask[0:1]))
    rmin, rmax, cmin, cmax = find_box(mask)
    print(f"BOX [{rmin}:{rmax}, {cmin}:{cmax}]  GT HF={gt_hf:.3f}\n", flush=True)

    results = {}
    for idx, (name, kw) in enumerate(CONFIGS, 1):
        print(f"  {idx}/{len(CONFIGS)} {name}...", end=" ", flush=True)
        try:
            xf, pa, res, t, hf = run_hybrid_v2(model, config, gt, y, operator, sigma_n, DEVICE, **kw)
            dhf = abs(hf - gt_hf)
            results[name] = dict(x=xf, psnr=pa, hf=hf, dhf=dhf, res=res, time=t)
            print(f"PSNR={pa:.2f}  HF={hf:.3f}  |dHF|={dhf:.3f}  res={res:.0f}  t={t:.0f}s", flush=True)
        except Exception as e:
            import traceback
            print(f"ERROR: {e}", flush=True)
            traceback.print_exc()

    # ---- Box-region grid ----
    n = len(results)
    fig, axes = plt.subplots(3, n+1, figsize=((n+1)*2.5, 7.5))
    gt_img, gt_img2 = to_img(gt[0]), to_img(gt[1])

    axes[0,0].imshow(gt_img); axes[0,0].set_title(f"GT\nHF={gt_hf:.3f}", fontsize=6); axes[0,0].axis("off")
    axes[1,0].imshow(gt_img[rmin:rmax+1,cmin:cmax+1]); axes[1,0].set_title("GT box", fontsize=6); axes[1,0].axis("off")
    axes[2,0].imshow(gt_img2[rmin:rmax+1,cmin:cmax+1]); axes[2,0].set_title("GT box2", fontsize=6); axes[2,0].axis("off")

    for i, (name, v) in enumerate(results.items()):
        xf0, xf1 = to_img(v["x"][0]), to_img(v["x"][1])
        axes[0,i+1].imshow(xf0)
        axes[0,i+1].set_title(f"{name}\nP={v['psnr']:.1f} r={v['res']:.0f}", fontsize=4)
        axes[0,i+1].axis("off")
        axes[1,i+1].imshow(xf0[rmin:rmax+1,cmin:cmax+1])
        axes[1,i+1].set_title(f"hf={v['hf']:.3f} |d|={v['dhf']:.3f}", fontsize=5)
        axes[1,i+1].axis("off")
        axes[2,i+1].imshow(xf1[rmin:rmax+1,cmin:cmax+1])
        axes[2,i+1].set_title(f"t={v['time']:.0f}s", fontsize=5)
        axes[2,i+1].axis("off")

    plt.suptitle(f"Hybrid v2: NO warm restart at stage 3 (GT HF={gt_hf:.3f})\n"
                 f"Row 1: full | Row 2: box img1 | Row 3: box img2", fontsize=8)
    plt.tight_layout(rect=[0,0,1,0.93])
    plt.savefig(f"{RESULT_DIR}/hybrid_v2_grid.png", dpi=200)
    plt.close()

    # Per-config images
    for name, v in results.items():
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for ii in range(2):
            gt_i, xf_i = to_img(gt[ii]), to_img(v["x"][ii])
            y_i = to_img((mask[0:1]*gt)[ii])
            axes[ii,0].imshow(gt_i); axes[ii,0].set_title("GT",fontsize=8); axes[ii,0].axis("off")
            axes[ii,1].imshow(y_i); axes[ii,1].set_title("Meas",fontsize=8); axes[ii,1].axis("off")
            axes[ii,2].imshow(xf_i); axes[ii,2].set_title(name,fontsize=8); axes[ii,2].axis("off")
            cg = gt_i[rmin:rmax+1,cmin:cmax+1]
            cx = xf_i[rmin:rmax+1,cmin:cmax+1]
            combined = np.concatenate([cg, np.ones((cg.shape[0],3,3)), cx], axis=1)
            axes[ii,3].imshow(combined); axes[ii,3].set_title("GT|Result",fontsize=8); axes[ii,3].axis("off")
        plt.suptitle(f"{name} — PSNR={v['psnr']:.2f} HF={v['hf']:.3f} |dHF|={v['dhf']:.3f} res={v['res']:.0f}", fontsize=10)
        plt.tight_layout(rect=[0,0,1,0.95])
        plt.savefig(f"{RESULT_DIR}/{name}.png", dpi=150)
        plt.close()

    print(f"\nSaved {RESULT_DIR}/hybrid_v2_grid.png + {len(results)} per-config images")
    print("DONE.", flush=True)


if __name__ == "__main__":
    main()
