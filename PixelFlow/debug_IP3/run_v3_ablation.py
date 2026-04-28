#!/usr/bin/env python
"""
v3 ablation: null-space boosted Langevin, reduced-lambda, Tikhonov.

Key fix from v3a: applying g_prior directly on observed pixels destroys
data consistency (the CG 450x damping on observed IS essential). Correct
approaches:
  - null_boost: add prior boost ONLY on null space (unobserved pixels)
  - reduced_lambda: smaller lambda_reg → more prior on unobserved via CG
  - tikhonov: one-shot solve, observed→data, unobserved→model (cleanest)
"""
import sys, os, math, copy, time
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
    principle_langevin_sample,
)
from pixelflow.scheduling_pixelflow import PixelFlowScheduler
from pixelflow.utils import config as config_utils
from pixelflow.utils.misc import seed_everything

from debug_IP3.langevin_v3 import tikhonov_update, nullboost_sample

DEVICE = "cuda:0"
RESULT_DIR = os.path.join(os.path.dirname(__file__), "results_v3")
os.makedirs(RESULT_DIR, exist_ok=True)


def run_pipeline(model, config, gt, y, operator, sigma_n, method="baseline", **kw):
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
    ode_steps = kw.get("ode_steps", 10)

    # Null-boost params
    h_null = kw.get("h_null", 0.0)              # scalar or list[4]
    max_score_norm = kw.get("max_score_norm", 500.0)
    sigma_ref_sq = kw.get("sigma_ref_sq", 0.01)

    # Tikhonov params
    lambda_tik = kw.get("lambda_tik", 50.0)     # scalar or list[4]

    h, w = 32, 32
    x1_k = torch.randn(B, 3, h, w, device=DEVICE)
    eps_k = randn_tensor((B, 3, h, w), device=DEVICE)
    lat = eps_k.clone()

    all_residuals = []
    all_logs = []
    t_start = time.time()

    for si in range(4):
        sc = copy.deepcopy(scheduler)
        sc.set_timesteps(ode_steps, si, device=DEVICE, shift=1.0)
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

        # Per-stage params
        hn_val = h_null[si] if isinstance(h_null, (list, tuple)) else h_null
        lt_val = lambda_tik[si] if isinstance(lambda_tik, (list, tuple)) else lambda_tik

        for step_idx, T in enumerate(sc.Timesteps):
            tau = float(sc.t[step_idx].to(DEVICE))
            sig = compute_sigma_tau(tau, sk, ek)
            xtau = apply_H_tau(x1_k, tau, sk, ek, stage_idx=si) + sig*eps_k
            vfn = make_velocity_fn(model, T, prompt_embeds, st, rope, False, 0.0, si)

            # ── Warm restart ──
            with torch.no_grad(): mu = vfn(xtau)
            x1_k = direct_estimate_x1(xtau - tau*mu, xtau + (1-tau)*mu, sk, ek).detach()
            if sig > 1e-8:
                eps_k = (xtau - apply_H_tau(x1_k, tau, sk, ek, stage_idx=si))/sig

            step_logs = []
            if sig >= 0.01:
                if method == "baseline":
                    x1_k, eps_k, _, step_logs = principle_langevin_sample(
                        x1_init=x1_k, eps_init=eps_k, tau=tau, s_k=sk, e_k=ek,
                        velocity_fn=vfn, A_k_fn=Ak, AT_k_fn=ATk,
                        y=y, sigma_n=sigma_n, h_x=h_x, h_epsilon=h_eps,
                        lambda_x=0.01, lambda_reg=lambda_reg, rho_s=1.0, rho_e=1.0,
                        cg_tol=1e-5, cg_max_iter=50, num_Langevin=num_langevin,
                        device=DEVICE, stage_idx=si, x1_init_mode="model",
                        noise_scale=0.0, return_traj=True, record_every=5)

                elif method == "null_boost":
                    x1_k, eps_k, _, step_logs = nullboost_sample(
                        x1_init=x1_k, eps_init=eps_k, tau=tau, s_k=sk, e_k=ek,
                        velocity_fn=vfn, A_k_fn=Ak, AT_k_fn=ATk,
                        y=y, sigma_n=sigma_n, h_x=h_x, h_epsilon=h_eps,
                        lambda_x=0.01, lambda_reg=lambda_reg, rho_s=1.0, rho_e=1.0,
                        cg_tol=1e-5, cg_max_iter=50, num_Langevin=num_langevin,
                        device=DEVICE, stage_idx=si, x1_init_mode="model",
                        noise_scale=0.0,
                        sigma_ref_sq=sigma_ref_sq, max_score_norm=max_score_norm,
                        h_null=hn_val,
                        return_traj=True, record_every=5)

                elif method == "reduced_lambda":
                    x1_k, eps_k, _, step_logs = principle_langevin_sample(
                        x1_init=x1_k, eps_init=eps_k, tau=tau, s_k=sk, e_k=ek,
                        velocity_fn=vfn, A_k_fn=Ak, AT_k_fn=ATk,
                        y=y, sigma_n=sigma_n, h_x=h_x, h_epsilon=h_eps,
                        lambda_x=0.01, lambda_reg=lambda_reg, rho_s=1.0, rho_e=1.0,
                        cg_tol=1e-5, cg_max_iter=50, num_Langevin=num_langevin,
                        device=DEVICE, stage_idx=si, x1_init_mode="model",
                        noise_scale=0.0, return_traj=True, record_every=5)

                elif method == "tikhonov":
                    x1_k, tik_log = tikhonov_update(
                        x1_hat=x1_k, A_k_fn=Ak, AT_k_fn=ATk,
                        b=y, eta=sigma_n, lambda_tik=lt_val,
                        cg_tol=1e-5, cg_max_iter=100)
                    if sig > 1e-8:
                        eps_k = (xtau - apply_H_tau(x1_k, tau, sk, ek, stage_idx=si))/sig
                    step_logs = [tik_log]

            lat = apply_H_tau(x1_k, tau, sk, ek, stage_idx=si) + sig*eps_k

            if h == 256:
                res = (y - Ak(x1_k)).pow(2).sum().item()
            else:
                xu = F.interpolate(x1_k, size=(256,256), mode="bilinear", align_corners=False)
                Af, _ = make_Ak_fns(operator, y, (B, 3, 256, 256), DEVICE)
                res = (y - Af(xu)).pow(2).sum().item()
            all_residuals.append(res)

            if step_logs:
                last = step_logs[-1]
                last["stage"] = si; last["step"] = step_idx
                last["tau"] = tau; last["sigma_tau"] = sig
                all_logs.append(last)

    elapsed = time.time() - t_start
    xf = x1_k.detach()
    psnr_all = -10*torch.log10(((xf - gt)**2).mean()).item()
    psnr_obs = -10*torch.log10(((mask[0:1]*xf - mask[0:1]*gt)**2).mean()).item()
    final_res = all_residuals[-1]

    return xf.cpu(), all_residuals, psnr_obs, psnr_all, final_res, all_logs, elapsed


def main():
    print("Loading model...", flush=True)
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
            print(f"  GT from {pp}", flush=True)
            break
    if gt is None:
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
        ci = [i for i, (_, yl) in enumerate(dataset.samples) if int(yl) == 10]
        perm = torch.randperm(len(ci))[:2]
        gt = torch.stack([dataset[ci[i]][0] for i in perm.tolist()], dim=0).to(DEVICE)
        print(f"  GT from ImageNet", flush=True)

    config = OmegaConf.load("pretrained_models/c2img/config.yaml")
    model = config_utils.instantiate_from_config(config.model).to(DEVICE)
    ckpt = torch.load("pretrained_models/c2img/model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt, strict=True); model.eval()
    print("Model loaded.\n", flush=True)

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
            # ── Baseline (current v2) ──
            ("baseline", "baseline", dict()),

            # ── Null-space prior boost: h_null sweep ──
            ("nb_h0.005", "null_boost", dict(h_null=0.005)),
            ("nb_h0.01",  "null_boost", dict(h_null=0.01)),
            ("nb_h0.02",  "null_boost", dict(h_null=0.02)),
            ("nb_h0.05",  "null_boost", dict(h_null=0.05)),
            ("nb_h0.10",  "null_boost", dict(h_null=0.10)),

            # ── Null-space boost: staged (stronger at late stages) ──
            ("nb_staged_A", "null_boost", dict(h_null=[0.0, 0.005, 0.01, 0.05])),
            ("nb_staged_B", "null_boost", dict(h_null=[0.0, 0.01, 0.02, 0.10])),

            # ── Null-space boost: different clip ──
            ("nb_clip100", "null_boost", dict(h_null=0.02, max_score_norm=100.0)),
            ("nb_clip1k",  "null_boost", dict(h_null=0.02, max_score_norm=1000.0)),

            # ── Reduced lambda_reg ──
            ("lam20", "reduced_lambda", dict(lambda_reg=20.0)),
            ("lam10", "reduced_lambda", dict(lambda_reg=10.0)),
            ("lam30", "reduced_lambda", dict(lambda_reg=30.0)),

            # ── Tikhonov: lambda sweep ──
            ("tik_lam25",  "tikhonov", dict(lambda_tik=25.0)),
            ("tik_lam50",  "tikhonov", dict(lambda_tik=50.0)),
            ("tik_lam100", "tikhonov", dict(lambda_tik=100.0)),
            ("tik_lam200", "tikhonov", dict(lambda_tik=200.0)),

            # ── Tikhonov: more ODE steps ──
            ("tik_l50_20s",  "tikhonov", dict(lambda_tik=50.0,  ode_steps=20)),
            ("tik_l100_20s", "tikhonov", dict(lambda_tik=100.0, ode_steps=20)),
            ("tik_l50_30s",  "tikhonov", dict(lambda_tik=50.0,  ode_steps=30)),

            # ── Tikhonov: staged lambda ──
            ("tik_stgA", "tikhonov", dict(lambda_tik=[25, 50, 75, 100])),
            ("tik_stgB", "tikhonov", dict(lambda_tik=[50, 75, 100, 200])),
        ]

        for name, meth, kw in ablations:
            print(f"  {name} ({meth})...", end=" ", flush=True)
            x1, res, po, pa, fr, logs, elapsed = run_pipeline(
                model, config, gt, y, operator, sigma_n, method=meth, **kw)
            results[f"{mask_type}_{name}"] = (x1, res, po, pa, fr, logs, elapsed)
            ratio = np.mean([l.get("prior_data_ratio", 0) for l in logs
                            if l.get("stage") == 3 and l.get("prior_data_ratio", 0) > 0]) if logs else 0
            print(f"res={fr:.0f}  PSNR_obs={po:.2f}  PSNR_all={pa:.2f}  "
                  f"P/D_S3={ratio:.2f}  t={elapsed:.1f}s", flush=True)

    # ══ Summary table ══
    print(f"\n{'='*90}", flush=True)
    print(f"{'Config':<30} {'Res':>7} {'PSNR_obs':>9} {'PSNR_all':>9} {'Time':>6}", flush=True)
    print("-"*65, flush=True)
    for k in sorted(results.keys()):
        _, _, po, pa, fr, _, elapsed = results[k]
        print(f"{k:<30} {fr:>7.0f} {po:>9.2f} {pa:>9.2f} {elapsed:>5.1f}s", flush=True)

    # ══ Visualization ══
    for mask_type in ["box", "random"]:
        keys = sorted([k for k in results if k.startswith(mask_type)])
        if not keys: continue
        n = len(keys)
        cols = min(7, n+1)
        rows = (n+1+cols-1)//cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols*3.5, rows*3.5))
        if rows == 1: axes = axes[np.newaxis, :]
        axes = axes.flatten()

        img_gt = (gt[0].cpu().permute(1,2,0)*0.5+0.5).clamp(0,1).numpy()
        axes[0].imshow(img_gt); axes[0].set_title("GT", fontsize=7); axes[0].axis("off")
        for i, k in enumerate(keys):
            x1, _, po, pa, fr, _, elapsed = results[k]
            img = (x1[0].permute(1,2,0)*0.5+0.5).clamp(0,1).numpy()
            short = k.replace(f"{mask_type}_", "")
            axes[i+1].imshow(img)
            axes[i+1].set_title(f"{short}\n{pa:.1f}dB {elapsed:.0f}s", fontsize=5)
            axes[i+1].axis("off")
        for j in range(i+2, len(axes)): axes[j].axis("off")
        plt.suptitle(f"{mask_type} — v3 ablation (corrected)", fontsize=10)
        plt.tight_layout()
        plt.savefig(f"{RESULT_DIR}/v3_ablation_{mask_type}.png", dpi=150)
        plt.close()
        print(f"Saved v3_ablation_{mask_type}.png", flush=True)

    # ══ Residual curves ══
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for idx, mask_type in enumerate(["box", "random"]):
        for k in sorted(results.keys()):
            if not k.startswith(mask_type): continue
            _, res_list, _, _, _, _, _ = results[k]
            short = k.replace(f"{mask_type}_", "")
            axes[idx].plot(res_list, "-", linewidth=0.8, label=short, alpha=0.7)
        axes[idx].set_yscale("log")
        axes[idx].set_xlabel("Step"); axes[idx].set_ylabel("Residual ||y-Ax||²")
        axes[idx].set_title(f"{mask_type}: residual convergence")
        axes[idx].legend(fontsize=4, ncol=3); axes[idx].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/v3_residuals.png", dpi=150)
    plt.close()
    print("Saved v3_residuals.png", flush=True)

    # ══ Best configs ══
    print(f"\n{'='*60}", flush=True)
    print("TOP 5 by PSNR_all:", flush=True)
    ranked = sorted(results.items(), key=lambda kv: kv[1][3], reverse=True)
    for i, (k, v) in enumerate(ranked[:5]):
        _, _, po, pa, fr, _, elapsed = v
        print(f"  {i+1}. {k}: PSNR_all={pa:.2f}  res={fr:.0f}  t={elapsed:.1f}s", flush=True)

    # ══ Save summary ══
    with open(f"{RESULT_DIR}/v3_summary.txt", "w") as f:
        f.write(f"{'Config':<30} {'Res':>8} {'PSNR_obs':>9} {'PSNR_all':>9} {'Time':>7}\n")
        f.write("-"*68 + "\n")
        for k in sorted(results.keys()):
            _, _, po, pa, fr, _, elapsed = results[k]
            f.write(f"{k:<30} {fr:>8.0f} {po:>9.2f} {pa:>9.2f} {elapsed:>6.1f}s\n")
        f.write("\nTOP 5:\n")
        for i, (k, v) in enumerate(ranked[:5]):
            _, _, po, pa, fr, _, elapsed = v
            f.write(f"  {i+1}. {k}: PSNR_all={pa:.2f}  res={fr:.0f}  t={elapsed:.1f}s\n")
    print("Saved v3_summary.txt", flush=True)

    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()
