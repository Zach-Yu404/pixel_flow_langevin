#!/usr/bin/env python
"""
v4 ablation: frequency-aware posterior sampling.
Tests x1_hat sharpening, score HF boost, annealed noise, and combinations.
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

from debug_IP3.langevin_v4 import v4_langevin_sample

DEVICE = "cuda:0"
RESULT_DIR = os.path.join(os.path.dirname(__file__), "results_v4")
os.makedirs(RESULT_DIR, exist_ok=True)


def high_freq_energy(x):
    fft = torch.fft.fft2(x)
    mag = fft.abs()
    H, W = x.shape[-2:]
    total = mag.pow(2).sum().item()
    lf = mag[:, :, :H//4, :W//4].pow(2).sum().item()
    return 1 - lf / max(total, 1e-12)


def run_pipeline(model, config, gt, y, operator, sigma_n, **kw):
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

    # v4 params
    noise_init = kw.get("noise_init", 0.0)
    noise_anneal_frac = kw.get("noise_anneal_frac", 0.7)
    x1hat_hf_boost = kw.get("x1hat_hf_boost", 0.0)
    score_hf_boost = kw.get("score_hf_boost", 0.0)
    hf_kernel = kw.get("hf_kernel", 5)
    use_baseline = kw.get("use_baseline", False)

    h, w = 32, 32
    x1_k = torch.randn(B, 3, h, w, device=DEVICE)
    eps_k = randn_tensor((B, 3, h, w), device=DEVICE)
    lat = eps_k.clone()

    all_residuals = []
    t_start = time.time()

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

            if sig >= 0.01:
                if use_baseline:
                    x1_k, eps_k, _, _ = principle_langevin_sample(
                        x1_init=x1_k, eps_init=eps_k, tau=tau, s_k=sk, e_k=ek,
                        velocity_fn=vfn, A_k_fn=Ak, AT_k_fn=ATk,
                        y=y, sigma_n=sigma_n, h_x=h_x, h_epsilon=h_eps,
                        lambda_x=0.01, lambda_reg=lambda_reg, rho_s=1.0, rho_e=1.0,
                        cg_tol=1e-5, cg_max_iter=50, num_Langevin=num_langevin,
                        device=DEVICE, stage_idx=si, x1_init_mode="model",
                        noise_scale=0.0, return_traj=True, record_every=5)
                else:
                    x1_k, eps_k, _, _ = v4_langevin_sample(
                        x1_init=x1_k, eps_init=eps_k, tau=tau, s_k=sk, e_k=ek,
                        velocity_fn=vfn, A_k_fn=Ak, AT_k_fn=ATk,
                        y=y, sigma_n=sigma_n, h_x=h_x, h_epsilon=h_eps,
                        lambda_x=0.01, lambda_reg=lambda_reg, rho_s=1.0, rho_e=1.0,
                        cg_tol=1e-5, cg_max_iter=50, num_Langevin=num_langevin,
                        device=DEVICE, stage_idx=si, x1_init_mode="model",
                        noise_init=noise_init,
                        noise_anneal_frac=noise_anneal_frac,
                        x1hat_hf_boost=x1hat_hf_boost,
                        score_hf_boost=score_hf_boost,
                        hf_kernel=hf_kernel,
                        return_traj=True, record_every=5)

            lat = apply_H_tau(x1_k, tau, sk, ek, stage_idx=si) + sig*eps_k

            if h == 256:
                res = (y - Ak(x1_k)).pow(2).sum().item()
            else:
                xu = F.interpolate(x1_k, size=(256,256), mode="bilinear", align_corners=False)
                Af, _ = make_Ak_fns(operator, y, (B, 3, 256, 256), DEVICE)
                res = (y - Af(xu)).pow(2).sum().item()
            all_residuals.append(res)

    elapsed = time.time() - t_start
    xf = x1_k.detach()
    psnr_all = -10*torch.log10(((xf - gt)**2).mean()).item()
    psnr_obs = -10*torch.log10(((mask[0:1]*xf - mask[0:1]*gt)**2).mean()).item()
    hf_unobs = high_freq_energy(xf * (1 - mask[0:1]))
    hf_obs = high_freq_energy(xf * mask[0:1])
    final_res = all_residuals[-1]

    return xf.cpu(), all_residuals, psnr_obs, psnr_all, final_res, elapsed, hf_unobs, hf_obs


def main():
    print("Loading model...", flush=True)
    pt_candidates = [
        "trajectory_videos/posterior_sampling/baseline_05/baseline_05_mode-f_x1.pt",
        "trajectory_videos/posterior_sampling/principle_final_box/principle_final_box_mode-f_x1.pt",
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

    # GT high-freq reference
    mask_ref = None
    print("Model loaded.\n", flush=True)

    sigma_n = 0.05
    results = {}

    for mask_type, op_kw in [
        ("box", dict(mask_type="box", mask_len_range=(80,160), mask_prob_range=None)),
        ("random", dict(mask_type="random", mask_len_range=None, mask_prob_range=(0.8,0.8))),
    ]:
        operator = get_operator("inpainting", resolution=256, device=DEVICE, sigma=sigma_n, **op_kw)
        y = operator(gt).detach()
        mask = operator.get_mask(x=gt).float().to(DEVICE)

        # GT reference HF
        gt_hf_unobs = high_freq_energy(gt * (1 - mask[0:1]))
        gt_hf_obs = high_freq_energy(gt * mask[0:1])

        print(f"\n{'='*60}", flush=True)
        print(f"  {mask_type.upper()} INPAINTING  (GT HF: obs={gt_hf_obs:.3f} unobs={gt_hf_unobs:.3f})", flush=True)
        print(f"{'='*60}", flush=True)

        ablations = [
            # ── Baseline ──
            ("baseline", dict(use_baseline=True)),

            # ── A: x1_hat HF boost sweep ──
            ("x1hf_0.5",  dict(x1hat_hf_boost=0.5)),
            ("x1hf_1.0",  dict(x1hat_hf_boost=1.0)),
            ("x1hf_2.0",  dict(x1hat_hf_boost=2.0)),
            ("x1hf_3.0",  dict(x1hat_hf_boost=3.0)),
            ("x1hf_5.0",  dict(x1hat_hf_boost=5.0)),

            # ── B: Score HF boost sweep ──
            ("shf_0.5",   dict(score_hf_boost=0.5)),
            ("shf_1.0",   dict(score_hf_boost=1.0)),
            ("shf_2.0",   dict(score_hf_boost=2.0)),
            ("shf_3.0",   dict(score_hf_boost=3.0)),

            # ── C: Annealed noise sweep ──
            ("noise_0.05", dict(noise_init=0.05)),
            ("noise_0.1",  dict(noise_init=0.1)),
            ("noise_0.2",  dict(noise_init=0.2)),
            ("noise_0.3",  dict(noise_init=0.3)),
            ("noise_0.5",  dict(noise_init=0.5)),

            # ── D: Kernel size variants (with x1hf=2.0) ──
            ("x1hf2_k3",  dict(x1hat_hf_boost=2.0, hf_kernel=3)),
            ("x1hf2_k7",  dict(x1hat_hf_boost=2.0, hf_kernel=7)),

            # ── E: Combinations ──
            ("x1hf2+n0.1",   dict(x1hat_hf_boost=2.0, noise_init=0.1)),
            ("x1hf2+n0.2",   dict(x1hat_hf_boost=2.0, noise_init=0.2)),
            ("x1hf2+shf1",   dict(x1hat_hf_boost=2.0, score_hf_boost=1.0)),
            ("shf1+n0.1",    dict(score_hf_boost=1.0, noise_init=0.1)),
            ("shf1+n0.2",    dict(score_hf_boost=1.0, noise_init=0.2)),
            ("all_mod",      dict(x1hat_hf_boost=2.0, score_hf_boost=1.0, noise_init=0.1)),
            ("all_strong",   dict(x1hat_hf_boost=3.0, score_hf_boost=2.0, noise_init=0.2)),
        ]

        for name, kw in ablations:
            print(f"  {name}...", end=" ", flush=True)
            x1, res, po, pa, fr, elapsed, hf_u, hf_o = run_pipeline(
                model, config, gt, y, operator, sigma_n, **kw)
            results[f"{mask_type}_{name}"] = (x1, res, po, pa, fr, elapsed, hf_u, hf_o)
            print(f"res={fr:.0f}  PSNR_obs={po:.2f}  PSNR_all={pa:.2f}  "
                  f"HF_unobs={hf_u:.3f}  t={elapsed:.0f}s", flush=True)

    # ══ Summary ══
    print(f"\n{'='*95}", flush=True)
    print(f"{'Config':<28} {'Res':>6} {'PSNR_obs':>9} {'PSNR_all':>9} "
          f"{'HF_unobs':>9} {'HF_obs':>7} {'Time':>6}", flush=True)
    print("-"*78, flush=True)
    for k in sorted(results.keys()):
        _, _, po, pa, fr, elapsed, hf_u, hf_o = results[k]
        print(f"{k:<28} {fr:>6.0f} {po:>9.2f} {pa:>9.2f} "
              f"{hf_u:>9.3f} {hf_o:>7.3f} {elapsed:>5.0f}s", flush=True)

    # ══ Visualization ══
    for mask_type in ["box", "random"]:
        keys = sorted([k for k in results if k.startswith(mask_type)])
        if not keys: continue
        n = len(keys)
        cols = min(8, n+1)
        rows = (n+1+cols-1)//cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
        if rows == 1: axes = axes[np.newaxis, :]
        axes = axes.flatten()

        img_gt = (gt[0].cpu().permute(1,2,0)*0.5+0.5).clamp(0,1).numpy()
        axes[0].imshow(img_gt); axes[0].set_title("GT", fontsize=6); axes[0].axis("off")
        for i, k in enumerate(keys):
            x1, _, po, pa, fr, elapsed, hf_u, _ = results[k]
            img = (x1[0].permute(1,2,0)*0.5+0.5).clamp(0,1).numpy()
            short = k.replace(f"{mask_type}_", "")
            axes[i+1].imshow(img)
            axes[i+1].set_title(f"{short}\n{pa:.1f}dB hf={hf_u:.2f}", fontsize=5)
            axes[i+1].axis("off")
        for j in range(i+2, len(axes)): axes[j].axis("off")
        plt.suptitle(f"{mask_type} — v4 frequency-aware ablation", fontsize=9)
        plt.tight_layout()
        plt.savefig(f"{RESULT_DIR}/v4_ablation_{mask_type}.png", dpi=150)
        plt.close()
        print(f"Saved v4_ablation_{mask_type}.png", flush=True)

    # ══ Best configs ══
    print(f"\n{'='*60}", flush=True)
    print("TOP 5 by PSNR_all:", flush=True)
    ranked = sorted(results.items(), key=lambda kv: kv[1][3], reverse=True)
    for i, (k, v) in enumerate(ranked[:5]):
        _, _, po, pa, fr, elapsed, hf_u, hf_o = v
        print(f"  {i+1}. {k}: PSNR_all={pa:.2f}  HF_unobs={hf_u:.3f}  res={fr:.0f}  t={elapsed:.0f}s", flush=True)

    print("\nTOP 5 by HF_unobs (highest high-freq energy in unobserved):", flush=True)
    ranked_hf = sorted(results.items(), key=lambda kv: kv[1][6], reverse=True)
    for i, (k, v) in enumerate(ranked_hf[:5]):
        _, _, po, pa, fr, elapsed, hf_u, hf_o = v
        print(f"  {i+1}. {k}: HF_unobs={hf_u:.3f}  PSNR_all={pa:.2f}  res={fr:.0f}", flush=True)

    # Save summary
    with open(f"{RESULT_DIR}/v4_summary.txt", "w") as f:
        f.write(f"{'Config':<28} {'Res':>8} {'PSNR_obs':>9} {'PSNR_all':>9} "
                f"{'HF_unobs':>9} {'Time':>7}\n")
        f.write("-"*75 + "\n")
        for k in sorted(results.keys()):
            _, _, po, pa, fr, elapsed, hf_u, _ = results[k]
            f.write(f"{k:<28} {fr:>8.0f} {po:>9.2f} {pa:>9.2f} "
                    f"{hf_u:>9.3f} {elapsed:>6.0f}s\n")
    print("Saved v4_summary.txt", flush=True)

    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()
