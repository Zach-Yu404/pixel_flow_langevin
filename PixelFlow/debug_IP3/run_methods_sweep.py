#!/usr/bin/env python
"""
Sweep fundamentally different posterior sampling strategies.
NOT just parameter tuning — different ARCHITECTURES.

Methods:
1. baseline: current PRINCIPLE (CG-Langevin inner loop)
2. dps_like: DPS-style gradient correction at each ODE step (no inner loop)
3. replacement: replace observed pixels directly after each ODE step
4. replacement+dps: replace observed + DPS gradient on unobserved
5. pgdm: Pseudoinverse-Guided Diffusion (project onto measurement subspace)
6. resample: MCG-style resample with annealed mixing
7. alternating: alternate between unconditional ODE steps and data-consistency steps
"""
import sys, os, math, copy, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import torch, torch.nn.functional as F, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from diffusers.models.embeddings import get_2d_rotary_pos_embed
from diffusers.utils.torch_utils import randn_tensor
from inpaintingStart import get_operator
from ms_posterior_sampling_article_version_final_utils import (
    apply_G, apply_H_tau, compute_sigma_tau, direct_estimate_x1,
    make_Ak_fns, make_velocity_fn, sample_block_noise,
    principle_langevin_sample, cg_solve,
)
from pixelflow.scheduling_pixelflow import PixelFlowScheduler
from pixelflow.utils import config as config_utils
from pixelflow.utils.misc import seed_everything

DEVICE = "cuda:0"
RESULT_DIR = os.path.join(os.path.dirname(__file__), "results_methods")
os.makedirs(RESULT_DIR, exist_ok=True)

def hf_energy(x):
    fft = torch.fft.fft2(x); mag = fft.abs()
    H, W = x.shape[-2:]
    total = mag.pow(2).sum().item()
    lf = mag[:, :, :H//4, :W//4].pow(2).sum().item()
    return 1 - lf / max(total, 1e-12)

def run(model, config, gt, y, operator, sigma_n, method="baseline", **kw):
    B = gt.shape[0]; seed_everything(20000120)
    scheduler = PixelFlowScheduler(config.scheduler.num_train_timesteps,
                                   num_stages=config.scheduler.num_stages, gamma=-1/3)
    pe_labels = torch.tensor([10]*B, dtype=torch.int32, device=DEVICE)
    mask_full = operator.get_mask(x=gt).float().to(DEVICE)  # 1=observed

    num_langevin = kw.get("num_langevin", 10)
    zeta = kw.get("zeta", 1.0)  # DPS gradient scale
    omega = kw.get("omega", 0.5)  # replacement mixing
    ode_steps = kw.get("ode_steps", 10)

    h, w = 32, 32
    x1_k = torch.randn(B, 3, h, w, device=DEVICE)
    eps_k = randn_tensor((B, 3, h, w), device=DEVICE); lat = eps_k.clone()
    t0 = time.time()

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
            grid_size=(h//model.patch_size, w//model.patch_size), device=DEVICE, output_type="pt")
        rope = torch.stack(pe, -1)

        # Mask at current resolution
        if h < 256:
            mask_k = F.interpolate(mask_full, size=(h, w), mode="nearest")
        else:
            mask_k = mask_full

        for step_idx, T in enumerate(sc.Timesteps):
            tau = float(sc.t[step_idx].to(DEVICE))
            sig = compute_sigma_tau(tau, sk, ek)
            xtau = apply_H_tau(x1_k, tau, sk, ek, stage_idx=si) + sig*eps_k
            vfn = make_velocity_fn(model, T, pe_labels, st, rope, False, 0.0, si)

            # ═══ Warm restart (all methods get model prediction) ═══
            with torch.no_grad(): mu = vfn(xtau)
            x1_model = direct_estimate_x1(xtau - tau*mu, xtau + (1-tau)*mu, sk, ek).detach()

            if method == "baseline":
                x1_k = x1_model.clone()
                if sig > 1e-8:
                    eps_k = (xtau - apply_H_tau(x1_k, tau, sk, ek, stage_idx=si))/sig
                if sig >= 0.01:
                    x1_k, eps_k, _, _ = principle_langevin_sample(
                        x1_init=x1_k, eps_init=eps_k, tau=tau, s_k=sk, e_k=ek,
                        velocity_fn=vfn, A_k_fn=Ak, AT_k_fn=ATk,
                        y=y, sigma_n=sigma_n, h_x=0.1, h_epsilon=0.01,
                        lambda_x=0.01, lambda_reg=50.0, rho_s=1.0, rho_e=1.0,
                        cg_tol=1e-5, cg_max_iter=50, num_Langevin=num_langevin,
                        device=DEVICE, stage_idx=si, x1_init_mode="model",
                        noise_scale=0.0, return_traj=True, record_every=99)

            elif method == "dps":
                # DPS: x1 = x1_model + zeta * A^T(y - A(x1_model)) / ||y-A(x1_model)||
                x1_k = x1_model.clone()
                residual = y - Ak(x1_k)
                res_norm = residual.reshape(B, -1).norm(dim=1).reshape(B, 1, 1, 1).clamp(min=1e-8)
                x1_k = x1_k + zeta * ATk(residual) / res_norm
                if sig > 1e-8:
                    eps_k = (xtau - apply_H_tau(x1_k, tau, sk, ek, stage_idx=si))/sig

            elif method == "replacement":
                # Replace observed pixels with measurement, keep model prediction for unobserved
                x1_k = x1_model.clone()
                # At full resolution, direct replacement
                if h == 256:
                    x1_k = mask_k * y + (1 - mask_k) * x1_k
                else:
                    # At lower res: project y to current res, replace
                    y_down = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
                    x1_k = mask_k * y_down + (1 - mask_k) * x1_k
                if sig > 1e-8:
                    eps_k = (xtau - apply_H_tau(x1_k, tau, sk, ek, stage_idx=si))/sig

            elif method == "replacement+dps":
                # Replace observed + DPS gradient for fine-tuning
                x1_k = x1_model.clone()
                # DPS gradient first
                residual = y - Ak(x1_k)
                res_norm = residual.reshape(B, -1).norm(dim=1).reshape(B, 1, 1, 1).clamp(min=1e-8)
                x1_k = x1_k + zeta * ATk(residual) / res_norm
                # Then replace observed
                if h == 256:
                    x1_k = mask_k * y + (1 - mask_k) * x1_k
                else:
                    y_down = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
                    x1_k = mask_k * y_down + (1 - mask_k) * x1_k
                if sig > 1e-8:
                    eps_k = (xtau - apply_H_tau(x1_k, tau, sk, ek, stage_idx=si))/sig

            elif method == "pgdm":
                # Pseudoinverse-guided: Tikhonov solve for data consistency
                x1_k = x1_model.clone()
                if sig >= 0.01:
                    lam = kw.get("pgdm_lam", 100.0)
                    inv_eta_sq = 1.0 / (sigma_n ** 2)
                    def system(x):
                        return inv_eta_sq * ATk(Ak(x)) + lam * x
                    rhs = inv_eta_sq * ATk(y) + lam * x1_model
                    x1_k = cg_solve(system, rhs, tol=1e-5, max_iter=50)
                if sig > 1e-8:
                    eps_k = (xtau - apply_H_tau(x1_k, tau, sk, ek, stage_idx=si))/sig

            elif method == "resample":
                # MCG-style: mix model prediction with measurement-consistent version
                x1_k = x1_model.clone()
                # Create measurement-consistent version via replacement
                if h == 256:
                    x1_dc = mask_k * y + (1 - mask_k) * x1_k
                else:
                    y_down = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
                    x1_dc = mask_k * y_down + (1 - mask_k) * x1_k
                # Annealed mixing: start with model, end with DC
                progress = step_idx / max(len(sc.Timesteps) - 1, 1)
                alpha = omega * (1 - progress)  # decreasing model weight
                x1_k = alpha * x1_model + (1 - alpha) * x1_dc
                if sig > 1e-8:
                    eps_k = (xtau - apply_H_tau(x1_k, tau, sk, ek, stage_idx=si))/sig

            elif method == "alternating":
                # Alternate: odd steps = unconditional, even steps = data consistency
                x1_k = x1_model.clone()
                if step_idx % 2 == 0 and sig >= 0.01:
                    # Data consistency step
                    lam = kw.get("alt_lam", 100.0)
                    inv_eta_sq = 1.0 / (sigma_n ** 2)
                    def system(x):
                        return inv_eta_sq * ATk(Ak(x)) + lam * x
                    rhs = inv_eta_sq * ATk(y) + lam * x1_model
                    x1_k = cg_solve(system, rhs, tol=1e-5, max_iter=50)
                # Odd steps: just use model prediction (unconditional)
                if sig > 1e-8:
                    eps_k = (xtau - apply_H_tau(x1_k, tau, sk, ek, stage_idx=si))/sig

            elif method == "dps+langevin":
                # DPS gradient + few Langevin steps (hybrid)
                x1_k = x1_model.clone()
                # DPS correction first
                residual = y - Ak(x1_k)
                res_norm = residual.reshape(B, -1).norm(dim=1).reshape(B, 1, 1, 1).clamp(min=1e-8)
                x1_k = x1_k + zeta * ATk(residual) / res_norm
                if sig > 1e-8:
                    eps_k = (xtau - apply_H_tau(x1_k, tau, sk, ek, stage_idx=si))/sig
                # Then few Langevin steps for refinement
                if sig >= 0.01:
                    x1_k, eps_k, _, _ = principle_langevin_sample(
                        x1_init=x1_k, eps_init=eps_k, tau=tau, s_k=sk, e_k=ek,
                        velocity_fn=vfn, A_k_fn=Ak, AT_k_fn=ATk,
                        y=y, sigma_n=sigma_n, h_x=0.1, h_epsilon=0.01,
                        lambda_x=0.01, lambda_reg=50.0, rho_s=1.0, rho_e=1.0,
                        cg_tol=1e-5, cg_max_iter=20, num_Langevin=3,
                        device=DEVICE, stage_idx=si, x1_init_mode="model",
                        noise_scale=0.0, return_traj=True, record_every=99)

            lat = apply_H_tau(x1_k, tau, sk, ek, stage_idx=si) + sig*eps_k

    elapsed = time.time() - t0
    xf = x1_k.detach()
    Af, _ = make_Ak_fns(operator, y, (B, 3, 256, 256), DEVICE)
    # Publication PSNR: MAX=2 on [-1,1] range → 10*log10(4/MSE).
    diff2 = (xf - gt) ** 2
    C = xf.shape[1]
    m = mask_full[0:1]
    mse_all   = diff2.mean()
    n_obs     = m.sum().item() * C * B
    n_unobs   = (1 - m).sum().item() * C * B
    mse_obs   = (m * diff2).sum() / max(n_obs, 1)
    psnr_all = (10 * torch.log10(4.0 / mse_all.clamp(min=1e-12))).item()
    psnr_obs = (10 * torch.log10(4.0 / mse_obs.clamp(min=1e-12))).item()
    res = (y - Af(xf)).pow(2).sum().item()
    hf_u = hf_energy(xf * (1-mask_full[0:1]))
    return xf.cpu(), psnr_obs, psnr_all, res, elapsed, hf_u

def main():
    print("Loading...", flush=True)
    pt = "trajectory_videos/posterior_sampling/baseline_05/baseline_05_mode-f_x1.pt"
    if os.path.exists(pt):
        d = torch.load(pt, map_location="cpu", weights_only=False)
        gt = d["gt"][:2].to(DEVICE)
    else:
        from torchvision import transforms, datasets
        from ms_posterior_sampling_article_version_final_utils import center_crop_arr
        data_dir = "/data/Zach_dataset/imageNet256/ILSVRC/Data/CLS-LOC/train/"
        tf = transforms.Compose([transforms.Lambda(lambda p: center_crop_arr(p,256)),
              transforms.ToTensor(), transforms.Normalize([.5]*3,[.5]*3,inplace=True)])
        ds = datasets.ImageFolder(data_dir, tf); torch.manual_seed(20000120)
        ci = [i for i,(_, y) in enumerate(ds.samples) if int(y)==10]
        perm = torch.randperm(len(ci))[:2]
        gt = torch.stack([ds[ci[i]][0] for i in perm.tolist()]).to(DEVICE)
    config = OmegaConf.load("pretrained_models/c2img/config.yaml")
    model = config_utils.instantiate_from_config(config.model).to(DEVICE)
    ckpt = torch.load("pretrained_models/c2img/model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt, strict=True); model.eval()
    print("Model loaded.\n", flush=True)

    sigma_n = 0.05
    results = {}

    for mask_type, op_kw in [
        ("box", dict(mask_type="box", mask_len_range=(80,160), mask_prob_range=None)),
    ]:
        operator = get_operator("inpainting", resolution=256, device=DEVICE, sigma=sigma_n, **op_kw)
        y = operator(gt).detach()
        mask = operator.get_mask(x=gt).float().to(DEVICE)
        gt_hf = hf_energy(gt * (1-mask[0:1]))
        print(f"  {mask_type.upper()} (GT HF={gt_hf:.3f})", flush=True)

        configs = [
            # Baseline
            ("baseline_L10", "baseline", dict(num_langevin=10)),
            ("baseline_L3", "baseline", dict(num_langevin=3)),

            # DPS-style (no inner loop!)
            ("dps_z0.5", "dps", dict(zeta=0.5)),
            ("dps_z1.0", "dps", dict(zeta=1.0)),
            ("dps_z2.0", "dps", dict(zeta=2.0)),
            ("dps_z5.0", "dps", dict(zeta=5.0)),
            ("dps_z10",  "dps", dict(zeta=10.0)),

            # Replacement (observed = measurement)
            ("replace", "replacement", dict()),

            # Replacement + DPS
            ("rep+dps_z1", "replacement+dps", dict(zeta=1.0)),
            ("rep+dps_z5", "replacement+dps", dict(zeta=5.0)),

            # PGDM (Tikhonov at each step, no inner loop)
            ("pgdm_l50",  "pgdm", dict(pgdm_lam=50.0)),
            ("pgdm_l100", "pgdm", dict(pgdm_lam=100.0)),
            ("pgdm_l200", "pgdm", dict(pgdm_lam=200.0)),

            # MCG-style resample with annealed mixing
            ("resamp_w0.3", "resample", dict(omega=0.3)),
            ("resamp_w0.5", "resample", dict(omega=0.5)),
            ("resamp_w0.7", "resample", dict(omega=0.7)),

            # Alternating: unconditional + DC
            ("alt_l100", "alternating", dict(alt_lam=100.0)),
            ("alt_l200", "alternating", dict(alt_lam=200.0)),

            # DPS + few Langevin (hybrid)
            ("dps+L3_z1", "dps+langevin", dict(zeta=1.0)),
            ("dps+L3_z5", "dps+langevin", dict(zeta=5.0)),

            # More ODE steps variants
            ("dps_z1_20s", "dps", dict(zeta=1.0, ode_steps=20)),
            ("replace_20s", "replacement", dict(ode_steps=20)),
            ("pgdm_l100_20s", "pgdm", dict(pgdm_lam=100.0, ode_steps=20)),
        ]

        for name, meth, kw in configs:
            print(f"  {name}...", end=" ", flush=True)
            xf, po, pa, res, t, hf = run(model, config, gt, y, operator, sigma_n,
                                          method=meth, **kw)
            results[f"{mask_type}_{name}"] = (xf, po, pa, res, t, hf)
            print(f"res={res:.0f}  PSNR_obs={po:.2f}  PSNR_all={pa:.2f}  "
                  f"HF={hf:.3f}  t={t:.0f}s", flush=True)

    # Summary
    print(f"\n{'='*80}", flush=True)
    print(f"{'Config':<25} {'Method':<15} {'Res':>7} {'PSNR_obs':>9} {'PSNR_all':>9} "
          f"{'HF':>6} {'Time':>6}", flush=True)
    print("-"*80, flush=True)
    ranked = sorted(results.items(), key=lambda kv: kv[1][2], reverse=True)
    for k, v in ranked:
        xf, po, pa, res, t, hf = v
        print(f"{k:<25} {'':>15} {res:>7.0f} {po:>9.2f} {pa:>9.2f} "
              f"{hf:>6.3f} {t:>5.0f}s", flush=True)

    # Visualization
    keys = [k for k, _ in ranked]
    n = len(keys)
    cols = min(8, n+1); rows = (n+1+cols-1)//cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    if rows == 1: axes = axes[np.newaxis, :]
    axes = axes.flatten()
    img_gt = (gt[0].cpu().permute(1,2,0)*0.5+0.5).clamp(0,1).numpy()
    axes[0].imshow(img_gt); axes[0].set_title("GT", fontsize=7); axes[0].axis("off")
    for i, k in enumerate(keys):
        xf, po, pa, res, t, hf = results[k]
        img = (xf[0].permute(1,2,0)*0.5+0.5).clamp(0,1).numpy()
        short = k.replace("box_","")
        axes[i+1].imshow(img)
        axes[i+1].set_title(f"{short}\n{pa:.1f}dB hf={hf:.2f}", fontsize=5)
        axes[i+1].axis("off")
    for j in range(i+2, len(axes)): axes[j].axis("off")
    plt.suptitle("Methods sweep: DPS, replacement, PGDM, resample, alternating", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/methods_sweep.png", dpi=150)
    plt.close()
    print(f"Saved methods_sweep.png", flush=True)
    print("DONE.", flush=True)

if __name__ == "__main__":
    main()
