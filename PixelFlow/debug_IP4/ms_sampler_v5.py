"""
debug_IP4 / ms_sampler_v5.py

Self-contained PRINCIPLE sampler with all IP4 axes exposed:

  Per-stage schedules (each is scalar OR length-num_stages list):
    num_langevin, h_x, h_epsilon, lambda_reg, noise_scale

  Per-step / structural knobs:
    guidance_scale (CFG; was fixed to 0 in IP3)
    dps_kick_zeta          : DPS pre-kick before Langevin, stage-list OK
    terminal_replace_weight: final x1 <- mask*y + (1-mask)*x1 blend, 0-1
    soft_replace_weight    : per-step replacement blend at stage 3 (0 disables)
    h_x_obs_ratio          : mask-aware step size (observed region), stage-list OK
    ode_steps_per_stage    : scalar or list, replaces inference_each_step

Returns generated x1_final (B,3,256,256), residual, psnr_all, hf_unobs, time.
"""

import math, time, copy
import torch
import torch.nn.functional as F

from diffusers.models.embeddings import get_2d_rotary_pos_embed
from diffusers.utils.torch_utils import randn_tensor

from ms_posterior_sampling_article_version_final_utils import (
    apply_G, apply_H_tau, compute_sigma_tau, direct_estimate_x1,
    wls_estimate_x1,
    make_Ak_fns, make_velocity_fn, sample_block_noise,
)
from pixelflow.scheduling_pixelflow import PixelFlowScheduler

from langevin_v5 import (
    principle_langevin_v5, dps_gradient_kick, terminal_replacement,
)


def _ps(val, k, default):
    if val is None:
        return default
    if isinstance(val, (list, tuple)):
        return val[k]
    return val


def hf_energy(x):
    fft = torch.fft.fft2(x); mag = fft.abs()
    H, W = x.shape[-2:]
    total = mag.pow(2).sum().item()
    lf = mag[:, :, :H // 4, :W // 4].pow(2).sum().item()
    return 1 - lf / max(total, 1e-12)


@torch.no_grad()
def run_ip4(
    model, config, gt, y, operator, sigma_n, device,
    *,
    # Per-stage schedules (scalar or 4-list)
    num_langevin=10, h_x=0.1, h_epsilon=0.01,
    lambda_reg=50.0, noise_scale=0.0,
    # Timesteps
    ode_steps_per_stage=10, shift=1.0,
    # Structural / physical
    guidance_scale=0.0, class_label=10,
    warm_restart=True, g_bypass_stage3=True,
    x1_init_mode="model", lambda_x=0.01,
    rho_s=1.0, rho_e=1.0, cg_tol=1e-5, cg_max_iter=50,
    # Hybrid knobs
    dps_kick_zeta=None,           # e.g. 5.0 or [0,0,0,5]
    terminal_replace_weight=0.0,  # 0 = off; 1 = hard replace at end
    soft_replace_weight=0.0,      # 0 = off; per-step blend at stage 3
    h_x_obs_ratio=1.0,            # 1.0 = no mask-awareness
    lambda_prox=0.0,              # paper step (d) 3rd term: λ(sg(x1_hat) - x1_k) weight; scalar or 4-list
    skip_eps_update=False,        # True = skip step (f) entirely; scalar or 4-list of bools
    reset_eps_per_ode_step=False, # True = before each ODE step, resample eps ~ N(0,I); keeps training distribution
    # Round-6 g_eps augmentation:
    lambda_eps_norm=0.0,          # push ||eps||² toward N (N1)
    mask_aware_eps=False,         # unobs region: drop Tweedie term, pure prior (N2)
    lambda_eps_prox=0.0,          # proximal to initial Gaussian eps_0 (N3)
    lambda_eps_inject=0.0,        # active noise injection per step (N4)
    sigma_ref_sq=0.01,            # Tweedie soft-damping floor; scalar or 4-list
    final_denoise=False,          # extra WLS-CG pass on the final x1 (post-sampling, pre-tr)
    seed=20000120,
):
    from pixelflow.utils.misc import seed_everything
    B = gt.shape[0]
    seed_everything(seed)
    torch.manual_seed(seed)

    num_stages = int(config.scheduler.num_stages)
    scheduler = PixelFlowScheduler(
        config.scheduler.num_train_timesteps,
        num_stages=num_stages, gamma=-1 / 3,
    )

    if isinstance(class_label, (list, tuple)) or torch.is_tensor(class_label):
        pe_labels = torch.as_tensor(class_label, dtype=torch.int32, device=device)
        assert pe_labels.shape[0] == B, f"class_label list len {pe_labels.shape[0]} != B {B}"
    else:
        pe_labels = torch.tensor([class_label] * B, dtype=torch.int32, device=device)
    do_cfg = guidance_scale > 0
    if do_cfg:
        uncond_label = int(model.num_classes)
        neg = uncond_label * torch.ones_like(pe_labels)
        pe_combined = torch.cat([neg, pe_labels], dim=0)
    else:
        pe_combined = pe_labels

    mask_full = operator.get_mask(x=gt).float().to(device)  # (1|B,1,256,256)
    if mask_full.shape[0] == 1 and B > 1:
        mask_full = mask_full.expand(B, -1, -1, -1)

    init_factor = 2 ** (num_stages - 1)
    h = w = 256 // init_factor
    shape = (B, 3, h, w)

    x1_k = torch.randn(shape, device=device, dtype=torch.float32)
    eps_k = randn_tensor(shape, device=device, dtype=torch.float32)
    lat = eps_k.clone()
    eps0_init = eps_k.clone()  # reference for N3 proximal
    t0 = time.time()

    for si in range(num_stages):
        sc = copy.deepcopy(scheduler)
        ode_steps = int(_ps(ode_steps_per_stage, si, 10))
        sc.set_timesteps(ode_steps, si, device=device, shift=shift)
        sk = float(sc.start_t[si])
        ek = float(sc.end_t[si])
        eff_si = si if g_bypass_stage3 else None

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

        # Mask at this stage (observed=1)
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

        # Per-stage parameters
        n_lang_k = int(_ps(num_langevin, si, 10))
        h_x_k = float(_ps(h_x, si, 0.1))
        h_e_k = float(_ps(h_epsilon, si, 0.01))
        lam_k = float(_ps(lambda_reg, si, 50.0))
        noi_k = float(_ps(noise_scale, si, 0.0))
        zeta_k = float(_ps(dps_kick_zeta, si, 0.0)) if dps_kick_zeta is not None else 0.0
        hr_k = float(_ps(h_x_obs_ratio, si, 1.0))
        lp_k = float(_ps(lambda_prox, si, 0.0))
        sk_eps_k = bool(_ps(skip_eps_update, si, False))
        reset_eps_k = bool(_ps(reset_eps_per_ode_step, si, False))
        srs_k = float(_ps(sigma_ref_sq, si, 0.01))

        for step_idx, T in enumerate(sc.Timesteps):
            tau = float(sc.t[step_idx].to(device))
            sig = compute_sigma_tau(tau, sk, ek)
            xtau = apply_H_tau(x1_k, tau, sk, ek, stage_idx=eff_si) + sig * eps_k

            vfn = make_velocity_fn(
                model, T, pe_combined, st, rope,
                do_cfg, guidance_scale, si,
            )

            if warm_restart:
                mu = vfn(xtau)
                xs_h = xtau - tau * mu
                xe_h = xtau + (1 - tau) * mu
                x1_k = direct_estimate_x1(xs_h, xe_h, sk, ek).detach().clone()
                if sig > 1e-8:
                    eps_k = (xtau - apply_H_tau(x1_k, tau, sk, ek, stage_idx=eff_si)) / sig
                else:
                    eps_k = torch.randn_like(x1_k)

            if reset_eps_k:
                eps_k = torch.randn_like(x1_k)

            # DPS pre-kick (hybrid)
            if zeta_k > 0:
                x1_k = dps_gradient_kick(x1_k, Ak, ATk, y, zeta_k)
                if sig > 1e-8:
                    eps_k = (xtau - apply_H_tau(x1_k, tau, sk, ek, stage_idx=eff_si)) / sig

            # Langevin inner
            if sig >= 0.01 and n_lang_k > 0:
                # eps reference for N3: upsample initial eps to current stage resolution
                if lambda_eps_prox > 0:
                    if eps0_init.shape[-2:] == eps_k.shape[-2:]:
                        eps_ref_here = eps0_init
                    else:
                        eps_ref_here = F.interpolate(eps0_init, size=eps_k.shape[-2:], mode="nearest")
                else:
                    eps_ref_here = None
                x1_k, eps_k = principle_langevin_v5(
                    x1_init=x1_k, eps_init=eps_k,
                    tau=tau, s_k=sk, e_k=ek,
                    velocity_fn=vfn, A_k_fn=Ak, AT_k_fn=ATk,
                    y=y, sigma_n=sigma_n,
                    h_x=h_x_k, h_epsilon=h_e_k,
                    lambda_x=lambda_x, lambda_reg=lam_k,
                    rho_s=rho_s, rho_e=rho_e,
                    cg_tol=cg_tol, cg_max_iter=cg_max_iter,
                    num_Langevin=n_lang_k, device=device,
                    stage_idx=eff_si, x1_init_mode=x1_init_mode,
                    noise_scale=noi_k,
                    mask_k=mask_k, h_x_obs_ratio=hr_k,
                    lambda_prox=lp_k,
                    skip_eps_update=sk_eps_k,
                    lambda_eps_norm=lambda_eps_norm,
                    mask_aware_eps=mask_aware_eps,
                    lambda_eps_prox=lambda_eps_prox,
                    eps_ref=eps_ref_here,
                    lambda_eps_inject=lambda_eps_inject,
                    sigma_ref_sq=srs_k,
                )

            # Per-step soft replacement at stage 3 (optional)
            if soft_replace_weight > 0 and h == 256:
                x1_k = soft_replace_weight * (mask_k * y + (1 - mask_k) * x1_k) + \
                       (1 - soft_replace_weight) * x1_k
                if sig > 1e-8:
                    eps_k = (xtau - apply_H_tau(x1_k, tau, sk, ek, stage_idx=eff_si)) / sig

            lat = apply_H_tau(x1_k, tau, sk, ek, stage_idx=eff_si) + sig * eps_k

    # Final WLS-CG denoise (optional): rebuild x_tau from current (x1_k, eps_k),
    # re-evaluate velocity, then project x1 via WLS-CG. Uses the LAST stage's
    # (sk, ek, tau, eff_si, vfn) which are still in scope after the loop.
    if final_denoise:
        xtau_f = apply_H_tau(x1_k, tau, sk, ek, stage_idx=eff_si) + sig * eps_k
        mu_f   = vfn(xtau_f)
        xs_f   = xtau_f - tau * mu_f
        xe_f   = xtau_f + (1.0 - tau) * mu_f
        x1_k   = wls_estimate_x1(
            xs_f, xe_f, sk, ek,
            rho_s, rho_e, lambda_x,
            cg_tol, cg_max_iter,
            stage_idx=eff_si,
        ).detach()

    # Terminal replacement (optional, runs AFTER final_denoise so tr can polish observed pixels)
    if terminal_replace_weight > 0:
        x1_k = terminal_replacement(x1_k, y, mask_full, terminal_replace_weight)

    elapsed = time.time() - t0
    xf = x1_k.detach()
    # Publication PSNR: 10*log10(MAX²/MSE). Data in [-1,1] → MAX=2 → 10*log10(4/MSE).
    # Equivalent to -10*log10(MSE_[0,1]) after denormalizing to [0,1]. Reference:
    # test_psnr_inpainting.py (same formula as DPS/PGDM inpainting papers).
    Af, _ = make_Ak_fns(operator, y, (B, 3, 256, 256), device)
    diff2 = (xf - gt) ** 2
    C = xf.shape[1]

    # Full-image PSNR
    mse_all = diff2.mean()
    psnr_all = (10 * torch.log10(4.0 / mse_all.clamp(min=1e-12))).item()

    # Region PSNRs — divide by #valid positions (B·C·N_region), NOT B·C·H·W
    n_obs_total   = mask_full.sum().item() * C            # total observed scalars in batch
    n_unobs_total = (1.0 - mask_full).sum().item() * C
    mse_obs   = (mask_full       * diff2).sum() / max(n_obs_total,   1)
    mse_unobs = ((1 - mask_full) * diff2).sum() / max(n_unobs_total, 1)
    psnr_obs   = (10 * torch.log10(4.0 / mse_obs.clamp(min=1e-12))).item()
    psnr_unobs = (10 * torch.log10(4.0 / mse_unobs.clamp(min=1e-12))).item()

    res = (y - Af(xf)).pow(2).sum().item()
    hf_u = hf_energy(xf * (1 - mask_full[0:1]))
    # Backwards-compat return: (xf, psnr_obs, psnr_all, res, elapsed, hf_u).
    # Now all three PSNRs use publication convention (MAX=2 for [-1,1] data).
    return xf.cpu(), psnr_obs, psnr_all, res, elapsed, hf_u
