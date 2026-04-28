#!/usr/bin/env python
"""
Module-level diagnostics: test each component in isolation before running full pipeline.
Tests: measurement, A_k adjoint, G vs DownUp, gradient direction, step size, Langevin convergence.
"""
import sys, os, math, json, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder

from inpaintingStart import get_operator
from ms_posterior_sampling_utils import (
    DownUp_operation, Up_k, center_crop_arr, sample_block_noise,
    objective_split_prior, langevin_update_split_prior,
    get_x1_and_noise_with_stage_start_and_end2, pred_x1_x0_with_vt,
    get_xt_from_x1_sameStage, get_lr,
)
from ms_posterior_sampling_article_version_utils import (
    apply_G, apply_H_tau, apply_HT_tau, compute_sigma_tau,
    wls_estimate_x1, make_Ak_fns, cg_solve, _langevin_step,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
RESULT_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULT_DIR, exist_ok=True)

# Quick log helper
LOG = []
def log(msg):
    print(msg)
    LOG.append(msg)

def save_log():
    with open(os.path.join(RESULT_DIR, "diagnose_modules.log"), "w") as f:
        f.write("\n".join(LOG))


# ═══════════════════════════════════════════════════════════════════════
# Setup: load one image, create operator, measurement
# ═══════════════════════════════════════════════════════════════════════
def setup():
    resolution = 256
    sigma_n = 0.05

    # Try loading GT from existing results; fall back to synthetic
    torch.manual_seed(20000120)
    pt_paths = [
        "trajectory_videos/posterior_sampling/baseline_05/baseline_05_mode-f_x1.pt",
        "article_v_results/test_box_inpainting/test_box_inpainting_mode-f_x1.pt",
    ]
    gt = None
    for pp in pt_paths:
        if os.path.exists(pp):
            d = torch.load(pp, map_location="cpu", weights_only=False)
            gt = d["gt"][:2].to(DEVICE)
            log(f"  Loaded GT from {pp}, shape={gt.shape}")
            break
    if gt is None:
        log("  No saved GT found — using synthetic checkerboard + gradient image")
        # Create structured synthetic images for reliable testing
        gt = torch.zeros(2, 3, resolution, resolution, device=DEVICE)
        # Image 0: checkerboard pattern
        for i in range(0, resolution, 32):
            for j in range(0, resolution, 32):
                if (i//32 + j//32) % 2 == 0:
                    gt[0, :, i:i+32, j:j+32] = 0.8
                else:
                    gt[0, :, i:i+32, j:j+32] = -0.8
        # Image 1: smooth gradient
        gt[1, 0] = torch.linspace(-1, 1, resolution, device=DEVICE).unsqueeze(0).expand(resolution, -1)
        gt[1, 1] = torch.linspace(-1, 1, resolution, device=DEVICE).unsqueeze(1).expand(-1, resolution)
        gt[1, 2] = 0.5

    operator = get_operator(
        "inpainting", mask_type="box",
        mask_len_range=(80, 160), mask_prob_range=None,
        resolution=resolution, device=DEVICE, sigma=sigma_n,
    )

    y_call = operator(gt).detach()
    y_measure = operator.measure(gt).detach()

    return gt, operator, y_call, y_measure, sigma_n


# ═══════════════════════════════════════════════════════════════════════
# TEST 1: Measurement sanity
# ═══════════════════════════════════════════════════════════════════════
def test_measurement(gt, operator, y_call, y_measure, sigma_n):
    log("\n" + "="*70)
    log("TEST 1: Measurement sanity")
    log("="*70)

    mask = operator.get_mask(x=gt).float()
    coverage = mask.mean().item()
    log(f"  Mask coverage (fraction of OBSERVED pixels): {coverage:.4f}")
    log(f"  Mask shape: {mask.shape}, gt shape: {gt.shape}")

    # y_call should equal mask * gt
    diff_call = (y_call - mask * gt).abs().max().item()
    log(f"  |y_call - mask*gt|_inf = {diff_call:.2e}")

    # y_measure should be close to mask * gt + noise
    diff_measure = (y_measure - mask * gt).abs().mean().item()
    log(f"  |y_measure - mask*gt|_mean = {diff_measure:.4f} (expect ~{sigma_n*0.8:.4f} for sigma_n={sigma_n})")

    # Save visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i in range(min(2, gt.shape[0])):
        img_gt = (gt[i].cpu().permute(1,2,0)*0.5+0.5).clamp(0,1).numpy()
        img_y_call = (y_call[i].cpu().permute(1,2,0)*0.5+0.5).clamp(0,1).numpy()
        img_y_meas = (y_measure[i].cpu().permute(1,2,0)*0.5+0.5).clamp(0,1).numpy()
        img_mask = mask[0].cpu().squeeze().numpy()  # mask is (1,1,H,W), same for all

        axes[i,0].imshow(img_gt); axes[i,0].set_title("GT")
        axes[i,1].imshow(img_y_call); axes[i,1].set_title("y (call, noiseless)")
        axes[i,2].imshow(img_y_meas); axes[i,2].set_title("y (measure, noisy)")
        axes[i,3].imshow(img_mask, cmap="gray"); axes[i,3].set_title(f"mask ({coverage:.2%})")
    for ax in axes.flat: ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "test1_measurement.png"), dpi=150)
    plt.close()
    log("  Saved test1_measurement.png")

    return mask


# ═══════════════════════════════════════════════════════════════════════
# TEST 2: A_k / AT_k adjoint test at all stage resolutions
# ═══════════════════════════════════════════════════════════════════════
def test_Ak_adjoint(operator, y_call, gt):
    log("\n" + "="*70)
    log("TEST 2: A_k / AT_k adjoint test")
    log("="*70)

    B = gt.shape[0]
    stage_sizes = [(32,32), (64,64), (128,128), (256,256)]

    for k, (h, w) in enumerate(stage_sizes):
        stage_shape = (B, 3, h, w)
        A_k, AT_k = make_Ak_fns(operator, y_call, stage_shape, DEVICE)

        x = torch.randn(B, 3, h, w, device=DEVICE)
        r = torch.randn_like(y_call)

        Ax = A_k(x)
        ATr = AT_k(r)

        lhs = (Ax * r).sum().item()
        rhs = (x * ATr).sum().item()
        rel_err = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-12)

        log(f"  Stage {k} ({h}x{w}): <Ax,r>={lhs:.6f}, <x,A^T r>={rhs:.6f}, rel_err={rel_err:.2e}")

        # Also check A_k(gt_at_stage) ≈ y
        gt_at_stage = F.interpolate(gt, size=(h, w), mode="bilinear", align_corners=False)
        Agt = A_k(gt_at_stage)
        residual = (Agt - y_call).pow(2).sum().sqrt().item()
        log(f"    ||A_k(gt_downscaled) - y||_2 = {residual:.4f}")


# ═══════════════════════════════════════════════════════════════════════
# TEST 3: G (new) vs DownUp_operation (old) — the stage 3 identity issue
# ═══════════════════════════════════════════════════════════════════════
def test_G_vs_DownUp(gt):
    log("\n" + "="*70)
    log("TEST 3: apply_G (new) vs DownUp_operation (old)")
    log("="*70)

    stage_sizes = [(32,32), (64,64), (128,128), (256,256)]

    for k, (h, w) in enumerate(stage_sizes):
        x = F.interpolate(gt, size=(h, w), mode="bilinear", align_corners=False)

        g_new = apply_G(x)

        # Old DownUp with proper kwargs to avoid positional arg bug
        if k == 3:
            # stage_idx=3 returns identity in old code
            g_old = DownUp_operation(x, scale_factor=2, stage_idx=3)
        else:
            g_old = DownUp_operation(x, scale_factor=2, stage_idx=k)

        diff = (g_new - g_old).abs().max().item()
        g_new_diff = (g_new - x).abs().max().item()
        g_old_diff = (g_old - x).abs().max().item()

        log(f"  Stage {k} ({h}x{w}):")
        log(f"    |G_new - G_old|_inf = {diff:.6f}")
        log(f"    |G_new - x|_inf = {g_new_diff:.6f} (new always low-pass)")
        log(f"    |G_old - x|_inf = {g_old_diff:.6f} {'(IDENTITY at stage 3!)' if k==3 else ''}")

    # G self-adjoint test
    log("\n  G self-adjoint test:")
    for h in [32, 64, 128, 256]:
        x = torch.randn(2, 3, h, h, device=DEVICE)
        y_rand = torch.randn(2, 3, h, h, device=DEVICE)
        lhs = (apply_G(x) * y_rand).sum().item()
        rhs = (x * apply_G(y_rand)).sum().item()
        rel_err = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-12)
        log(f"    {h}x{h}: <Gx,y>={lhs:.4f}, <x,Gy>={rhs:.4f}, rel_err={rel_err:.2e}")


# ═══════════════════════════════════════════════════════════════════════
# TEST 4: Data-consistency gradient comparison (old autograd vs new explicit)
# ═══════════════════════════════════════════════════════════════════════
def test_gradient_comparison(gt, operator, y_call, sigma_n):
    log("\n" + "="*70)
    log("TEST 4: Data-consistency gradient: old autograd vs new explicit")
    log("="*70)

    B = gt.shape[0]

    # Use a scheduler to get realistic start_t / end_t
    from pixelflow.scheduling_pixelflow import PixelFlowScheduler
    from omegaconf import OmegaConf
    config = OmegaConf.load("pretrained_models/c2img/config.yaml")
    scheduler = PixelFlowScheduler(
        config.scheduler.num_train_timesteps,
        num_stages=config.scheduler.num_stages, gamma=-1/3,
    )

    stage_sizes = [(32,32), (64,64), (128,128), (256,256)]

    for k, (h, w) in enumerate(stage_sizes):
        import copy
        sched = copy.deepcopy(scheduler)
        sched.set_timesteps(10, k, device=DEVICE, shift=1.0)
        start_t = sched.start_t[k]
        end_t = sched.end_t[k]

        # Create a plausible x1 (use downscaled gt + noise)
        x1 = F.interpolate(gt, size=(h, w), mode="bilinear", align_corners=False)
        x1 = x1 + 0.1 * torch.randn_like(x1)

        # ── OLD gradient via autograd ──
        # objective_split_prior uses Up_k(x0hat, y) which upsamples to y resolution
        # and operator() which applies mask
        # Also uses DownUp_operation(x0hat) — beware the positional arg bug
        x1_ag = x1.clone().detach().requires_grad_(True)

        # Fake stage endpoints from x1 (approximate)
        noise = torch.randn(B, 3, h, w, device=DEVICE)
        t_mid = (float(start_t) + float(end_t)) / 2  # middle of stage
        xs_fake = float(start_t) * DownUp_operation(x1.detach(), scale_factor=2, stage_idx=k if k < 3 else None) + (1.0 - float(start_t)) * noise
        xe_fake = float(end_t) * x1.detach() + (1.0 - float(end_t)) * noise

        loss_old, l1, l2, l3 = objective_split_prior(
            x1_ag, operator, end_t, start_t, xe_fake, xs_fake,
            y_call, sigma_n, None, lambda_prior=0.00001,
        )
        grad_old = torch.autograd.grad(loss_old, x1_ag)[0]

        # ── NEW gradient (explicit) ──
        stage_shape = (B, 3, h, w)
        A_k, AT_k = make_Ak_fns(operator, y_call, stage_shape, DEVICE)

        residual = y_call - A_k(x1)
        eta = sigma_n
        grad_dc_new = -(1.0 / eta**2) * AT_k(residual)  # negative because we want descent

        # Compare directions (cosine similarity)
        cos_sim = F.cosine_similarity(
            grad_old.reshape(B, -1), grad_dc_new.reshape(B, -1), dim=1
        ).mean().item()

        old_norm = grad_old.norm().item()
        new_dc_norm = grad_dc_new.norm().item()

        log(f"\n  Stage {k} ({h}x{w}), start_t={float(start_t):.4f}, end_t={float(end_t):.4f}:")
        log(f"    Old loss: total={float(loss_old):.4f}, l1={float(l1):.4f}, l2={float(l2):.4f}, l3={float(l3):.4f}")
        log(f"    Old grad norm: {old_norm:.4f}")
        log(f"    New DC grad norm (1/eta^2 * AT(residual)): {new_dc_norm:.4f}")
        log(f"    Cosine similarity (old_grad, new_dc_grad): {cos_sim:.4f}")
        log(f"    Ratio old/new norms: {old_norm/max(new_dc_norm,1e-12):.4f}")

        # Effective step comparison
        lr_old = get_lr(0.5, lr_base=0.02, lr_min_ratio=0.01)
        h_x_new = 1e-3
        old_delta = lr_old * old_norm
        new_delta = (h_x_new / 2) * new_dc_norm  # ignoring CG preconditioning
        log(f"    Old effective delta (lr={lr_old:.4f}): {old_delta:.4f}")
        log(f"    New effective delta (h_x/2={h_x_new/2:.4f}): {new_delta:.4f}")
        log(f"    Old/New delta ratio: {old_delta/max(new_delta,1e-12):.1f}x")


# ═══════════════════════════════════════════════════════════════════════
# TEST 5: Single Langevin step comparison (old vs new)
# ═══════════════════════════════════════════════════════════════════════
def test_single_langevin_step(gt, operator, y_call, sigma_n):
    log("\n" + "="*70)
    log("TEST 5: Single Langevin step — residual reduction")
    log("="*70)

    B = gt.shape[0]
    from pixelflow.scheduling_pixelflow import PixelFlowScheduler
    from omegaconf import OmegaConf
    import copy

    config = OmegaConf.load("pretrained_models/c2img/config.yaml")
    scheduler = PixelFlowScheduler(
        config.scheduler.num_train_timesteps,
        num_stages=config.scheduler.num_stages, gamma=-1/3,
    )

    # Test at stage 3 (256x256) where the difference is most apparent
    k = 3
    h, w = 256, 256
    sched = copy.deepcopy(scheduler)
    sched.set_timesteps(10, k, device=DEVICE, shift=1.0)
    start_t = sched.start_t[k]
    end_t = sched.end_t[k]
    tau = float(sched.t[5])  # middle timestep

    log(f"  Stage {k}, tau={tau:.4f}, start_t={float(start_t):.4f}, end_t={float(end_t):.4f}")

    # Start from gt + noise as x1 init
    x1_init = gt + 0.3 * torch.randn_like(gt)

    A_k, AT_k = make_Ak_fns(operator, y_call, (B, 3, h, w), DEVICE)

    residual_before = (y_call - A_k(x1_init)).pow(2).sum().item()
    log(f"  Residual before: {residual_before:.4f}")

    # ── OLD: multiple lr values ──
    noise_fake = torch.randn_like(x1_init)
    xs_fake = float(start_t) * DownUp_operation(x1_init.detach(), scale_factor=2, stage_idx=3) + (1-float(start_t)) * noise_fake
    xe_fake = float(end_t) * x1_init.detach() + (1-float(end_t)) * noise_fake

    for lr in [0.001, 0.005, 0.01, 0.02, 0.05]:
        x1_old = x1_init.clone()
        for _ in range(10):
            x1_old, _ = langevin_update_split_prior(
                x1_old, y_call, operator, end_t, start_t, xe_fake, xs_fake,
                sigma_n, step_size=lr, proj=False, o_M=None, lambda_prior=0.00001, device=DEVICE,
            )
        res_after = (y_call - A_k(x1_old)).pow(2).sum().item()
        log(f"  OLD (lr={lr:.3f}, 10 steps): residual={res_after:.4f}, reduction={residual_before-res_after:.4f}")

    # ── NEW: multiple h_x values ──
    s_k = float(start_t)
    e_k = float(end_t)
    sigma_tau = compute_sigma_tau(tau, s_k, e_k)

    for h_x in [0.001, 0.01, 0.05, 0.1, 0.5]:
        x1_new = x1_init.clone()
        eps_new = torch.randn_like(x1_init)
        x_tau_new = apply_H_tau(x1_new, tau, s_k, e_k) + sigma_tau * eps_new

        # We need a velocity_fn — use identity as placeholder (no model needed)
        # This isolates the Langevin update from model quality
        def fake_velocity_fn(x):
            # Return zero velocity — just test data-consistency correction
            return torch.zeros_like(x)

        for _ in range(10):
            x1_new, eps_new, x_tau_new, step_log = _langevin_step(
                x1_new, eps_new, x_tau_new, tau, s_k, e_k, sigma_tau,
                fake_velocity_fn, A_k, AT_k, y_call, sigma_n,
                h_x, 1e-3, 0.01, 0.01,
                1.0, 1.0, 1e-5, 50,
            )
        res_after = (y_call - A_k(x1_new)).pow(2).sum().item()
        log(f"  NEW (h_x={h_x:.3f}, 10 steps, zero velocity): residual={res_after:.4f}, reduction={residual_before-res_after:.4f}")


# ═══════════════════════════════════════════════════════════════════════
# TEST 6: sigma_tau analysis across stages
# ═══════════════════════════════════════════════════════════════════════
def test_sigma_tau_analysis():
    log("\n" + "="*70)
    log("TEST 6: sigma_tau and 1/sigma_tau^2 across stages")
    log("="*70)

    from pixelflow.scheduling_pixelflow import PixelFlowScheduler
    from omegaconf import OmegaConf
    import copy

    config = OmegaConf.load("pretrained_models/c2img/config.yaml")
    scheduler = PixelFlowScheduler(
        config.scheduler.num_train_timesteps,
        num_stages=config.scheduler.num_stages, gamma=-1/3,
    )

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for k in range(4):
        sched = copy.deepcopy(scheduler)
        sched.set_timesteps(10, k, device="cpu", shift=1.0)
        start_t = float(sched.start_t[k])
        end_t = float(sched.end_t[k])

        taus = [float(sched.t[i]) for i in range(len(sched.t)-1)]
        sigmas = [compute_sigma_tau(tau, start_t, end_t) for tau in taus]
        inv_sigma2 = [1.0/max(s, 0.01)**2 for s in sigmas]

        log(f"\n  Stage {k}: start_t={start_t:.4f}, end_t={end_t:.4f}")
        for i, (tau, sig, inv) in enumerate(zip(taus, sigmas, inv_sigma2)):
            skip = "SKIP" if sig < 0.01 else ""
            log(f"    step {i}: tau={tau:.4f}, sigma_tau={sig:.6f}, 1/sigma^2={inv:.1f} {skip}")

        axes[k].plot(taus, sigmas, 'b-o', label='sigma_tau')
        axes[k].axhline(y=0.01, color='r', linestyle='--', label='skip threshold')
        axes[k].set_title(f"Stage {k}")
        axes[k].set_xlabel("tau")
        axes[k].set_ylabel("sigma_tau")
        axes[k].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "test6_sigma_tau.png"), dpi=150)
    plt.close()
    log("  Saved test6_sigma_tau.png")


# ═══════════════════════════════════════════════════════════════════════
# TEST 7: H_tau / x_tau reconstruction consistency
# ═══════════════════════════════════════════════════════════════════════
def test_Htau_consistency():
    log("\n" + "="*70)
    log("TEST 7: H_tau consistency and old xt reconstruction comparison")
    log("="*70)

    from pixelflow.scheduling_pixelflow import PixelFlowScheduler
    from omegaconf import OmegaConf
    import copy

    config = OmegaConf.load("pretrained_models/c2img/config.yaml")
    scheduler = PixelFlowScheduler(
        config.scheduler.num_train_timesteps,
        num_stages=config.scheduler.num_stages, gamma=-1/3,
    )

    for k in range(4):
        h = 32 * (2**k)
        sched = copy.deepcopy(scheduler)
        sched.set_timesteps(10, k, device=DEVICE, shift=1.0)
        start_t = sched.start_t[k]
        end_t = sched.end_t[k]
        s_k = float(start_t)
        e_k = float(end_t)

        x1 = torch.randn(2, 3, h, h, device=DEVICE)
        noise = torch.randn(2, 3, h, h, device=DEVICE)

        tau = float(sched.t[5])
        sigma_tau = compute_sigma_tau(tau, s_k, e_k)
        xt_new = apply_H_tau(x1, tau, s_k, e_k) + sigma_tau * noise

        # Old get_xt_from_x1_sameStage has DownUp_operation(D_x1, stage_idx) positional arg bug:
        # stage_idx=0 → scale_factor=0 → ZeroDivisionError
        # stage_idx=1 → scale_factor=1 → no actual downscale (trivial)
        # So we manually compute the old formula for comparison
        if k == 3:
            # Old: DownUp returns identity at stage 3
            DU_x1 = x1
        else:
            DU_x1 = DownUp_operation(x1, scale_factor=2, stage_idx=k)

        # Old formula: xt = t*(e*x1 + (1-e)*noise) + (1-t)*(s*DU(x1) + (1-s)*noise)
        xt_old = tau * (e_k * x1 + (1-e_k) * noise) + (1-tau) * (s_k * DU_x1 + (1-s_k) * noise)

        diff = (xt_old - xt_new).abs().max().item()
        log(f"  Stage {k} ({h}x{h}): |xt_old - xt_new|_inf = {diff:.6f}"
            f"  {'(EXPECTED DIFF: G!=DownUp at stage 3)' if k==3 else '(should be ~0)'}")


# ═══════════════════════════════════════════════════════════════════════
# TEST 8: WLS x1 estimate quality
# ═══════════════════════════════════════════════════════════════════════
def test_wls_estimate():
    log("\n" + "="*70)
    log("TEST 8: WLS x1 estimate quality vs direct x1 prediction")
    log("="*70)

    from pixelflow.scheduling_pixelflow import PixelFlowScheduler
    from omegaconf import OmegaConf
    import copy

    config = OmegaConf.load("pretrained_models/c2img/config.yaml")
    scheduler = PixelFlowScheduler(
        config.scheduler.num_train_timesteps,
        num_stages=config.scheduler.num_stages, gamma=-1/3,
    )

    for k in range(4):
        h = 32 * (2**k)
        sched = copy.deepcopy(scheduler)
        sched.set_timesteps(10, k, device=DEVICE, shift=1.0)
        s_k = float(sched.start_t[k])
        e_k = float(sched.end_t[k])
        tau = float(sched.t[5])

        # Ground truth x1 and noise
        x1_gt = torch.randn(2, 3, h, h, device=DEVICE)
        noise = torch.randn(2, 3, h, h, device=DEVICE)

        # Generate "perfect" endpoints from GT
        xs_hat = s_k * apply_G(x1_gt) + (1-s_k) * noise
        xe_hat = e_k * x1_gt + (1-e_k) * noise

        # WLS estimate
        x1_wls = wls_estimate_x1(xs_hat, xe_hat, s_k, e_k,
                                  rho_s=1.0, rho_e=1.0,
                                  lambda_x=0.01, cg_tol=1e-5, cg_max_iter=50)

        # Direct estimate (old-style): x1 = ((1-s)*xe - (1-e)*xs) / (e-s)
        x1_direct = ((1-s_k) * xe_hat - (1-e_k) * xs_hat) / max(e_k - s_k, 1e-8)

        err_wls = (x1_wls - x1_gt).pow(2).mean().sqrt().item()
        err_direct = (x1_direct - x1_gt).pow(2).mean().sqrt().item()

        log(f"  Stage {k} ({h}x{h}): s_k={s_k:.4f}, e_k={e_k:.4f}")
        log(f"    WLS x1 error (RMSE): {err_wls:.6f}")
        log(f"    Direct x1 error (RMSE): {err_direct:.6f}")
        log(f"    Ratio WLS/Direct: {err_wls/max(err_direct,1e-12):.4f}")


# ═══════════════════════════════════════════════════════════════════════
# TEST 9: Langevin convergence curves — old vs new, parameter sweep
# ═══════════════════════════════════════════════════════════════════════
def test_langevin_convergence(gt, operator, y_call, sigma_n):
    log("\n" + "="*70)
    log("TEST 9: Langevin convergence — old vs new, 50 steps")
    log("="*70)

    B = gt.shape[0]
    from pixelflow.scheduling_pixelflow import PixelFlowScheduler
    from omegaconf import OmegaConf
    import copy

    config = OmegaConf.load("pretrained_models/c2img/config.yaml")
    scheduler = PixelFlowScheduler(
        config.scheduler.num_train_timesteps,
        num_stages=config.scheduler.num_stages, gamma=-1/3,
    )

    # Test at full resolution (stage 3)
    k = 3
    h, w = 256, 256
    sched = copy.deepcopy(scheduler)
    sched.set_timesteps(10, k, device=DEVICE, shift=1.0)
    start_t = sched.start_t[k]
    end_t = sched.end_t[k]
    s_k = float(start_t)
    e_k = float(end_t)
    tau = float(sched.t[3])  # early in stage for larger sigma
    sigma_tau = compute_sigma_tau(tau, s_k, e_k)

    log(f"  Stage {k}, tau={tau:.4f}, sigma_tau={sigma_tau:.6f}")

    x1_init = gt + 0.3 * torch.randn_like(gt)
    A_k, AT_k = make_Ak_fns(operator, y_call, (B, 3, h, w), DEVICE)

    noise_fake = torch.randn_like(gt)
    xs_fake = s_k * DownUp_operation(x1_init.detach(), scale_factor=2, stage_idx=3) + (1-s_k) * noise_fake
    xe_fake = e_k * x1_init.detach() + (1-e_k) * noise_fake

    def fake_velocity_fn(x):
        return torch.zeros_like(x)

    num_steps = 50
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # OLD: various lr
    for lr in [0.005, 0.01, 0.02]:
        residuals = []
        x1_cur = x1_init.clone()
        for i in range(num_steps):
            res = (y_call - A_k(x1_cur)).pow(2).sum().item()
            residuals.append(res)
            x1_cur, _ = langevin_update_split_prior(
                x1_cur, y_call, operator, end_t, start_t, xe_fake, xs_fake,
                sigma_n, step_size=lr, proj=False, o_M=None, lambda_prior=0.00001, device=DEVICE,
            )
        ax.plot(residuals, label=f"OLD lr={lr}")

    # NEW: various h_x
    for h_x in [0.01, 0.05, 0.1, 0.5, 1.0]:
        residuals = []
        x1_cur = x1_init.clone()
        eps_cur = torch.randn_like(x1_init)
        x_tau_cur = apply_H_tau(x1_cur, tau, s_k, e_k) + sigma_tau * eps_cur

        for i in range(num_steps):
            res = (y_call - A_k(x1_cur)).pow(2).sum().item()
            residuals.append(res)
            x1_cur, eps_cur, x_tau_cur, _ = _langevin_step(
                x1_cur, eps_cur, x_tau_cur, tau, s_k, e_k, sigma_tau,
                fake_velocity_fn, A_k, AT_k, y_call, sigma_n,
                h_x, 1e-3, 0.01, 0.01,
                1.0, 1.0, 1e-5, 50,
            )
        ax.plot(residuals, '--', label=f"NEW h_x={h_x}")

    ax.set_xlabel("Langevin step")
    ax.set_ylabel("||A(x1) - y||^2")
    ax.set_title(f"Langevin convergence at stage {k}, tau={tau:.3f}")
    ax.legend()
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "test9_langevin_convergence.png"), dpi=150)
    plt.close()
    log("  Saved test9_langevin_convergence.png")


# ═══════════════════════════════════════════════════════════════════════
# TEST 10: lambda_reg sensitivity — does CG preconditioning help or hurt?
# ═══════════════════════════════════════════════════════════════════════
def test_lambda_reg_sensitivity(gt, operator, y_call, sigma_n):
    log("\n" + "="*70)
    log("TEST 10: lambda_reg sensitivity in CG preconditioner")
    log("="*70)

    B = gt.shape[0]
    from pixelflow.scheduling_pixelflow import PixelFlowScheduler
    from omegaconf import OmegaConf
    import copy

    config = OmegaConf.load("pretrained_models/c2img/config.yaml")
    scheduler = PixelFlowScheduler(
        config.scheduler.num_train_timesteps,
        num_stages=config.scheduler.num_stages, gamma=-1/3,
    )

    k = 3
    h, w = 256, 256
    sched = copy.deepcopy(scheduler)
    sched.set_timesteps(10, k, device=DEVICE, shift=1.0)
    s_k = float(sched.start_t[k])
    e_k = float(sched.end_t[k])
    tau = float(sched.t[3])
    sigma_tau = compute_sigma_tau(tau, s_k, e_k)

    x1_init = gt + 0.3 * torch.randn_like(gt)
    A_k, AT_k = make_Ak_fns(operator, y_call, (B, 3, h, w), DEVICE)

    def fake_velocity_fn(x):
        return torch.zeros_like(x)

    results = {}
    for h_x in [0.1, 0.5]:
        for lam_reg in [0.001, 0.01, 0.1, 1.0, 10.0]:
            x1_cur = x1_init.clone()
            eps_cur = torch.randn_like(x1_init)
            x_tau_cur = apply_H_tau(x1_cur, tau, s_k, e_k) + sigma_tau * eps_cur

            residuals = []
            for i in range(30):
                res = (y_call - A_k(x1_cur)).pow(2).sum().item()
                residuals.append(res)
                x1_cur, eps_cur, x_tau_cur, _ = _langevin_step(
                    x1_cur, eps_cur, x_tau_cur, tau, s_k, e_k, sigma_tau,
                    fake_velocity_fn, A_k, AT_k, y_call, sigma_n,
                    h_x, 1e-3, 0.01, lam_reg,
                    1.0, 1.0, 1e-5, 50,
                )
            results[(h_x, lam_reg)] = residuals[-1]
            log(f"  h_x={h_x}, lambda_reg={lam_reg}: final residual={residuals[-1]:.4f}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    log("="*70)
    log("PRINCIPLE Debug: Module-Level Diagnostics")
    log("="*70)

    gt, operator, y_call, y_measure, sigma_n = setup()
    log(f"Setup complete: gt={gt.shape}, y={y_call.shape}, device={DEVICE}")

    mask = test_measurement(gt, operator, y_call, y_measure, sigma_n)
    test_Ak_adjoint(operator, y_call, gt)
    test_G_vs_DownUp(gt)
    test_gradient_comparison(gt, operator, y_call, sigma_n)
    test_single_langevin_step(gt, operator, y_call, sigma_n)
    test_sigma_tau_analysis()
    test_Htau_consistency()
    test_wls_estimate()
    test_langevin_convergence(gt, operator, y_call, sigma_n)
    test_lambda_reg_sensitivity(gt, operator, y_call, sigma_n)

    save_log()
    log("\nAll tests complete. Results in debug_IP/results/")
