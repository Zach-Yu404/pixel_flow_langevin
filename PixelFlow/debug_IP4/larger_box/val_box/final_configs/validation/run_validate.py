"""Validate final_denoise + tr matrix on balanced_perceptual base.

Matrix: (final_denoise=False/True) × (tr=0/1) = 4 configs.
Base = balanced_perceptual: h_eps=0.01, h_x=0.1, L=10, srs=1e-4, lambda_reg=lambda_prox=150,
       guidance=2, noise=1.

Same 2-image baseline_05 GT, 128x128 box, seed=7919.
"""
import os, sys, json, time

HERE = os.path.dirname(os.path.abspath(__file__))            # validation
FC   = os.path.dirname(HERE)                                 # final_configs
LB   = os.path.dirname(FC)                                   # larger_box
DBG  = os.path.dirname(LB)                                   # debug_IP4
REPO = os.path.dirname(DBG)                                  # PixelFlow
sys.path.insert(0, DBG); sys.path.insert(0, REPO)
os.chdir(REPO)

import torch, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import piq

from inpaintingStart import get_operator
from pixelflow.utils import config as config_utils
from ms_sampler_v5 import run_ip4, hf_energy


BASE = dict(
    h_epsilon=0.01,
    h_x=0.1,
    num_langevin=10,
    lambda_reg=150.0,
    lambda_prox=150.0,
    noise_scale=1.0,
    guidance_scale=2.0,
    sigma_ref_sq=1e-4,
    terminal_replace_weight=0.0,
    final_denoise=False,
)


def cfg(**ovr):
    out = dict(BASE); out.update(ovr); return out


CONFIGS = [
    ("base_fd0_tr0", cfg()),
    ("base_fd1_tr0", cfg(final_denoise=True)),
    ("base_fd0_tr1", cfg(terminal_replace_weight=1.0)),
    ("base_fd1_tr1", cfg(final_denoise=True, terminal_replace_weight=1.0)),
]


def to_img(t):
    return (t.cpu().permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1).numpy()


def main():
    DEVICE = "cuda:0"
    config = OmegaConf.load("pretrained_models/c2img/config.yaml")
    model = config_utils.instantiate_from_config(config.model).to(DEVICE)
    ckpt = torch.load("pretrained_models/c2img/model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt, strict=True); model.eval()

    pt = "trajectory_videos/posterior_sampling/baseline_05/baseline_05_mode-f_x1.pt"
    gt = torch.load(pt, map_location="cpu", weights_only=False)["gt"][:2].to(DEVICE)
    B = gt.shape[0]; sigma_n = 0.05

    torch.manual_seed(7919)
    operator = get_operator("inpainting", resolution=256, device=DEVICE, sigma=sigma_n,
                            mask_type="box", mask_len_range=(128, 129), mask_prob_range=None)
    y = operator(gt).detach()
    mask = operator.get_mask(x=gt).float().to(DEVICE)
    if mask.shape[0] == 1 and B > 1: mask = mask.expand(B, -1, -1, -1)
    gt_hf = hf_energy(gt * (1 - mask[0:1]))
    print(f"GT HF={gt_hf:.4f}", flush=True)

    lpips_fn = piq.LPIPS(replace_pooling=True).to(DEVICE)

    rows = []
    for name, kw in CONFIGS:
        print(f"\n=== {name} === {kw}", flush=True)
        t0 = time.time()
        xf, _, _, res, _, hf = run_ip4(
            model, config, gt, y, operator, sigma_n, DEVICE,
            class_label=10, seed=20000120, **kw,
        )
        elapsed = time.time() - t0
        xfD = xf.to(DEVICE)
        recon_01 = ((xfD + 1) / 2).clamp(0, 1)
        gt_01    = ((gt   + 1) / 2).clamp(0, 1)
        psnrs, ssims, lpipss, psnr_us = [], [], [], []
        with torch.no_grad():
            for b in range(B):
                r1, g1 = recon_01[b:b+1], gt_01[b:b+1]
                psnrs.append(piq.psnr(r1, g1, data_range=1.0).item())
                ssims.append(piq.ssim(r1, g1, data_range=1.0).item())
                lpipss.append(lpips_fn(r1, g1).item())
                m = (1 - mask[b:b+1])
                d2 = (r1 - g1).pow(2)
                n_unobs = m.sum().item() * r1.shape[1]
                mse_u = (m * d2).sum().item() / max(n_unobs, 1)
                psnr_us.append(-10 * float(np.log10(max(mse_u, 1e-12))))
        psnr, ssim, lpips, psnr_u = np.mean(psnrs), np.mean(ssims), np.mean(lpipss), np.mean(psnr_us)
        dhf = float(abs(hf - gt_hf))
        print(f"  PSNR={psnr:.2f}  SSIM={ssim:.4f}  LPIPS={lpips:.4f}  PSNR_u={psnr_u:.2f}  |dHF|={dhf:.3f}  t={elapsed:.0f}s", flush=True)
        rows.append(dict(name=name, kw=kw, psnr=psnr, ssim=ssim, lpips=lpips,
                         psnr_unobs=psnr_u, hf=float(hf), dhf=dhf, gt_hf=float(gt_hf), time=elapsed))

        # also save image
        fig, axes = plt.subplots(B, 3, figsize=(12, 4*B))
        if B == 1: axes = axes[None, :]
        for b in range(B):
            axes[b,0].imshow(to_img(gt[b])); axes[b,0].set_title("GT", fontsize=9); axes[b,0].axis("off")
            axes[b,1].imshow(to_img((mask[b]*gt[b] + (1-mask[b])*(-torch.ones_like(gt[b]))).cpu()))
            axes[b,1].set_title("Meas", fontsize=9); axes[b,1].axis("off")
            axes[b,2].imshow(to_img(xfD[b])); axes[b,2].set_title(f"PSNR={psnrs[b]:.2f} LPIPS={lpipss[b]:.4f}", fontsize=9); axes[b,2].axis("off")
        plt.suptitle(f"{name}: PSNR={psnr:.2f} SSIM={ssim:.4f} LPIPS={lpips:.4f} |dHF|={dhf:.3f}", fontsize=10)
        plt.tight_layout(rect=[0,0,1,0.93])
        plt.savefig(f"{HERE}/{name}.png", dpi=140); plt.close()

    with open(f"{HERE}/validation.json", "w") as f:
        json.dump(rows, f, indent=2, default=str)

    # Summary table
    print("\n" + "="*88)
    print(f"{'config':<20s}  {'PSNR':>6s}  {'SSIM':>7s}  {'LPIPS':>7s}  {'PSNR_u':>7s}  {'|dHF|':>6s}")
    print("="*88)
    for r in rows:
        print(f"{r['name']:<20s}  {r['psnr']:6.2f}  {r['ssim']:7.4f}  {r['lpips']:7.4f}  {r['psnr_unobs']:7.2f}  {r['dhf']:6.3f}")
    print("="*88)


if __name__ == "__main__":
    main()
