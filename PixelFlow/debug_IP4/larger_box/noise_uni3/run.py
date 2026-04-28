"""Run G_hx_uni3 setting with noise_scale=1.0 (full ULA).

Base: G_hx_uni3 from exploration/scripts/sweep_configs_e.py
  h_x=0.3, h_epsilon=1e-3, num_langevin=5, lambda_reg=50.0,
  noise_scale=0.0, terminal_replace_weight=1.0
Override: noise_scale = 1.0

Same 2-image GT, 128x128 box, seed=7919 — directly comparable to
exploration/runs/G_hx_uni3.* (LPIPS=0.1328, PSNR=16.83, |dHF|=0.007).
"""
import os, sys, json, time, argparse

HERE = os.path.dirname(os.path.abspath(__file__))           # noise_uni3
LB   = os.path.dirname(HERE)                                # larger_box
DBG  = os.path.dirname(LB)                                  # debug_IP4
REPO = os.path.dirname(DBG)                                 # PixelFlow root
sys.path.insert(0, LB)
sys.path.insert(0, DBG)
sys.path.insert(0, REPO)
os.chdir(REPO)

import torch
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

import piq
from inpaintingStart import get_operator
from pixelflow.utils import config as config_utils
from ms_sampler_v5 import run_ip4, hf_energy


NAME = "noise_uni3"
KW = dict(
    h_x=0.3,
    h_epsilon=1e-3,
    num_langevin=5,
    lambda_reg=50.0,
    noise_scale=1.0,                # <-- the swept axis
    terminal_replace_weight=1.0,
)


def to_img(t):
    return (t.cpu().permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1).numpy()


def find_box_bbox(mask):
    inv = (1.0 - mask[0, 0].cpu()).numpy()
    rows = np.any(inv > 0.5, axis=1)
    cols = np.any(inv > 0.5, axis=0)
    if not rows.any():
        return 0, 255, 0, 255
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    pad = 5
    return max(0, rmin-pad), min(255, rmax+pad), max(0, cmin-pad), min(255, cmax+pad)


def load_gt(device):
    pt = "trajectory_videos/posterior_sampling/baseline_05/baseline_05_mode-f_x1.pt"
    d = torch.load(pt, map_location="cpu", weights_only=False)
    return d["gt"][:2].to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    DEVICE = f"cuda:{args.gpu}"
    print(f"[noise_uni3] gpu={args.gpu}  KW={KW}", flush=True)

    config = OmegaConf.load("pretrained_models/c2img/config.yaml")
    model = config_utils.instantiate_from_config(config.model).to(DEVICE)
    ckpt = torch.load("pretrained_models/c2img/model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt, strict=True); model.eval()

    gt = load_gt(DEVICE)
    B = gt.shape[0]
    sigma_n = 0.05

    torch.manual_seed(7919)  # SAME seed as larger_box / exploration sweeps
    operator = get_operator(
        "inpainting", resolution=256, device=DEVICE, sigma=sigma_n,
        mask_type="box", mask_len_range=(128, 129), mask_prob_range=None,
    )
    y = operator(gt).detach()
    mask = operator.get_mask(x=gt).float().to(DEVICE)
    if mask.shape[0] == 1 and B > 1:
        mask = mask.expand(B, -1, -1, -1)

    rmin, rmax, cmin, cmax = find_box_bbox(mask)
    gt_hf = hf_energy(gt * (1 - mask[0:1]))
    print(f"  box bbox = [{rmin}:{rmax}, {cmin}:{cmax}]  GT HF={gt_hf:.4f}", flush=True)

    lpips_fn = piq.LPIPS(replace_pooling=True).to(DEVICE)

    t0 = time.time()
    xf, _, _, res, _, hf = run_ip4(
        model, config, gt, y, operator, sigma_n, DEVICE,
        class_label=10, seed=20000120, **KW,
    )
    xfD = xf.to(DEVICE)
    elapsed = time.time() - t0

    recon_01 = ((xfD + 1) / 2).clamp(0, 1)
    gt_01    = ((gt   + 1) / 2).clamp(0, 1)

    per_img = []
    with torch.no_grad():
        for b in range(B):
            r1, g1 = recon_01[b:b+1], gt_01[b:b+1]
            p = piq.psnr(r1, g1, data_range=1.0).item()
            s = piq.ssim(r1, g1, data_range=1.0).item()
            l = lpips_fn(r1, g1).item()
            m = (1 - mask[b:b+1])
            diff2 = (r1 - g1).pow(2)
            n_unobs = m.sum().item() * r1.shape[1]
            mse_u = (m * diff2).sum().item() / max(n_unobs, 1)
            psnr_u = -10 * float(np.log10(max(mse_u, 1e-12)))
            per_img.append(dict(idx=b, psnr=p, ssim=s, lpips=l, psnr_unobs=psnr_u))

    avg_lpips = float(np.mean([m["lpips"] for m in per_img]))
    avg_psnr  = float(np.mean([m["psnr"]  for m in per_img]))
    avg_ssim  = float(np.mean([m["ssim"]  for m in per_img]))
    avg_psnru = float(np.mean([m["psnr_unobs"] for m in per_img]))
    dhf = float(abs(hf - gt_hf))
    print(f"  PSNR={avg_psnr:.2f} SSIM={avg_ssim:.4f} LPIPS={avg_lpips:.4f} "
          f"PSNR_u={avg_psnru:.2f} HF={hf:.3f} |dHF|={dhf:.3f}  t={elapsed:.0f}s", flush=True)

    meta = dict(name=NAME, kw=KW,
                psnr=avg_psnr, ssim=avg_ssim, lpips=avg_lpips,
                psnr_unobs=avg_psnru, hf=float(hf), dhf=dhf, gt_hf=float(gt_hf),
                res=float(res), time=elapsed,
                per_image=per_img,
                baseline=dict(name="G_hx_uni3", source="exploration/runs/G_hx_uni3.json"))
    with open(f"{HERE}/{NAME}.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    torch.save({"xf": xfD.cpu(), "name": NAME, "kw": KW, "metrics": meta},
               f"{HERE}/{NAME}.pt")

    fig, axes = plt.subplots(B, 4, figsize=(16, 4*B))
    if B == 1:
        axes = axes[None, :]
    for b in range(B):
        gi_img = to_img(gt[b])
        yi_img = to_img((mask[b]*gt[b] + (1-mask[b])*(-torch.ones_like(gt[b]))).cpu())
        xfi = to_img(xfD[b])
        axes[b,0].imshow(gi_img); axes[b,0].set_title("GT", fontsize=9); axes[b,0].axis("off")
        axes[b,1].imshow(yi_img); axes[b,1].set_title("Measurement", fontsize=9); axes[b,1].axis("off")
        axes[b,2].imshow(xfi);    axes[b,2].set_title(f"recon  PSNR={per_img[b]['psnr']:.2f} LPIPS={per_img[b]['lpips']:.4f}",
                                                       fontsize=9); axes[b,2].axis("off")
        crop_gt = gi_img[rmin:rmax+1, cmin:cmax+1]
        crop_xf = xfi   [rmin:rmax+1, cmin:cmax+1]
        sep = np.ones((crop_gt.shape[0], 4, 3))
        axes[b,3].imshow(np.concatenate([crop_gt, sep, crop_xf], axis=1))
        axes[b,3].set_title("GT box | recon box", fontsize=9); axes[b,3].axis("off")
    plt.suptitle(f"{NAME}  (G_hx_uni3 + noise_scale=1.0)\n"
                 f"PSNR={avg_psnr:.2f} SSIM={avg_ssim:.4f} LPIPS={avg_lpips:.4f} "
                 f"PSNR_u={avg_psnru:.2f} HF={hf:.3f} |dHF|={dhf:.3f}",
                 fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(f"{HERE}/{NAME}.png", dpi=140)
    plt.close()
    print("[noise_uni3] done.", flush=True)


if __name__ == "__main__":
    main()
