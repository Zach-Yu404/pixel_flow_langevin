"""Per-GPU chunk runner for the superresolution-4 sweep.

Each config:
  1. Loads 8 GT images from ImageNet (class 10, fixed seed)
  2. Builds bicubic 4x downsampling operator
  3. Patches ms_sampler_v5 to use SR-aware A_k / A_k^T and terminal replacement
  4. Calls run_ip4 with the kw
  5. Computes per-image PSNR / SSIM / LPIPS over the full 256×256 reconstruction
  6. Saves results/<name>.{json,png,pt}

Usage:
  python run_chunk.py --start 0 --end 7 --gpu 0
"""
import os, sys, json, time, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "debug_IP4"))
os.chdir(REPO)

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.datasets import ImageFolder

import piq
from inpaintingStart import get_operator
from pixelflow.utils import config as config_utils
from ms_posterior_sampling_utils import center_crop_arr
import ms_sampler_v5
from ms_sampler_v5 import run_ip4, hf_energy
import langevin_v5
from sweep_configs import CONFIGS

RESULT_DIR = os.path.join(HERE, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

DATA_DIR = "/data/Zach_dataset/imageNet256/ILSVRC/Data/CLS-LOC/train/"
RESOLUTION = 256
SCALE_FACTOR = 4
LR_SIZE = RESOLUTION // SCALE_FACTOR    # 64
SIGMA_N = 0.05
CLASS_LABEL = 10
NUM_IMAGES = 8
GT_SEED = 7919   # for selecting which 8 images of class 10 we use


# ============================================================================
# SR-aware A_k / A_k^T  (replaces inpainting-style make_Ak_fns)
# ============================================================================

def make_Ak_fns_sr(operator, y, stage_shape, device):
    """A_k: x (stage_h × stage_w) → upsample to 256 → bicubic downsample to LR (64×64)
       A_k^T: r (LR) → adjoint of bicubic → adjoint of bilinear stage upsample
    """
    stage_h, stage_w = stage_shape[-2:]
    full_h = full_w = operator.resolution
    target_h = target_w = operator.target_size
    need_resize = (stage_h != full_h) or (stage_w != full_w)

    def A_k(x):
        x_full = (F.interpolate(x, size=(full_h, full_w), mode="bilinear", align_corners=False)
                  if need_resize else x)
        return F.interpolate(x_full, size=(target_h, target_w),
                             mode="bicubic", align_corners=False)

    def AT_k(r):
        # adjoint of bicubic downsample (LR → full). Bicubic backward isn't
        # deterministic on CUDA, so wrap in warn_only context.
        prev = torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            with torch.enable_grad():
                xp = torch.zeros((r.shape[0], r.shape[1], full_h, full_w),
                                  device=r.device, dtype=r.dtype, requires_grad=True)
                ds = F.interpolate(xp, size=(target_h, target_w),
                                    mode="bicubic", align_corners=False)
                (grad,) = torch.autograd.grad(ds, xp, grad_outputs=r.detach())
            if not need_resize:
                return grad.detach()
            with torch.enable_grad():
                xs = torch.zeros((r.shape[0], r.shape[1], stage_h, stage_w),
                                  device=r.device, dtype=r.dtype, requires_grad=True)
                up = F.interpolate(xs, size=(full_h, full_w),
                                    mode="bilinear", align_corners=False)
                (grad2,) = torch.autograd.grad(up, xs, grad_outputs=grad.detach())
            return grad2.detach()
        finally:
            torch.use_deterministic_algorithms(prev)

    return A_k, AT_k


def terminal_replacement_sr(x1_k, y, mask_full_unused, weight=1.0):
    """SR analog of inpainting terminal replacement:
       x1_final = (1 - w) * x1 + w * bicubic_upsample(y)
    For SR, "observed" = LR pixels; the natural HR analog is the bicubic upsample.
    """
    h_target = x1_k.shape[-1]
    y_up = F.interpolate(y, size=(h_target, h_target),
                          mode="bicubic", align_corners=False)
    return (1.0 - weight) * x1_k + weight * y_up


# Patch into both modules so run_ip4 (which imports make_Ak_fns into its namespace)
# and the inner Langevin (which uses langevin_v5 helpers) both see the SR versions.
ms_sampler_v5.make_Ak_fns = make_Ak_fns_sr
ms_sampler_v5.terminal_replacement = terminal_replacement_sr
langevin_v5.terminal_replacement = terminal_replacement_sr


# ============================================================================
# Operator wrapper (adds get_mask=all-ones so make_Ak_fns mask check passes)
# ============================================================================

class SROperator:
    def __init__(self, resolution=256, scale_factor=4, device="cuda", sigma=0.05):
        self.resolution = resolution
        self.scale_factor = scale_factor
        self.target_size = resolution // scale_factor
        self.sigma = sigma
        self.device = device

    def __call__(self, x):
        return F.interpolate(x, size=(self.target_size, self.target_size),
                              mode="bicubic", align_corners=False)

    def get_mask(self, x=None):
        # All-ones mask at full resolution. For SR there is no inpainting mask;
        # this signals "every pixel is in the support" so the inpainting machinery
        # downstream of this object doesn't break. Our SR-aware A_k bypasses it.
        if x is None:
            shape = (1, 1, self.resolution, self.resolution)
            device = self.device
        else:
            shape = (x.shape[0], 1, self.resolution, self.resolution)
            device = x.device
        return torch.ones(shape, device=device)


# ============================================================================
# Data: 8 ImageNet class-10 images, deterministic
# ============================================================================

def load_gt_8(device):
    transform = transforms.Compose([
        transforms.Lambda(lambda im: center_crop_arr(im, RESOLUTION)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3, inplace=True),
    ])
    dataset = ImageFolder(DATA_DIR, transform=transform)
    class_indices = [i for i, (_, y) in enumerate(dataset.samples) if int(y) == CLASS_LABEL]
    if len(class_indices) < NUM_IMAGES:
        raise RuntimeError(f"class {CLASS_LABEL} has only {len(class_indices)} samples")
    g = torch.Generator().manual_seed(GT_SEED)
    perm = torch.randperm(len(class_indices), generator=g)[:NUM_IMAGES].tolist()
    sel = [class_indices[i] for i in perm]
    gt = torch.stack([dataset[i][0] for i in sel], dim=0).to(device)
    return gt, sel


def to_img(t):
    return (t.cpu().permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1).numpy()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end",   type=int, required=True)
    parser.add_argument("--gpu",   type=int, default=0)
    args = parser.parse_args()

    DEVICE = f"cuda:{args.gpu}"
    chunk = CONFIGS[args.start:args.end]
    print(f"[chunk gpu={args.gpu}] running configs[{args.start}:{args.end}] -> {len(chunk)} configs",
          flush=True)

    # Load model + GT once
    config = OmegaConf.load("pretrained_models/c2img/config.yaml")
    model = config_utils.instantiate_from_config(config.model).to(DEVICE)
    ckpt = torch.load("pretrained_models/c2img/model.pt",
                      map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt, strict=True); model.eval()

    gt, sel = load_gt_8(DEVICE)
    print(f"[chunk gpu={args.gpu}] loaded {gt.shape[0]} GT images (idx={sel})", flush=True)

    operator = SROperator(resolution=RESOLUTION, scale_factor=SCALE_FACTOR,
                           device=DEVICE, sigma=SIGMA_N)
    y_clean = operator(gt).detach()
    y = y_clean + SIGMA_N * torch.randn_like(y_clean)
    print(f"[chunk gpu={args.gpu}] LR shape = {tuple(y.shape)}  (sigma_n={SIGMA_N})", flush=True)

    lpips_fn = piq.LPIPS(replace_pooling=True).to(DEVICE)

    for i, (name, kw) in enumerate(chunk):
        gi = args.start + i
        print(f"\n[{gi+1}/{len(CONFIGS)}] {name}  he={kw['h_epsilon']:.0e} L={kw['num_langevin']} "
              f"hx={kw['h_x']:.2f} srs={kw['sigma_ref_sq']:.0e} tr={kw['terminal_replace_weight']:.0f}",
              flush=True)
        t0 = time.time()
        try:
            xf, psnr_obs, psnr_all, res, t_run, hf = run_ip4(
                model, config, gt, y, operator, SIGMA_N, DEVICE,
                class_label=CLASS_LABEL, seed=20000120, **kw,
            )
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            import traceback; traceback.print_exc()
            continue
        xfD = xf.to(DEVICE)
        elapsed = time.time() - t0

        recon_01 = ((xfD + 1) / 2).clamp(0, 1)
        gt_01    = ((gt   + 1) / 2).clamp(0, 1)
        B = gt.shape[0]

        per_img = []
        with torch.no_grad():
            for b in range(B):
                r1, g1 = recon_01[b:b+1], gt_01[b:b+1]
                p = piq.psnr(r1, g1, data_range=1.0).item()
                s = piq.ssim(r1, g1, data_range=1.0).item()
                l = lpips_fn(r1, g1).item()
                per_img.append(dict(idx=b, psnr=p, ssim=s, lpips=l))

        avg_lpips = float(np.mean([m["lpips"] for m in per_img]))
        avg_psnr  = float(np.mean([m["psnr"]  for m in per_img]))
        avg_ssim  = float(np.mean([m["ssim"]  for m in per_img]))

        # Bicubic upsample baseline for reference (no model)
        y_up = F.interpolate(y, size=(RESOLUTION, RESOLUTION),
                              mode="bicubic", align_corners=False)
        y_up01 = ((y_up + 1) / 2).clamp(0, 1)
        baseline_psnrs = [piq.psnr(y_up01[b:b+1], gt_01[b:b+1], data_range=1.0).item() for b in range(B)]
        baseline_lpips = [lpips_fn(y_up01[b:b+1], gt_01[b:b+1]).item() for b in range(B)]
        baseline_psnr = float(np.mean(baseline_psnrs))
        baseline_lpips_mean = float(np.mean(baseline_lpips))

        print(f"  PSNR={avg_psnr:.2f} SSIM={avg_ssim:.4f} LPIPS={avg_lpips:.4f}  "
              f"(bicubic baseline P={baseline_psnr:.2f} L={baseline_lpips_mean:.4f})  "
              f"t={elapsed:.0f}s", flush=True)

        meta = dict(name=name, kw=kw,
                    psnr=avg_psnr, ssim=avg_ssim, lpips=avg_lpips,
                    res=float(res), time=elapsed,
                    baseline_bicubic_psnr=baseline_psnr,
                    baseline_bicubic_lpips=baseline_lpips_mean,
                    per_image=per_img,
                    selected_indices=sel)
        with open(f"{RESULT_DIR}/{name}.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)

        # Per-config PNG: 4 columns (GT | LR-upsampled | recon | diff) × first 4 images
        n_show = min(B, 4)
        fig, axes = plt.subplots(n_show, 4, figsize=(13, 3.0 * n_show))
        if n_show == 1: axes = axes[None, :]
        y_up_img = ((y_up + 1) / 2).clamp(0, 1).cpu()
        for b in range(n_show):
            axes[b, 0].imshow(to_img(gt[b]));     axes[b, 0].set_title(f"GT (idx={sel[b]})", fontsize=8); axes[b, 0].axis("off")
            axes[b, 1].imshow(y_up_img[b].permute(1, 2, 0).numpy()); axes[b, 1].set_title(f"bicubic ↑  P={baseline_psnrs[b]:.2f}  L={baseline_lpips[b]:.3f}", fontsize=8); axes[b, 1].axis("off")
            axes[b, 2].imshow(to_img(xfD[b]));    axes[b, 2].set_title(f"recon  P={per_img[b]['psnr']:.2f}  L={per_img[b]['lpips']:.3f}", fontsize=8); axes[b, 2].axis("off")
            diff = (recon_01[b] - gt_01[b]).abs().mean(0).cpu().numpy()
            axes[b, 3].imshow(diff, cmap="hot", vmin=0, vmax=0.5); axes[b, 3].set_title("|recon−GT|", fontsize=8); axes[b, 3].axis("off")
        plt.suptitle(f"{name}   he={kw['h_epsilon']}  L={kw['num_langevin']}  hx={kw['h_x']}  "
                     f"srs={kw['sigma_ref_sq']}  tr={kw['terminal_replace_weight']}  "
                     f"fd={int(kw['final_denoise'])}  ns={kw['noise_scale']}\n"
                     f"PSNR={avg_psnr:.2f}  SSIM={avg_ssim:.4f}  LPIPS={avg_lpips:.4f}   "
                     f"(bicubic baseline P={baseline_psnr:.2f} L={baseline_lpips_mean:.4f})",
                     fontsize=10)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.savefig(f"{RESULT_DIR}/{name}.png", dpi=130)
        plt.close()

    print(f"\n[chunk gpu={args.gpu}] done.", flush=True)


if __name__ == "__main__":
    main()
