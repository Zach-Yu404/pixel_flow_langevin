"""Per-GPU runner for folder_woNoiseLast. Same protocol as sweep2/run_chunk.py."""
import os, sys, json, time, argparse

HERE = os.path.dirname(os.path.abspath(__file__))            # folder_woNoiseLast
LB   = os.path.dirname(HERE)                                 # larger_box
DBG  = os.path.dirname(LB)                                   # debug_IP4
REPO = os.path.dirname(DBG)                                  # PixelFlow root
sys.path.insert(0, HERE); sys.path.insert(0, LB)
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
from wnl_configs import CONFIGS


RUN_DIR = os.path.join(HERE, "runs")
os.makedirs(RUN_DIR, exist_ok=True)


def to_img(t):
    return (t.cpu().permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1).numpy()


def find_box_bbox(mask):
    inv = (1.0 - mask[0, 0].cpu()).numpy()
    rows = np.any(inv > 0.5, axis=1); cols = np.any(inv > 0.5, axis=0)
    if not rows.any(): return 0, 255, 0, 255
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    pad = 5
    return max(0, rmin-pad), min(255, rmax+pad), max(0, cmin-pad), min(255, cmax+pad)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--end",   type=int, required=True)
    p.add_argument("--gpu",   type=int, default=0)
    args = p.parse_args()

    DEVICE = f"cuda:{args.gpu}"
    chunk = CONFIGS[args.start:args.end]
    print(f"[gpu={args.gpu}] CONFIGS[{args.start}:{args.end}] = {len(chunk)} configs", flush=True)

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
    rmin, rmax, cmin, cmax = find_box_bbox(mask)
    gt_hf = hf_energy(gt * (1 - mask[0:1]))
    print(f"  GT HF={gt_hf:.4f}  bbox=[{rmin}:{rmax},{cmin}:{cmax}]", flush=True)

    lpips_fn = piq.LPIPS(replace_pooling=True).to(DEVICE)

    for i, (name, kw) in enumerate(chunk):
        gi = args.start + i
        print(f"\n[{gi+1}/{len(CONFIGS)}] {name}  kw={kw}", flush=True)
        t0 = time.time()
        try:
            xf, _, _, res, _, hf = run_ip4(
                model, config, gt, y, operator, sigma_n, DEVICE,
                class_label=10, seed=20000120, **kw,
            )
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  ERROR: {e}", flush=True); continue
        xfD = xf.to(DEVICE); elapsed = time.time() - t0

        recon_01 = ((xfD + 1) / 2).clamp(0, 1)
        gt_01    = ((gt   + 1) / 2).clamp(0, 1)
        per_img = []
        with torch.no_grad():
            for b in range(B):
                r1, g1 = recon_01[b:b+1], gt_01[b:b+1]
                p_ = piq.psnr(r1, g1, data_range=1.0).item()
                s_ = piq.ssim(r1, g1, data_range=1.0).item()
                l_ = lpips_fn(r1, g1).item()
                m  = (1 - mask[b:b+1])
                d2 = (r1 - g1).pow(2)
                n_unobs = m.sum().item() * r1.shape[1]
                mse_u = (m * d2).sum().item() / max(n_unobs, 1)
                psnr_u = -10 * float(np.log10(max(mse_u, 1e-12)))
                per_img.append(dict(idx=b, psnr=p_, ssim=s_, lpips=l_, psnr_unobs=psnr_u))

        avg = lambda k: float(np.mean([m[k] for m in per_img]))
        psnr, ssim, lpips, psnr_u = avg("psnr"), avg("ssim"), avg("lpips"), avg("psnr_unobs")
        dhf = float(abs(hf - gt_hf))
        print(f"  PSNR={psnr:.2f} SSIM={ssim:.4f} LPIPS={lpips:.4f} PSNR_u={psnr_u:.2f} |dHF|={dhf:.3f}  t={elapsed:.0f}s", flush=True)

        meta = dict(name=name, kw=kw, psnr=psnr, ssim=ssim, lpips=lpips,
                    psnr_unobs=psnr_u, hf=float(hf), dhf=dhf, gt_hf=float(gt_hf),
                    res=float(res), time=elapsed, per_image=per_img)
        with open(f"{RUN_DIR}/{name}.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)

        fig, axes = plt.subplots(B, 4, figsize=(16, 4*B))
        if B == 1: axes = axes[None, :]
        for b in range(B):
            gi_img = to_img(gt[b])
            yi_img = to_img((mask[b]*gt[b] + (1-mask[b])*(-torch.ones_like(gt[b]))).cpu())
            xfi = to_img(xfD[b])
            axes[b,0].imshow(gi_img); axes[b,0].set_title("GT", fontsize=9); axes[b,0].axis("off")
            axes[b,1].imshow(yi_img); axes[b,1].set_title("Measurement", fontsize=9); axes[b,1].axis("off")
            axes[b,2].imshow(xfi);    axes[b,2].set_title(f"recon  PSNR={per_img[b]['psnr']:.2f} LPIPS={per_img[b]['lpips']:.4f}", fontsize=9); axes[b,2].axis("off")
            crop_gt = gi_img[rmin:rmax+1, cmin:cmax+1]; crop_xf = xfi[rmin:rmax+1, cmin:cmax+1]
            sep = np.ones((crop_gt.shape[0], 4, 3))
            axes[b,3].imshow(np.concatenate([crop_gt, sep, crop_xf], axis=1))
            axes[b,3].set_title("GT box | recon box", fontsize=9); axes[b,3].axis("off")
        plt.suptitle(f"{name}\nPSNR={psnr:.2f} SSIM={ssim:.4f} LPIPS={lpips:.4f} |dHF|={dhf:.3f}", fontsize=10)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.savefig(f"{RUN_DIR}/{name}.png", dpi=140); plt.close()

    print(f"\n[gpu={args.gpu}] done.", flush=True)


if __name__ == "__main__":
    main()
