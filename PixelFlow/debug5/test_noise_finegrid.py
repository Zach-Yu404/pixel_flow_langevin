#!/usr/bin/env python
"""Fine-grid noise injection sweep around alpha=0.1 sweet spot. GPU:4."""
import sys, os, math, copy, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "debug_IP4"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import torch, torch.nn.functional as F, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from test_noise_inject import run_noise_inject, load_gt, to_img, find_box, hf_energy
from inpaintingStart import get_operator
from pixelflow.utils import config as config_utils

DEVICE = "cuda:4"
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_finegrid")
os.makedirs(RESULT_DIR, exist_ok=True)

CONFIGS = [
    ("a003",  dict(inject_mode="constant", inject_alpha=0.03, use_langevin_s3=False)),
    ("a005",  dict(inject_mode="constant", inject_alpha=0.05, use_langevin_s3=False)),
    ("a007",  dict(inject_mode="constant", inject_alpha=0.07, use_langevin_s3=False)),
    ("a008",  dict(inject_mode="constant", inject_alpha=0.08, use_langevin_s3=False)),
    ("a009",  dict(inject_mode="constant", inject_alpha=0.09, use_langevin_s3=False)),
    ("a010",  dict(inject_mode="constant", inject_alpha=0.10, use_langevin_s3=False)),
    ("a011",  dict(inject_mode="constant", inject_alpha=0.11, use_langevin_s3=False)),
    ("a012",  dict(inject_mode="constant", inject_alpha=0.12, use_langevin_s3=False)),
    ("a013",  dict(inject_mode="constant", inject_alpha=0.13, use_langevin_s3=False)),
    ("a015",  dict(inject_mode="constant", inject_alpha=0.15, use_langevin_s3=False)),
    ("a018",  dict(inject_mode="constant", inject_alpha=0.18, use_langevin_s3=False)),
    # With L=3 Langevin (minimal smoothing) at best alphas
    ("a008_L3", dict(inject_mode="constant", inject_alpha=0.08, use_langevin_s3=True, num_langevin=3)),
    ("a010_L3", dict(inject_mode="constant", inject_alpha=0.10, use_langevin_s3=True, num_langevin=3)),
    ("a012_L3", dict(inject_mode="constant", inject_alpha=0.12, use_langevin_s3=True, num_langevin=3)),
    # Multi-stage injection at best alpha
    ("a010_s23", dict(inject_mode="constant", inject_alpha=0.10, inject_stages=(2,3), use_langevin_s3=False)),
    ("a010_all", dict(inject_mode="constant", inject_alpha=0.10, inject_stages=(0,1,2,3), use_langevin_s3=False)),
]

def main():
    print(f"Fine-grid on {DEVICE}  ({len(CONFIGS)} configs)", flush=True)
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
            xf, pa, res, t, hf = run_noise_inject(model, config, gt, y, operator, sigma_n, DEVICE, **kw)
            dhf = abs(hf - gt_hf)
            results[name] = dict(x=xf, psnr=pa, hf=hf, dhf=dhf, res=res, time=t)
            print(f"PSNR={pa:.2f}  HF={hf:.3f}  |dHF|={dhf:.3f}  res={res:.0f}  t={t:.0f}s", flush=True)
        except Exception as e:
            print(f"ERROR: {e}", flush=True)

    # Grid visualization
    n = len(results)
    fig, axes = plt.subplots(3, n+1, figsize=((n+1)*2.2, 7))
    gt_img, gt_img2 = to_img(gt[0]), to_img(gt[1])
    axes[0,0].imshow(gt_img); axes[0,0].set_title(f"GT\nHF={gt_hf:.3f}", fontsize=5); axes[0,0].axis("off")
    axes[1,0].imshow(gt_img[rmin:rmax+1,cmin:cmax+1]); axes[1,0].set_title("GT box", fontsize=5); axes[1,0].axis("off")
    axes[2,0].imshow(gt_img2[rmin:rmax+1,cmin:cmax+1]); axes[2,0].set_title("GT box2", fontsize=5); axes[2,0].axis("off")

    for i, (name, v) in enumerate(results.items()):
        xf0, xf1 = to_img(v["x"][0]), to_img(v["x"][1])
        axes[0,i+1].imshow(xf0)
        axes[0,i+1].set_title(f"{name}\nP={v['psnr']:.1f}", fontsize=4); axes[0,i+1].axis("off")
        axes[1,i+1].imshow(xf0[rmin:rmax+1,cmin:cmax+1])
        axes[1,i+1].set_title(f"hf={v['hf']:.3f} |d|={v['dhf']:.3f}", fontsize=4); axes[1,i+1].axis("off")
        axes[2,i+1].imshow(xf1[rmin:rmax+1,cmin:cmax+1])
        axes[2,i+1].set_title(f"t={v['time']:.0f}s", fontsize=4); axes[2,i+1].axis("off")

    plt.suptitle(f"Fine-grid noise injection (GT HF={gt_hf:.3f})", fontsize=8)
    plt.tight_layout(rect=[0,0,1,0.93])
    plt.savefig(f"{RESULT_DIR}/finegrid.png", dpi=200)
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
        plt.suptitle(f"{name} — PSNR={v['psnr']:.2f} HF={v['hf']:.3f} |dHF|={v['dhf']:.3f}", fontsize=10)
        plt.tight_layout(rect=[0,0,1,0.95])
        plt.savefig(f"{RESULT_DIR}/{name}.png", dpi=150)
        plt.close()

    print(f"\nSaved {RESULT_DIR}/finegrid.png + {len(results)} per-config images")
    print("DONE.", flush=True)

if __name__ == "__main__":
    main()
