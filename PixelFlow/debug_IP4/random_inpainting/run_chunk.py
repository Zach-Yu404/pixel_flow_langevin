"""Per-GPU runner for random_inpainting OAT sweep.

Loads pre-built data.pt (8 GTs + 8 per-image masks at 70% prob) and runs each
config in [start:end] window. Each config = one B=8 forward pass through run_ip4
with per-image masks (operator.base_mask forced) and per-image labels.
"""
import os, sys, json, time, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
DBG  = os.path.dirname(HERE)
REPO = os.path.dirname(DBG)
for p in (HERE, DBG, REPO): sys.path.insert(0, p)
os.chdir(REPO)

import torch, numpy as np
from omegaconf import OmegaConf
import piq

from inpaintingStart import get_operator
from pixelflow.utils import config as config_utils
from ms_sampler_v5 import run_ip4, hf_energy
from oat_configs import CONFIGS


SIGMA_N = 0.05
RUN_DIR = os.path.join(HERE, "runs")
os.makedirs(RUN_DIR, exist_ok=True)


def to_img01(t):
    return ((t + 1) / 2).clamp(0, 1)


def force_mask(operator, mask_batch):
    operator.base_mask = mask_batch
    operator.mask = mask_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end",   type=int, required=True)
    ap.add_argument("--gpu",   type=int, default=0)
    args = ap.parse_args()

    DEVICE = f"cuda:{args.gpu}"
    chunk = CONFIGS[args.start:args.end]
    print(f"[gpu={args.gpu}] CONFIGS[{args.start}:{args.end}] = {len(chunk)} configs", flush=True)

    # --- load pre-built data
    d = torch.load(os.path.join(HERE, "data.pt"), map_location="cpu", weights_only=False)
    fnames, labels = d["fnames"], list(map(int, d["labels"]))
    gts   = d["gts"].to(DEVICE)        # [8, 3, 256, 256]
    masks = d["masks"].to(DEVICE)      # [8, 1, 256, 256]
    B = gts.shape[0]
    obs = float(masks.mean().item())
    print(f"  B={B}  observed={obs:.3f}  labels={labels}", flush=True)

    # --- load model
    config = OmegaConf.load("pretrained_models/c2img/config.yaml")
    model = config_utils.instantiate_from_config(config.model).to(DEVICE)
    ckpt = torch.load("pretrained_models/c2img/model.pt", map_location="cpu",
                      weights_only=False)
    model.load_state_dict(ckpt, strict=True); model.eval()

    lpips_fn = piq.LPIPS(replace_pooling=True).to(DEVICE)

    # --- HF energy of GT (under hole mask)
    hole = (1 - masks)
    gt_hf = hf_energy(gts * hole)
    print(f"  GT HF={gt_hf:.4f}", flush=True)

    for i, (name, kw) in enumerate(chunk):
        gi = args.start + i
        print(f"\n[{gi+1}/{len(CONFIGS)}] {name}", flush=True)
        t0 = time.time()

        # fresh operator with frozen per-image masks
        operator = get_operator(
            "inpainting", resolution=256, device=DEVICE, sigma=SIGMA_N,
            mask_type="random", mask_len_range=None, mask_prob_range=(0.7, 0.7 + 1e-6),
        )
        force_mask(operator, masks)

        y = operator(gts).detach()  # noiseless: y = mask * gt

        # deterministic seed across configs
        SEED = 42
        torch.manual_seed(SEED); np.random.seed(SEED); torch.cuda.manual_seed_all(SEED)

        try:
            xf, _, _, res, _, hf = run_ip4(
                model, config, gts, y, operator, SIGMA_N, DEVICE,
                class_label=labels, seed=SEED, **kw,
            )
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  ERROR: {e}", flush=True); continue
        xf = xf.to(DEVICE)
        elapsed = time.time() - t0

        # metrics
        with torch.no_grad():
            recon01 = to_img01(xf); gt01 = to_img01(gts)
            per = []
            for j in range(B):
                r1, g1 = recon01[j:j+1], gt01[j:j+1]
                p_ = piq.psnr(r1, g1, data_range=1.0).item()
                s_ = piq.ssim(r1, g1, data_range=1.0).item()
                l_ = lpips_fn(r1, g1).item()
                m  = (1 - masks[j:j+1])
                d2 = (r1 - g1).pow(2)
                n_unobs = m.sum().item() * r1.shape[1]
                mse_u = (m * d2).sum().item() / max(n_unobs, 1)
                psnr_u = -10 * float(np.log10(max(mse_u, 1e-12)))
                per.append(dict(idx=j, fname=fnames[j], label=labels[j],
                                psnr=p_, ssim=s_, lpips=l_, psnr_unobs=psnr_u))
        psnr  = float(np.mean([m["psnr"] for m in per]))
        ssim  = float(np.mean([m["ssim"] for m in per]))
        lp    = float(np.mean([m["lpips"] for m in per]))
        psnr_u= float(np.mean([m["psnr_unobs"] for m in per]))
        dhf   = float(abs(hf - gt_hf))
        print(f"  PSNR={psnr:.2f} SSIM={ssim:.4f} LPIPS={lp:.4f} PSNR_u={psnr_u:.2f} "
              f"|dHF|={dhf:.3f}  t={elapsed:.0f}s", flush=True)

        meta = dict(
            name=name, kw=kw,
            psnr_mean=psnr, ssim_mean=ssim, lpips_mean=lp, psnr_unobs_mean=psnr_u,
            hf=float(hf), dhf=dhf, gt_hf=float(gt_hf),
            res=float(res), time_s=elapsed, per_image=per,
        )
        out_dir = os.path.join(RUN_DIR, name)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)
        torch.save(dict(gts=gts.cpu(), masks=masks.cpu(), recons=xf.cpu()),
                   os.path.join(out_dir, "data.pt"))

    print(f"\n[gpu={args.gpu}] done.", flush=True)


if __name__ == "__main__":
    main()
