"""Run ONE (group_id, config_name) experiment.

Loads pre-built groups/group_{g}.pt (gts, masks, labels, fnames) and the
specified inference config, runs run_ip4 in batches of B=8 with per-image
masks (forced via operator.base_mask), saves recons + per-image PSNR/SSIM/LPIPS.

FID is computed later in aggregate.py to share Inception across all 15 runs.
"""
import os, sys, json, time, argparse

HERE = os.path.dirname(os.path.abspath(__file__))      # val/
LB   = os.path.dirname(HERE)                           # larger_box
DBG  = os.path.dirname(LB)                             # debug_IP4
REPO = os.path.dirname(DBG)                            # PixelFlow
FC_FV = os.path.join(LB, "final_configs", "full_validation")
for p in (HERE, LB, DBG, FC_FV, REPO): sys.path.insert(0, p)
os.chdir(REPO)

import torch, numpy as np
from omegaconf import OmegaConf
import piq

from inpaintingStart import get_operator
from pixelflow.utils import config as config_utils
from ms_sampler_v5 import run_ip4


# Three configs from the 36-cell sweep — all srs=1e-4, fd=1, tr=1
COMPLETE_DIR = os.path.join(FC_FV, "complete_configs")

CONFIG_MAP = {
    "pareto_dual_king_srs1e-4__fd1_tr1":     "pareto_dual_king_srs1e-4",
    "LPIPS_king_srs1e-4__fd1_tr1":           "LPIPS_king_srs1e-4",
    "balanced_perceptual_srs1e-4__fd1_tr1":  "balanced_perceptual_srs1e-4",
}

SIGMA_N = 0.05
B       = 8


def load_kw(config_name):
    base_tag = CONFIG_MAP[config_name]
    base = json.load(open(os.path.join(COMPLETE_DIR, f"{base_tag}.json")))
    kw = dict(base["kw"])
    # Lock the fd1, tr1 axes (final_denoise=True, terminal_replace_weight=1.0)
    kw["final_denoise"] = True
    kw["terminal_replace_weight"] = 1.0
    return kw


def force_mask(operator, mask_batch):
    """Force operator to use the supplied mask batch (per-image masks supported)."""
    operator.base_mask = mask_batch
    operator.mask = mask_batch


def to_img01(t):
    return ((t + 1) / 2).clamp(0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group",  type=int, required=True)
    ap.add_argument("--config", type=str, required=True, choices=list(CONFIG_MAP.keys()))
    ap.add_argument("--gpu",    type=int, default=0)
    ap.add_argument("--max_n",  type=int, default=0, help="if >0, only run first N images (smoke)")
    args = ap.parse_args()

    DEVICE = f"cuda:{args.gpu}"
    out_dir = os.path.join(HERE, "runs", f"group{args.group}_{args.config}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[start] group={args.group}  config={args.config}  gpu={args.gpu}", flush=True)
    print(f"        out={out_dir}", flush=True)

    # --- load group data
    gp = torch.load(os.path.join(HERE, "groups", f"group_{args.group}.pt"),
                    map_location="cpu", weights_only=False)
    fnames, labels  = gp["fnames"], gp["labels"]
    gts_full        = gp["gts"].to(DEVICE)        # [100, 3, 256, 256]
    masks_full      = gp["masks"].to(DEVICE)      # [100, 1, 256, 256]
    N = gts_full.shape[0]
    assert len(fnames) == N == len(labels)
    if args.max_n and args.max_n < N:
        N = args.max_n
        gts_full   = gts_full[:N]
        masks_full = masks_full[:N]
        fnames     = fnames[:N]
        labels     = labels[:N]
        print(f"  [SMOKE] truncated to N={N}", flush=True)
    print(f"  loaded {N} images, mask_seed={gp['mask_seed']}", flush=True)

    # --- load model + config
    config = OmegaConf.load("pretrained_models/c2img/config.yaml")
    model = config_utils.instantiate_from_config(config.model).to(DEVICE)
    ckpt = torch.load("pretrained_models/c2img/model.pt", map_location="cpu",
                      weights_only=False)
    model.load_state_dict(ckpt, strict=True); model.eval()
    print(f"  model loaded", flush=True)

    # --- inference kwargs
    kw = load_kw(args.config)
    print(f"  kw[final_denoise]={kw['final_denoise']} kw[terminal_replace_weight]={kw['terminal_replace_weight']}",
          flush=True)
    print(f"  kw[h_epsilon]={kw['h_epsilon']} kw[num_langevin]={kw['num_langevin']} kw[sigma_ref_sq]={kw['sigma_ref_sq']}",
          flush=True)

    lpips_fn = piq.LPIPS(replace_pooling=True).to(DEVICE)

    # --- per-image storage
    recons    = torch.empty_like(gts_full)
    per_image = []

    t_total0 = time.time()
    for bs in range(0, N, B):
        be = min(bs + B, N)
        b  = be - bs
        gt_b   = gts_full[bs:be]              # [b, 3, 256, 256]
        mask_b = masks_full[bs:be]            # [b, 1, 256, 256]
        labels_b = list(map(int, labels[bs:be]))

        # operator: fresh instance each batch, force the masks
        operator = get_operator(
            "inpainting", resolution=256, device=DEVICE, sigma=SIGMA_N,
            mask_type="box", mask_len_range=(128, 129), mask_prob_range=None,
        )
        force_mask(operator, mask_b)

        # noiseless measurement (matches sweep behavior: y = mask * gt, no extra noise)
        y_b = operator(gt_b).detach()

        # seed identical for all configs in same (group, batch_idx) so noise inside run_ip4 is shared
        per_batch_seed = 1_000_000 + args.group * 1000 + bs
        torch.manual_seed(per_batch_seed)
        np.random.seed(per_batch_seed)
        torch.cuda.manual_seed_all(per_batch_seed)

        t0 = time.time()
        try:
            xf, _, _, res, _, hf = run_ip4(
                model, config, gt_b, y_b, operator, SIGMA_N, DEVICE,
                class_label=labels_b, seed=per_batch_seed, **kw,
            )
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  [batch {bs}:{be}] ERROR: {e}", flush=True)
            return
        xf = xf.to(DEVICE)
        elapsed = time.time() - t0

        recons[bs:be] = xf

        with torch.no_grad():
            recon01 = to_img01(xf)
            gt01    = to_img01(gt_b)
            for j in range(b):
                r1, g1 = recon01[j:j+1], gt01[j:j+1]
                p_ = piq.psnr(r1, g1, data_range=1.0).item()
                s_ = piq.ssim(r1, g1, data_range=1.0).item()
                l_ = lpips_fn(r1, g1).item()
                m  = (1 - mask_b[j:j+1])
                d2 = (r1 - g1).pow(2)
                n_unobs = m.sum().item() * r1.shape[1]
                mse_u = (m * d2).sum().item() / max(n_unobs, 1)
                psnr_u = -10 * float(np.log10(max(mse_u, 1e-12)))
                per_image.append(dict(
                    idx=bs+j, fname=fnames[bs+j], label=labels_b[j],
                    psnr=p_, ssim=s_, lpips=l_, psnr_unobs=psnr_u,
                ))
        print(f"  [{bs:3d}:{be:3d}]  t={elapsed:.0f}s  "
              f"PSNR_avg={np.mean([m['psnr'] for m in per_image[-b:]]):.2f}  "
              f"LPIPS_avg={np.mean([m['lpips'] for m in per_image[-b:]]):.4f}", flush=True)

    t_total = time.time() - t_total0
    psnr  = float(np.mean([m["psnr"]  for m in per_image]))
    ssim  = float(np.mean([m["ssim"]  for m in per_image]))
    lpips = float(np.mean([m["lpips"] for m in per_image]))
    psnr_u= float(np.mean([m["psnr_unobs"] for m in per_image]))
    print(f"\n[done]  PSNR={psnr:.3f}  SSIM={ssim:.4f}  LPIPS={lpips:.4f}  "
          f"PSNR_u={psnr_u:.3f}  t_total={t_total:.0f}s", flush=True)

    # save
    torch.save(dict(
        gts=gts_full.cpu(), masks=masks_full.cpu(), recons=recons.cpu(),
        fnames=fnames, labels=labels,
    ), os.path.join(out_dir, "data.pt"))

    meta = dict(
        group_id=args.group, config_name=args.config, num_images=N,
        kw=kw, kw_complete_source=CONFIG_MAP[args.config] + ".json",
        psnr_mean=psnr, ssim_mean=ssim, lpips_mean=lpips, psnr_unobs_mean=psnr_u,
        time_total_s=t_total, mask_seed=int(gp["mask_seed"]),
        per_image=per_image,
    )
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"  saved {out_dir}/{{data.pt, meta.json}}", flush=True)


if __name__ == "__main__":
    main()
