"""Best-of-box per-config validation eval.

Setup
-----
- 4 TOP-Tier visual configs from final_ana/VISUAL_BEST_BOX.md:
    A0_reference_WINNER3, E1_W1+lam20, C1_W1+hx_multistage, H1_W1+X4+noise0.3
- Box inpainting: 128x128 mask, RANDOM position (mask_len_range=(128,129))
- 5 groups x 100 images each, sampled WITH replacement from ImageNet val
- Per-image PSNR / SSIM / LPIPS via piq; tensors saved for group-level FID
- Output: /data/Zach_dataset/imageNet_results/best_box/<config>/group_<g>/

Usage:
    python run_eval.py --config A0_reference_WINNER3 --group 0 --gpu 0
"""
import os, sys, json, time, argparse, csv

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "debug_IP4"))
os.chdir(REPO)

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from omegaconf import OmegaConf

import piq
from inpaintingStart import get_operator
from pixelflow.utils import config as config_utils
from ms_sampler_v5 import run_ip4

VAL_DIR  = "/data/Zach_dataset/imageNet256/ILSVRC/Data/CLS-LOC/val"
VAL_LBL  = "/data/Zach_dataset/imageNet256/LOC_val_solution.csv"
SYNSET   = "/data/Zach_dataset/imageNet256/LOC_synset_mapping.txt"
OUT_BASE = os.path.join(HERE, "output")
CFG_DIR  = os.path.join(HERE, "configs")

GROUP_SEEDS = [11, 22, 33, 44, 55]
N_PER_GROUP = 100
BATCH_SIZE  = 4

SIGMA_N      = 0.05
BOX_SIZE     = 128
MASK_SEED_MULT = 7919  # prime, distinct from sample seed


def load_val_label_map():
    synset_to_idx = {}
    with open(SYNSET) as f:
        for i, line in enumerate(f):
            synset_to_idx[line.split()[0]] = i
    fname_to_idx = {}
    with open(VAL_LBL) as f:
        rdr = csv.reader(f); next(rdr)
        for row in rdr:
            fname_to_idx[row[0]] = synset_to_idx[row[1].split()[0]]
    return fname_to_idx


def select_group(group_idx, all_files, label_map):
    """100 images sampled WITH replacement from val set. Deterministic per group."""
    rng = np.random.RandomState(GROUP_SEEDS[group_idx])
    idx = rng.choice(len(all_files), size=N_PER_GROUP, replace=True)
    selected = [(all_files[i], label_map[all_files[i].replace(".JPEG", "")]) for i in idx]
    return selected


def center_crop_arr(pil_image, size):
    while min(*pil_image.size) >= 2 * size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), Image.BOX)
    scale = size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), Image.BICUBIC)
    arr = np.array(pil_image)
    cy = (arr.shape[0] - size) // 2
    cx = (arr.shape[1] - size) // 2
    return Image.fromarray(arr[cy:cy + size, cx:cx + size])


def make_box_operator(device):
    """Random-position 128x128 box. Position re-randomized via torch.manual_seed."""
    return get_operator(
        "inpainting", resolution=256, device=device, sigma=SIGMA_N,
        mask_type="box",
        mask_len_range=(BOX_SIZE, BOX_SIZE + 1),
        mask_prob_range=None,
    )


def _save_img(x01, path):
    arr = (x01.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Config name (must match a file in best_box/configs/<name>.json)")
    parser.add_argument("--group",  type=int, required=True, choices=range(5))
    parser.add_argument("--gpu",    type=int, default=0)
    parser.add_argument("--batch",  type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    DEVICE = f"cuda:{args.gpu}"

    cfg_path = os.path.join(CFG_DIR, f"{args.config}.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    KW = dict(cfg["kw"])
    # Runtime-controlled kwargs are set per-batch below; never let JSON override.
    for k in ("class_label", "seed"):
        KW.pop(k, None)
    print(f"[CFG {args.config}] {cfg.get('desc','')}", flush=True)
    print(f"  overrides = {json.dumps(cfg.get('overrides', {}))}", flush=True)

    out_dir = os.path.join(OUT_BASE, args.config, f"group_{args.group}")
    os.makedirs(f"{out_dir}/recons", exist_ok=True)
    os.makedirs(f"{out_dir}/gts",    exist_ok=True)
    os.makedirs(f"{out_dir}/ys",     exist_ok=True)

    label_map = load_val_label_map()
    all_files = sorted(f for f in os.listdir(VAL_DIR) if f.endswith(".JPEG"))
    selected = select_group(args.group, all_files, label_map)
    print(f"[GPU {args.gpu}] cfg={args.config} group={args.group} N={len(selected)} (with replacement)", flush=True)

    config = OmegaConf.load("pretrained_models/c2img/config.yaml")
    model = config_utils.instantiate_from_config(config.model).to(DEVICE)
    ckpt = torch.load("pretrained_models/c2img/model.pt",
                      map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt, strict=True); model.eval()

    lpips_fn = piq.LPIPS(replace_pooling=True).to(DEVICE)

    tf = transforms.Compose([
        transforms.Lambda(lambda p: center_crop_arr(p, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    metrics_per_image = []
    recon_01_list, gt_01_list = [], []

    for batch_start in range(0, len(selected), args.batch):
        batch_items = selected[batch_start:batch_start + args.batch]
        gts, lbls, fnames = [], [], []
        for fname, lbl in batch_items:
            img = Image.open(os.path.join(VAL_DIR, fname)).convert("RGB")
            gts.append(tf(img)); lbls.append(lbl); fnames.append(fname)
        gt = torch.stack(gts).to(DEVICE)
        B = gt.shape[0]

        # Random box position per batch (deterministic from group seed + batch index)
        torch.manual_seed(GROUP_SEEDS[args.group] * MASK_SEED_MULT + batch_start)
        operator = make_box_operator(DEVICE)
        y = operator(gt).detach()

        t0 = time.time()
        xf, _, _, res, t_run, _ = run_ip4(
            model, config, gt, y, operator, SIGMA_N, DEVICE,
            class_label=lbls,
            seed=GROUP_SEEDS[args.group] * 1000 + batch_start,
            **KW,
        )
        xf = xf.to(DEVICE)
        elapsed = time.time() - t0
        print(f"  batch {batch_start}-{batch_start+B-1} done in {elapsed:.0f}s  res={res:.1f}", flush=True)

        recon_01 = ((xf + 1) / 2).clamp(0, 1)
        gt_01    = ((gt + 1) / 2).clamp(0, 1)
        y_01     = ((y  + 1) / 2).clamp(0, 1)

        with torch.no_grad():
            for i in range(B):
                r1 = recon_01[i:i+1]; g1 = gt_01[i:i+1]
                psnr_v  = piq.psnr(r1, g1, data_range=1.0).item()
                ssim_v  = piq.ssim(r1, g1, data_range=1.0).item()
                lpips_v = lpips_fn(r1, g1).item()
                metrics_per_image.append(dict(
                    file=fnames[i], class_label=int(lbls[i]),
                    psnr=psnr_v, ssim=ssim_v, lpips=lpips_v,
                ))
                idx = batch_start + i
                _save_img(r1[0], f"{out_dir}/recons/{idx:04d}.png")
                _save_img(g1[0], f"{out_dir}/gts/{idx:04d}.png")
                _save_img(y_01[i], f"{out_dir}/ys/{idx:04d}.png")
                recon_01_list.append(r1.detach().cpu())
                gt_01_list.append(g1.detach().cpu())

    psnrs  = [m["psnr"]  for m in metrics_per_image]
    ssims  = [m["ssim"]  for m in metrics_per_image]
    lpipss = [m["lpips"] for m in metrics_per_image]
    summary = dict(
        config=args.config, kw=KW, task="box_random_pos_128",
        group=args.group, n=len(metrics_per_image),
        psnr_mean=float(np.mean(psnrs)),  psnr_std=float(np.std(psnrs)),
        ssim_mean=float(np.mean(ssims)),  ssim_std=float(np.std(ssims)),
        lpips_mean=float(np.mean(lpipss)),lpips_std=float(np.std(lpipss)),
        per_image=metrics_per_image,
    )
    with open(f"{out_dir}/metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    torch.save({"recons_01": torch.cat(recon_01_list, dim=0),
                "gts_01":    torch.cat(gt_01_list,    dim=0)},
               f"{out_dir}/tensors.pt")

    print(f"\n[{args.config} group_{args.group}]   "
          f"PSNR={summary['psnr_mean']:.3f}+-{summary['psnr_std']:.3f}  "
          f"SSIM={summary['ssim_mean']:.4f}+-{summary['ssim_std']:.4f}  "
          f"LPIPS={summary['lpips_mean']:.4f}+-{summary['lpips_std']:.4f}", flush=True)
    print(f"Saved -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
