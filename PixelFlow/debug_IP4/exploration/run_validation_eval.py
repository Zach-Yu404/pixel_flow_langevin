"""
Validation evaluation per DAPS-style protocol.

Setup
-----
- E1_W1+lam20 setting: h_eps=0.001, lambda_reg=[50,50,50,20], terminal_replace=1.0
- Box inpainting: 128×128 mask
- Random inpainting: 70% of pixels masked
- 5 groups × 100 images each
- Validation set: ImageNet val (50k images), with proper class labels
- Metrics: PSNR, SSIM, LPIPS (replace_pooling=True), FID — all from `piq`
- Images normalized to [0,1] before metric eval
- Output: /data/Zach_dataset/imageNet_results/imageNet/{box,random}/

Usage:
    python run_validation_eval.py --group N --task box|random --gpu i
"""
import os, sys, json, time, argparse, csv
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from omegaconf import OmegaConf

import piq
from inpaintingStart import get_operator
from pixelflow.utils import config as config_utils

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "debug_IP4"))
from ms_sampler_v5 import run_ip4

VAL_DIR  = "/data/Zach_dataset/imageNet256/ILSVRC/Data/CLS-LOC/val"
VAL_LBL  = "/data/Zach_dataset/imageNet256/LOC_val_solution.csv"
SYNSET   = "/data/Zach_dataset/imageNet256/LOC_synset_mapping.txt"
OUT_BASE = "/data/Zach_dataset/imageNet_results/imageNet"

GROUP_SEEDS = [11, 22, 33, 44, 55]   # 5 fixed group seeds
N_PER_GROUP = 100
BATCH_SIZE  = 4                       # GPU memory permitting

# E1_W1+lam20 winning config
E1_KW = dict(
    h_epsilon=0.001,
    lambda_reg=[50, 50, 50, 20],
    h_x=0.1,                  # default
    terminal_replace_weight=1.0,
)


def load_val_label_map():
    """Returns dict {image_filename_no_ext: imagenet_class_index}."""
    synset_to_idx = {}
    with open(SYNSET) as f:
        for i, line in enumerate(f):
            synset = line.split()[0]
            synset_to_idx[synset] = i

    fname_to_idx = {}
    with open(VAL_LBL) as f:
        rdr = csv.reader(f)
        next(rdr)  # header
        for row in rdr:
            fname, pred = row[0], row[1]
            synset = pred.split()[0]   # e.g. n03995372
            fname_to_idx[fname] = synset_to_idx[synset]
    return fname_to_idx


def select_group(group_idx, all_files, label_map):
    """Deterministic 100-image selection for given group."""
    rng = np.random.RandomState(GROUP_SEEDS[group_idx])
    perm = rng.permutation(len(all_files))[:N_PER_GROUP]
    selected = [(all_files[i], label_map[all_files[i].replace(".JPEG", "")]) for i in perm]
    return selected


def center_crop_arr(pil_image, size):
    """Center-crop & resize to (size, size). Same as repo helper."""
    while min(*pil_image.size) >= 2 * size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), Image.BOX)
    scale = size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), Image.BICUBIC)
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - size) // 2
    crop_x = (arr.shape[1] - size) // 2
    return Image.fromarray(arr[crop_y:crop_y + size, crop_x:crop_x + size])


def make_operator(task, sigma_n=0.05, device="cuda:0", seed_override=None):
    """Build inpainting operator. mask gets re-randomized for each operator instance."""
    if task == "box":
        # 128×128 mask  (mask_len_range high is exclusive in randint, so use 129)
        op = get_operator(
            "inpainting", resolution=256, device=device, sigma=sigma_n,
            mask_type="box", mask_len_range=(128, 129), mask_prob_range=None,
        )
    else:  # random
        op = get_operator(
            "inpainting", resolution=256, device=device, sigma=sigma_n,
            mask_type="random", mask_len_range=None, mask_prob_range=(0.7, 0.7),
        )
    return op


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=int, required=True, choices=range(5))
    parser.add_argument("--task",  type=str, required=True, choices=["box", "random"])
    parser.add_argument("--gpu",   type=int, default=0)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    DEVICE = f"cuda:{args.gpu}"
    out_dir = os.path.join(OUT_BASE, args.task, f"group_{args.group}")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"{out_dir}/recons", exist_ok=True)
    os.makedirs(f"{out_dir}/gts",    exist_ok=True)
    os.makedirs(f"{out_dir}/ys",     exist_ok=True)

    # Resolve image filenames + class labels for this group
    label_map = load_val_label_map()
    all_files = sorted(f for f in os.listdir(VAL_DIR) if f.endswith(".JPEG"))
    selected = select_group(args.group, all_files, label_map)

    print(f"[GPU {args.gpu}] task={args.task}  group={args.group}  N={len(selected)}", flush=True)

    # Load model once
    config = OmegaConf.load("pretrained_models/c2img/config.yaml")
    model = config_utils.instantiate_from_config(config.model).to(DEVICE)
    ckpt = torch.load("pretrained_models/c2img/model.pt",
                      map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt, strict=True); model.eval()

    # piq metrics
    lpips_fn = piq.LPIPS(replace_pooling=True).to(DEVICE)

    sigma_n = 0.05
    tf = transforms.Compose([
        transforms.Lambda(lambda p: center_crop_arr(p, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),  # → [-1,1]
    ])

    metrics_per_image = []
    recon_01_list, gt_01_list = [], []  # for FID later (per group)

    for batch_start in range(0, len(selected), args.batch):
        batch_items = selected[batch_start:batch_start + args.batch]
        gts, lbls, fnames = [], [], []
        for fname, lbl in batch_items:
            img = Image.open(os.path.join(VAL_DIR, fname)).convert("RGB")
            x = tf(img)
            gts.append(x); lbls.append(lbl); fnames.append(fname)
        gt = torch.stack(gts).to(DEVICE)
        B = gt.shape[0]

        # Build operator (fresh mask per batch — DAPS uses different masks)
        # Actually DAPS uses ONE mask across all images per task. Use group-seeded mask.
        torch.manual_seed(GROUP_SEEDS[args.group] * 1000 + batch_start)
        operator = make_operator(args.task, sigma_n, DEVICE)
        y = operator(gt).detach()
        mask = operator.get_mask(x=gt).float().to(DEVICE)
        if mask.shape[0] == 1 and B > 1:
            mask_b = mask.expand(B, -1, -1, -1)
        else:
            mask_b = mask

        t0 = time.time()
        xf, _, _, res, t_run, _ = run_ip4(
            model, config, gt, y, operator, sigma_n, DEVICE,
            class_label=lbls,         # per-sample labels (list)
            seed=GROUP_SEEDS[args.group] * 1000 + batch_start,
            **E1_KW,
        )
        xf = xf.to(DEVICE)
        elapsed = time.time() - t0
        print(f"  batch {batch_start}-{batch_start+B-1} done in {elapsed:.0f}s", flush=True)

        # Compute metrics in [0,1] convention
        recon_01 = ((xf + 1) / 2).clamp(0, 1)
        gt_01    = ((gt + 1) / 2).clamp(0, 1)
        y_01     = ((y  + 1) / 2).clamp(0, 1)

        with torch.no_grad():
            for i in range(B):
                r1 = recon_01[i:i+1]; g1 = gt_01[i:i+1]
                psnr_v = piq.psnr(r1, g1, data_range=1.0).item()
                ssim_v = piq.ssim(r1, g1, data_range=1.0).item()
                lpips_v = lpips_fn(r1, g1).item()
                metrics_per_image.append(dict(
                    file=fnames[i], class_label=int(lbls[i]),
                    psnr=psnr_v, ssim=ssim_v, lpips=lpips_v,
                ))

                # save images
                idx = batch_start + i
                _save_img(r1[0], f"{out_dir}/recons/{idx:04d}.png")
                _save_img(g1[0], f"{out_dir}/gts/{idx:04d}.png")
                _save_img(y_01[i], f"{out_dir}/ys/{idx:04d}.png")
                recon_01_list.append(r1.detach().cpu())
                gt_01_list.append(g1.detach().cpu())

    # save per-image metrics
    psnrs  = [m["psnr"]  for m in metrics_per_image]
    ssims  = [m["ssim"]  for m in metrics_per_image]
    lpipss = [m["lpips"] for m in metrics_per_image]
    summary = dict(
        task=args.task, group=args.group, n=len(metrics_per_image),
        psnr_mean=float(np.mean(psnrs)),  psnr_std=float(np.std(psnrs)),
        ssim_mean=float(np.mean(ssims)),  ssim_std=float(np.std(ssims)),
        lpips_mean=float(np.mean(lpipss)), lpips_std=float(np.std(lpipss)),
        per_image=metrics_per_image,
    )
    with open(f"{out_dir}/metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    # save tensors for FID computation later
    torch.save({"recons_01": torch.cat(recon_01_list, dim=0),
                "gts_01":    torch.cat(gt_01_list,    dim=0)},
               f"{out_dir}/tensors.pt")

    print(f"\n[group_{args.group} {args.task}]   "
          f"PSNR={summary['psnr_mean']:.3f}±{summary['psnr_std']:.3f}  "
          f"SSIM={summary['ssim_mean']:.4f}±{summary['ssim_std']:.4f}  "
          f"LPIPS={summary['lpips_mean']:.4f}±{summary['lpips_std']:.4f}", flush=True)
    print(f"Saved → {out_dir}", flush=True)


def _save_img(x01, path):
    """x01 shape (3, H, W), values in [0,1]. Save as PNG."""
    arr = (x01.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


if __name__ == "__main__":
    main()
