"""Per-GPU runner. Same protocol as larger_box sweeps but B=8 (load 8
deterministic GT images), mask seed=7919 (box position fixed)."""
import os, sys, json, time, argparse

HERE = os.path.dirname(os.path.abspath(__file__))           # full_validation
FC   = os.path.dirname(HERE)                                # final_configs
LB   = os.path.dirname(FC)                                  # larger_box
DBG  = os.path.dirname(LB)                                  # debug_IP4
REPO = os.path.dirname(DBG)                                 # PixelFlow
for p in (HERE, FC, LB, DBG, REPO): sys.path.insert(0, p)
os.chdir(REPO)

import torch, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms

import piq
from inpaintingStart import get_operator
from pixelflow.utils import config as config_utils
from ms_sampler_v5 import run_ip4, hf_energy
from fv_configs import CONFIGS


RUN_DIR = os.path.join(HERE, "runs")
os.makedirs(RUN_DIR, exist_ok=True)

VAL = "/data/Zach_dataset/imageNet256/ILSVRC/Data/CLS-LOC/val"


def cca(p, sz=256):
    while min(*p.size) >= 2 * sz:
        p = p.resize(tuple(x // 2 for x in p.size), Image.BOX)
    sc = sz / min(*p.size)
    p = p.resize(tuple(round(x * sc) for x in p.size), Image.BICUBIC)
    a = np.array(p); cy = (a.shape[0]-sz)//2; cx = (a.shape[1]-sz)//2
    return Image.fromarray(a[cy:cy+sz, cx:cx+sz])


def load_8_gt(device):
    """8 deterministic GT with per-sample class labels.

    [0:4] from baseline_05 — labels=[10, 10, 10, 10]   (brambling)
    [4:8] from ImageNet val first 4 sorted by filename:
          val_00000001 -> n01751748 -> 65   (sea snake)
          val_00000002 -> n09193705 -> 970  (alp)
          val_00000003 -> n02105855 -> 230  (Shetland sheepdog)
          val_00000004 -> n04263257 -> 809  (soup bowl)
    """
    pt = "trajectory_videos/posterior_sampling/baseline_05/baseline_05_mode-f_x1.pt"
    gt_a = torch.load(pt, map_location="cpu", weights_only=False)["gt"][:4]  # 4 imgs

    tf = transforms.Compose([
        transforms.Lambda(lambda p: cca(p, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    fns = sorted(os.listdir(VAL))[:4]
    gt_b = torch.stack([tf(Image.open(os.path.join(VAL, fn)).convert("RGB")) for fn in fns])

    gt = torch.cat([gt_a, gt_b], dim=0).to(device)
    labels = [10, 10, 10, 10, 65, 970, 230, 809]
    return gt, labels


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

    gt, class_labels = load_8_gt(DEVICE)
    B = gt.shape[0]; sigma_n = 0.05
    print(f"  B={B}, gt range=[{gt.min().item():.3f}, {gt.max().item():.3f}]", flush=True)
    print(f"  class_labels (per-sample) = {class_labels}", flush=True)

    # MASK SEED = 42 (numpy + torch + cuda; _retrieve_box uses np.random.randint)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
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
        print(f"\n[{gi+1}/{len(CONFIGS)}] {name}", flush=True)
        t0 = time.time()
        try:
            xf, _, _, res, _, hf = run_ip4(
                model, config, gt, y, operator, sigma_n, DEVICE,
                class_label=class_labels, seed=42, **kw,
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

        # 8x3 grid: GT | meas | recon for each of 8 samples
        fig, axes = plt.subplots(B, 3, figsize=(11, 3.4*B))
        if B == 1: axes = axes[None, :]
        for b in range(B):
            gi_img = to_img(gt[b])
            yi_img = to_img((mask[b]*gt[b] + (1-mask[b])*(-torch.ones_like(gt[b]))).cpu())
            xfi = to_img(xfD[b])
            axes[b,0].imshow(gi_img); axes[b,0].set_title(f"GT[{b}]", fontsize=8); axes[b,0].axis("off")
            axes[b,1].imshow(yi_img); axes[b,1].set_title("Meas", fontsize=8); axes[b,1].axis("off")
            axes[b,2].imshow(xfi);    axes[b,2].set_title(f"PSNR={per_img[b]['psnr']:.2f} LPIPS={per_img[b]['lpips']:.4f}", fontsize=8); axes[b,2].axis("off")
        plt.suptitle(f"{name}\nPSNR={psnr:.2f} SSIM={ssim:.4f} LPIPS={lpips:.4f} |dHF|={dhf:.3f}", fontsize=10)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(f"{RUN_DIR}/{name}.png", dpi=120); plt.close()

    print(f"\n[gpu={args.gpu}] done.", flush=True)


if __name__ == "__main__":
    main()
