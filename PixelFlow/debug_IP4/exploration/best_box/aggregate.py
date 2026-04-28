"""Aggregate best_box results.

Per group (g = 0..4) compute:
  PSNR_g, SSIM_g, LPIPS_g  -- mean over the 100 images in that group
  FID_g                    -- FID over the 100 (recon, GT) pairs in that group

Then aggregate as mean +- std across the 5 groups (one number per group, std
across groups).

ALSO compute fid_pooled_500 -- single FID over all 500 (recon, GT) pairs
concatenated across the 5 groups. This number is directly comparable to the
old /data/Zach_dataset/imageNet_results protocol. FID has a positive bias
that scales as ~1/N, so per-group FID at N=100 is ~50 units higher than
the asymptotic FID at N=500; pooled FID is the cross-protocol reference.

Output: <OUT_BASE>/aggregate_summary.json + printed table.
"""
import os, json
import numpy as np
import torch
import piq

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_BASE = os.path.join(HERE, "output")
CFG_DIR  = os.path.join(HERE, "configs")
# Auto-discover all configs (base + CFG sweep), preserving a deterministic order.
BASE_NAMES = [
    "A0_reference_WINNER3",
    "E1_W1+lam20",
    "C1_W1+hx_multistage",
    "H1_W1+X4+noise0.3",
    "S3_stage3_only",
]
CFG_VARIANTS = ["", "_cfg1", "_cfg2", "_cfg3", "_cfg4"]
CONFIGS = [b + v for b in BASE_NAMES for v in CFG_VARIANTS
           if os.path.exists(os.path.join(OUT_BASE, b + v))]
DEVICE = "cuda:0"


def load_group_metrics(cfg, g):
    p = f"{OUT_BASE}/{cfg}/group_{g}/metrics.json"
    return json.load(open(p)) if os.path.exists(p) else None


def load_group_tensors(cfg, g):
    p = f"{OUT_BASE}/{cfg}/group_{g}/tensors.pt"
    if not os.path.exists(p):
        return None, None
    d = torch.load(p, map_location="cpu", weights_only=False)
    return d["recons_01"], d["gts_01"]


def build_inception(device):
    from torchvision.models import inception_v3
    m = inception_v3(pretrained=True, transform_input=False).to(device)
    m.fc = torch.nn.Identity(); m.eval()
    return m


def extract_inception_feats(imgs_01, model, device):
    from torch.nn.functional import interpolate
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    feats = []
    with torch.no_grad():
        for i in range(0, imgs_01.shape[0], 16):
            b = imgs_01[i:i+16].to(device)
            b = interpolate(b, size=(299, 299), mode="bilinear", align_corners=False)
            b = (b - mean) / std
            feats.append(model(b).cpu())
    return torch.cat(feats, dim=0)


def compute_group_fid(recons_01, gts_01, inception):
    fr = extract_inception_feats(recons_01, inception, DEVICE)
    fg = extract_inception_feats(gts_01,    inception, DEVICE)
    return float(piq.FID()(fr, fg))


def aggregate_one(cfg, inception):
    print(f"\n--- {cfg} ---", flush=True)
    per_group = {"psnr": [], "ssim": [], "lpips": [], "fid": []}
    group_rows = []
    recons_all, gts_all = [], []

    for g in range(5):
        m = load_group_metrics(cfg, g)
        if m is None:
            print(f"  group_{g}: missing metrics.json", flush=True)
            continue
        psnr_g  = m["psnr_mean"]
        ssim_g  = m["ssim_mean"]
        lpips_g = m["lpips_mean"]

        r, gt = load_group_tensors(cfg, g)
        fid_g = None
        if r is not None and r.shape[0] > 1:
            fid_g = compute_group_fid(r, gt, inception)
            recons_all.append(r); gts_all.append(gt)

        per_group["psnr"].append(psnr_g)
        per_group["ssim"].append(ssim_g)
        per_group["lpips"].append(lpips_g)
        if fid_g is not None:
            per_group["fid"].append(fid_g)

        group_rows.append(dict(group=g, n=m["n"],
                               psnr=psnr_g, ssim=ssim_g, lpips=lpips_g,
                               fid=fid_g))
        fid_str = f"{fid_g:.3f}" if fid_g is not None else "  --  "
        print(f"  group_{g}: PSNR={psnr_g:.3f}  SSIM={ssim_g:.4f}  LPIPS={lpips_g:.4f}  FID@100={fid_str}", flush=True)

    if not per_group["psnr"]:
        return None

    agg = {"per_group": group_rows}
    for k in ["psnr", "ssim", "lpips", "fid"]:
        if per_group[k]:
            arr = np.array(per_group[k])
            agg[f"{k}_mean"] = float(arr.mean())
            agg[f"{k}_std"]  = float(arr.std(ddof=0))

    if recons_all:
        R = torch.cat(recons_all, dim=0)
        G = torch.cat(gts_all,    dim=0)
        print(f"  computing pooled-{R.shape[0]} FID ...", flush=True)
        fr_all = extract_inception_feats(R, inception, DEVICE)
        fg_all = extract_inception_feats(G, inception, DEVICE)
        agg["fid_pooled"]   = float(piq.FID()(fr_all, fg_all))
        agg["fid_pooled_n"] = int(R.shape[0])

    print(f"  -> PSNR    = {agg['psnr_mean']:.3f} +- {agg['psnr_std']:.3f}")
    print(f"  -> SSIM    = {agg['ssim_mean']:.4f} +- {agg['ssim_std']:.4f}")
    print(f"  -> LPIPS   = {agg['lpips_mean']:.4f} +- {agg['lpips_std']:.4f}")
    if "fid_mean" in agg:
        print(f"  -> FID@100 = {agg['fid_mean']:.3f} +- {agg['fid_std']:.3f}  (per-group, biased high)")
    if "fid_pooled" in agg:
        print(f"  -> FID@{agg['fid_pooled_n']} = {agg['fid_pooled']:.3f}        (pooled, comparable to old runs)")
    return agg


def main():
    print(f"{'='*78}\nBEST-OF-BOX AGGREGATE  (5 groups x 100 imgs, 128x128 random-pos box)\n"
          f"per-group metric, mean +- std across 5 groups\n{'='*78}", flush=True)
    inception = build_inception(DEVICE)
    summary = {}
    for cfg in CONFIGS:
        agg = aggregate_one(cfg, inception)
        if agg is not None:
            summary[cfg] = agg

    print(f"\n{'='*92}\n{'CONFIG':<28} {'PSNR':>14} {'SSIM':>14} {'LPIPS':>14} {'FID@100':>14} {'FID@500':>9}\n{'='*92}")
    for cfg, a in summary.items():
        psnr = f"{a['psnr_mean']:.2f}+-{a['psnr_std']:.2f}"
        ssim = f"{a['ssim_mean']:.3f}+-{a['ssim_std']:.3f}"
        lp   = f"{a['lpips_mean']:.3f}+-{a['lpips_std']:.3f}"
        fid  = f"{a.get('fid_mean', float('nan')):.2f}+-{a.get('fid_std', float('nan')):.2f}"
        fid_p = f"{a.get('fid_pooled', float('nan')):.2f}"
        print(f"{cfg:<28} {psnr:>14} {ssim:>14} {lp:>14} {fid:>14} {fid_p:>9}")

    out = os.path.join(OUT_BASE, "aggregate_summary.json")
    os.makedirs(OUT_BASE, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n-> saved {out}", flush=True)


if __name__ == "__main__":
    main()
