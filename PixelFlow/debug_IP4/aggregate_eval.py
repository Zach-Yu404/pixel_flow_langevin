"""Aggregate validation eval results: 5 groups × 100 images per task.
Compute mean ± std of PSNR/SSIM/LPIPS across groups, plus FID over all 500
recon-vs-GT pairs (per piq.FID protocol)."""
import os, json
import numpy as np
import torch
import piq

OUT_BASE = "/data/Zach_dataset/imageNet_results/imageNet"


def load_group_metrics(task, group):
    p = f"{OUT_BASE}/{task}/group_{group}/metrics.json"
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def load_group_tensors(task, group):
    p = f"{OUT_BASE}/{task}/group_{group}/tensors.pt"
    if not os.path.exists(p):
        return None, None
    d = torch.load(p, map_location="cpu", weights_only=False)
    return d["recons_01"], d["gts_01"]


def compute_fid(recons_01, gts_01, device="cuda:0", feature_layer="inception"):
    """Use piq.FID with default Inception feature extractor."""
    fid_metric = piq.FID()
    # piq.FID expects features extracted via piq.feature_extractor.InceptionV3
    # The piq.FID() class itself can take feature tensors, NOT raw images.
    # piq.compute_feats(loader, feature_extractor) is the convenience.
    from torch.utils.data import DataLoader, TensorDataset

    feature_extractor = piq.FID()._build_feature_extractor("inception_v3").to(device)
    feature_extractor.eval()

    def extract(imgs):
        loader = DataLoader(TensorDataset(imgs), batch_size=16)
        feats = []
        with torch.no_grad():
            for (b,) in loader:
                b = b.to(device)
                if b.shape[1] != 3:
                    b = b.expand(-1, 3, -1, -1)
                f = feature_extractor(b)
                if isinstance(f, list):
                    f = f[0]
                f = f.flatten(1)
                feats.append(f.cpu())
        return torch.cat(feats, dim=0)

    fr = extract(recons_01)
    fg = extract(gts_01)
    return fid_metric(fr, fg).item()


def compute_fid_simple(recons_01, gts_01, device="cuda:0"):
    """Simpler API: piq.FID()(features_a, features_b). Use piq.feature_extractor."""
    # piq has a top-level computer:
    # piq.FID()(real_features, generated_features)
    # Need to extract features first via inception
    try:
        from piq.feature_extractor import InceptionV3
    except ImportError:
        # piq 0.8 uses different path
        from piq import FID
        feat_ext = FID()
        # Many piq versions have inception inside. Fall back to a manual route.
        # Use torchvision inception_v3 with logits removed.
        from torchvision.models import inception_v3
        m = inception_v3(pretrained=True, transform_input=False).to(device)
        m.fc = torch.nn.Identity()
        m.eval()

        def extract(imgs):
            from torch.nn.functional import interpolate
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            feats = []
            with torch.no_grad():
                for i in range(0, imgs.shape[0], 16):
                    b = imgs[i:i+16].to(device)
                    b = interpolate(b, size=(299, 299), mode="bilinear", align_corners=False)
                    b = (b - mean) / std
                    feats.append(m(b).cpu())
            return torch.cat(feats, dim=0)

        fr = extract(recons_01)
        fg = extract(gts_01)
        return float(piq.FID()(fr, fg))


def main():
    print(f"{'='*70}\nVALIDATION EVAL AGGREGATE\n{'='*70}", flush=True)
    summary = {}
    for task in ["box", "random"]:
        print(f"\n--- {task.upper()} ---", flush=True)
        per_group_means = {"psnr": [], "ssim": [], "lpips": []}
        recons_all, gts_all = [], []
        for g in range(5):
            m = load_group_metrics(task, g)
            if m is None:
                print(f"  group_{g}: missing metrics.json", flush=True)
                continue
            per_group_means["psnr"].append(m["psnr_mean"])
            per_group_means["ssim"].append(m["ssim_mean"])
            per_group_means["lpips"].append(m["lpips_mean"])
            r, gt = load_group_tensors(task, g)
            if r is not None:
                recons_all.append(r); gts_all.append(gt)
            print(f"  group_{g}: PSNR={m['psnr_mean']:.3f}  SSIM={m['ssim_mean']:.4f}  LPIPS={m['lpips_mean']:.4f}", flush=True)

        if not per_group_means["psnr"]:
            print(f"  no groups loaded", flush=True); continue

        # mean ± std across 5 groups
        agg = {}
        for k in ["psnr", "ssim", "lpips"]:
            arr = np.array(per_group_means[k])
            agg[f"{k}_mean"] = float(arr.mean())
            agg[f"{k}_std"]  = float(arr.std(ddof=0))

        # FID across all 500 reconstructions vs all 500 GTs (single FID, not per-group)
        if recons_all:
            recons_cat = torch.cat(recons_all, dim=0)
            gts_cat    = torch.cat(gts_all,    dim=0)
            print(f"  computing FID over {recons_cat.shape[0]} pairs...", flush=True)
            fid_v = compute_fid_simple(recons_cat, gts_cat, device="cuda:0")
            agg["fid"] = float(fid_v)

        summary[task] = agg
        print(f"\n  {task} aggregate:")
        print(f"    PSNR  = {agg['psnr_mean']:.3f} ± {agg['psnr_std']:.3f}")
        print(f"    SSIM  = {agg['ssim_mean']:.4f} ± {agg['ssim_std']:.4f}")
        print(f"    LPIPS = {agg['lpips_mean']:.4f} ± {agg['lpips_std']:.4f}")
        if "fid" in agg:
            print(f"    FID   = {agg['fid']:.3f}")

    with open(f"{OUT_BASE}/aggregate_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n→ saved {OUT_BASE}/aggregate_summary.json", flush=True)


if __name__ == "__main__":
    main()
