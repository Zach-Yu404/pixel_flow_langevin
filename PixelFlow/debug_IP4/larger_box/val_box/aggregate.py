"""Aggregate the 15 (group, config) runs:

Inputs:
  runs/group{g}_{cfg}/{data.pt, meta.json}  for g in 0..4, cfg in 3 configs
Outputs (in HERE):
  metrics_all_runs.csv
  metrics_by_config.csv
  metrics_by_group.csv
  validation_summary.md

FID:
  - per (group, config): N=100 each (noisy by FID standards; flagged in audit)
  - per config pooled:   N=500 each (5 groups concatenated, less noisy)
  Inception features: piq.feature_extractors.InceptionV3(output_blocks=[3])
  Inputs to Inception: images in [0, 1], will be auto-resized to 299x299 by piq.
"""
import os, sys, json, csv, glob

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(HERE, "runs")

import torch
import numpy as np
import piq
from piq.feature_extractors import InceptionV3


CONFIGS = [
    "pareto_dual_king_srs1e-4__fd1_tr1",
    "LPIPS_king_srs1e-4__fd1_tr1",
    "balanced_perceptual_srs1e-4__fd1_tr1",
]
GROUPS = list(range(5))


def to_img01(t):
    return ((t + 1) / 2).clamp(0, 1)


@torch.no_grad()
def extract_features(images01, inception, device, batch=32):
    """images01: [N, 3, H, W] in [0, 1]. Returns [N, 2048] features (block 3)."""
    feats = []
    inception.eval()
    for i in range(0, images01.shape[0], batch):
        x = images01[i:i+batch].to(device)
        out = inception(x)  # list of feature maps, output_blocks=[3] → 1 element
        f = out[0]  # [b, 2048, 1, 1] or [b, 2048]
        if f.dim() == 4:
            f = f.flatten(1)
        feats.append(f.cpu())
    return torch.cat(feats, dim=0)


def load_run(g, cfg):
    d = os.path.join(RUNS_DIR, f"group{g}_{cfg}")
    data = torch.load(os.path.join(d, "data.pt"), map_location="cpu", weights_only=False)
    meta = json.load(open(os.path.join(d, "meta.json")))
    return data, meta


def main():
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[aggregate] device={DEVICE}")

    inception = InceptionV3(output_blocks=[3], normalize_input=True,
                            requires_grad=False, use_fid_inception=True).to(DEVICE)
    fid_metric = piq.FID()

    # ---- Compute per-(group, config) FID ----
    # Pool features across configs per group: gt features same for all 3 configs in a group
    rows_all = []   # for metrics_all_runs.csv
    pooled_recon_feats = {c: [] for c in CONFIGS}
    pooled_gt_feats = []  # 5 groups × 100 = 500
    seen_gt_per_group = {}

    for g in GROUPS:
        gt_feats_g = None
        for cfg in CONFIGS:
            data, meta = load_run(g, cfg)
            gts    = to_img01(data["gts"])      # [100, 3, 256, 256]
            recons = to_img01(data["recons"])
            assert gts.shape[0] == 100, f"group{g}_{cfg} N={gts.shape[0]} != 100"

            if gt_feats_g is None:
                gt_feats_g = extract_features(gts, inception, DEVICE)
                seen_gt_per_group[g] = gt_feats_g
            recon_feats = extract_features(recons, inception, DEVICE)

            fid_gc = fid_metric.compute_metric(gt_feats_g, recon_feats).item()

            pooled_recon_feats[cfg].append(recon_feats)

            row = dict(
                group_id=g,
                config_name=cfg,
                num_images=int(meta["num_images"]),
                psnr_mean=float(meta["psnr_mean"]),
                ssim_mean=float(meta["ssim_mean"]),
                lpips_mean=float(meta["lpips_mean"]),
                psnr_unobs_mean=float(meta["psnr_unobs_mean"]),
                fid=fid_gc,
                output_dir=os.path.join("runs", f"group{g}_{cfg}"),
                notes="",
            )
            rows_all.append(row)
            print(f"  group {g}  {cfg:50s}  PSNR={row['psnr_mean']:.3f}  "
                  f"SSIM={row['ssim_mean']:.4f}  LPIPS={row['lpips_mean']:.4f}  FID={fid_gc:.2f}")
        pooled_gt_feats.append(gt_feats_g)

    # ---- Pooled FID per config (N=500) ----
    pooled_gt_all = torch.cat(pooled_gt_feats, dim=0)  # [500, 2048]
    pooled_fid = {}
    for cfg in CONFIGS:
        recon_all = torch.cat(pooled_recon_feats[cfg], dim=0)  # [500, 2048]
        pooled_fid[cfg] = fid_metric.compute_metric(pooled_gt_all, recon_all).item()
        print(f"\n[pooled N=500] {cfg}: FID={pooled_fid[cfg]:.2f}")

    # ---- write metrics_all_runs.csv ----
    p_all = os.path.join(HERE, "metrics_all_runs.csv")
    with open(p_all, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_all[0].keys()))
        w.writeheader()
        for r in rows_all:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"\n[wrote] {p_all}  ({len(rows_all)} rows)")

    # ---- metrics_by_config.csv ----
    p_cfg = os.path.join(HERE, "metrics_by_config.csv")
    with open(p_cfg, "w", newline="") as f:
        fields = ["config_name",
                  "psnr_mean", "psnr_std",
                  "ssim_mean", "ssim_std",
                  "lpips_mean", "lpips_std",
                  "fid_mean_pergroup", "fid_std_pergroup",
                  "fid_pooled_n500"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        per_cfg_summary = {}
        for cfg in CONFIGS:
            sub = [r for r in rows_all if r["config_name"] == cfg]
            psnrs = [r["psnr_mean"]  for r in sub]
            ssims = [r["ssim_mean"]  for r in sub]
            lps   = [r["lpips_mean"] for r in sub]
            fids  = [r["fid"]        for r in sub]
            row = dict(
                config_name=cfg,
                psnr_mean=float(np.mean(psnrs)),  psnr_std=float(np.std(psnrs)),
                ssim_mean=float(np.mean(ssims)),  ssim_std=float(np.std(ssims)),
                lpips_mean=float(np.mean(lps)),   lpips_std=float(np.std(lps)),
                fid_mean_pergroup=float(np.mean(fids)), fid_std_pergroup=float(np.std(fids)),
                fid_pooled_n500=float(pooled_fid[cfg]),
            )
            per_cfg_summary[cfg] = row
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()})
    print(f"[wrote] {p_cfg}")

    # ---- metrics_by_group.csv (wide: 5 rows, 3 configs × 4 metrics) ----
    p_grp = os.path.join(HERE, "metrics_by_group.csv")
    with open(p_grp, "w", newline="") as f:
        fields = ["group_id"]
        for cfg in CONFIGS:
            short = cfg.split("_srs1e-4__fd1_tr1")[0]
            for m in ("psnr", "ssim", "lpips", "fid"):
                fields.append(f"{short}__{m}")
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for g in GROUPS:
            row = {"group_id": g}
            for cfg in CONFIGS:
                short = cfg.split("_srs1e-4__fd1_tr1")[0]
                r = next(r for r in rows_all if r["group_id"] == g and r["config_name"] == cfg)
                row[f"{short}__psnr"]  = f"{r['psnr_mean']:.4f}"
                row[f"{short}__ssim"]  = f"{r['ssim_mean']:.4f}"
                row[f"{short}__lpips"] = f"{r['lpips_mean']:.4f}"
                row[f"{short}__fid"]   = f"{r['fid']:.4f}"
            w.writerow(row)
    print(f"[wrote] {p_grp}")

    # ---- validation_summary.md ----
    p_md = os.path.join(HERE, "validation_summary.md")
    # Pick "best overall" using rank-aggregate across PSNR↑, SSIM↑, LPIPS↓, pooled FID↓
    cfg_score = {}
    for metric, higher_is_better in (("psnr_mean", True), ("ssim_mean", True),
                                     ("lpips_mean", False), ("fid_pooled_n500", False)):
        vals = [(per_cfg_summary[c][metric], c) for c in CONFIGS]
        vals.sort(key=lambda x: x[0], reverse=higher_is_better)
        for r, (_, c) in enumerate(vals):
            cfg_score[c] = cfg_score.get(c, 0) + r  # lower rank-sum = better
    best_cfg = min(cfg_score, key=cfg_score.get)
    print(f"\n[best overall by rank-sum] {best_cfg}  (score={cfg_score})")

    with open(p_md, "w") as f:
        f.write("# Validation Summary — 5 groups × 3 configs (N=100 each, total 1500 inferences)\n\n")
        f.write("## Setup\n")
        f.write("- **Val source:** /data/Zach_dataset/imageNet256/ILSVRC/Data/CLS-LOC/val (50000 images)\n")
        f.write("- **Sampling:** 500 disjoint indices via `np.random.RandomState(seed=20260428).choice`, "
                "split into 5 disjoint groups of 100\n")
        f.write("- **Class labels:** per-image, from LOC_val_solution.csv (synset) + LOC_synset_mapping.txt (line→0-based idx)\n")
        f.write("- **Mask:** 128×128 box at random position, **per-image** (operator.base_mask forced to [B,1,H,W]); "
                "mask seed `1000+g` for group g; identical 100 masks across all 3 configs\n")
        f.write("- **Measurement:** y = mask * gt (noiseless, matches existing sweep convention; "
                "sigma_n=0.05 only enters the algorithm prox)\n")
        f.write("- **Configs (all srs=1e-4, fd=1, tr=1):**\n")
        for c in CONFIGS:
            f.write(f"  - `{c}` (kw from `complete_configs/{CONFIG_MAP[c]}.json`)\n")

        f.write("\n## Per-config summary (mean ± std across 5 groups, N=100 each)\n\n")
        f.write("| config | PSNR | SSIM | LPIPS | FID per-group (N=100) | FID pooled (N=500) |\n")
        f.write("|---|---|---|---|---|---|\n")
        for cfg in CONFIGS:
            r = per_cfg_summary[cfg]
            f.write(f"| `{cfg}` | "
                    f"{r['psnr_mean']:.3f} ± {r['psnr_std']:.3f} | "
                    f"{r['ssim_mean']:.4f} ± {r['ssim_std']:.4f} | "
                    f"{r['lpips_mean']:.4f} ± {r['lpips_std']:.4f} | "
                    f"{r['fid_mean_pergroup']:.2f} ± {r['fid_std_pergroup']:.2f} | "
                    f"{r['fid_pooled_n500']:.2f} |\n")

        f.write("\n## All 15 runs\n\n")
        f.write("| group | config | PSNR | SSIM | LPIPS | FID(N=100) |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in rows_all:
            f.write(f"| {r['group_id']} | `{r['config_name']}` | "
                    f"{r['psnr_mean']:.3f} | {r['ssim_mean']:.4f} | "
                    f"{r['lpips_mean']:.4f} | {r['fid']:.2f} |\n")

        f.write(f"\n## Best overall config\n\n")
        f.write(f"**`{best_cfg}`** (rank-sum across PSNR↑, SSIM↑, LPIPS↓, pooled-FID↓; lower is better).\n\n")
        f.write("Rank scores:\n")
        for c in CONFIGS:
            f.write(f"- `{c}`: {cfg_score[c]}\n")
        f.write("\nSee also `metric_logic_audit.md` for noteworthy considerations (FID with N=100 noise, etc.).\n")
    print(f"[wrote] {p_md}")
    print(f"\n[best overall] {best_cfg}")


# imported in summary writer
CONFIG_MAP = {
    "pareto_dual_king_srs1e-4__fd1_tr1":     "pareto_dual_king_srs1e-4",
    "LPIPS_king_srs1e-4__fd1_tr1":           "LPIPS_king_srs1e-4",
    "balanced_perceptual_srs1e-4__fd1_tr1":  "balanced_perceptual_srs1e-4",
}


if __name__ == "__main__":
    main()
