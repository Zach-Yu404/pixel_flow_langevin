"""Aggregate gamma2_stats/test/results.csv: per (arm, task) mean±std over 7 images x 3 seeds; paired per-cell differences
vs baseline (same task/image/seed) with win counts; overall means; markdown + csv + a PSNR bar figure."""
import os, csv, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
T = "/CBIG-Standard-ECE/Zach/MSFlow/PixelFlowICLR/Algorithm2/gamma2_stats/test"
d = pd.read_csv(f"{T}/results.csv").drop_duplicates(["arm", "task", "image", "seed"], keep="last")
ARMS = ["baseline", "gamma2_all", "gamma2_labelled", "baseline_rerun"]; TASKS = ["box_inpainting", "random_inpainting", "gaussian_blur", "motion_blur", "superresolution"]
M = ["psnr", "ssim", "lpips_alex", "lpips_piq", "mse_full", "mse_hole"]
g = d.groupby(["arm", "task"])[M].agg(["mean", "std"]); g.columns = [f"{a}_{b}" for a, b in g.columns]; g = g.join(d.groupby(["arm", "task"]).size().rename("n")).reset_index()
g.to_csv(f"{T}/per_task.csv", index=False, float_format="%.4f")
# paired differences vs baseline
b = d[d.arm == "baseline"].set_index(["task", "image", "seed"])
exp = 5*7*3*len(ARMS); missing = exp - len(d)
md = [f"# gamma2 tables under spectral_class S — {len(d)}/{exp} cells (5 tasks x 7 images x 3 seeds x {len(ARMS)} arms); missing={missing}\n",
      "baseline_rerun = same table and seed as baseline: its Δ is the GPU run-to-run noise floor.\n",
      "Mean over 7 images x 3 seeds; Δ = paired per-cell difference vs baseline (same task/image/seed), wins = cells where the arm beats baseline.\n"]
for metric, better in [("psnr", "higher"), ("ssim", "higher"), ("lpips_alex", "lower"), ("mse_hole", "lower")]:
    md += [f"\n## {metric} ({better} is better)\n", "| task | baseline | gamma2_all | Δ_all (wins) | gamma2_labelled | Δ_lab (wins) | baseline_rerun | Δ_rerun = GPU noise floor (wins) |", "|---|---|---|---|---|---|---|---|"]
    for t in TASKS + ["ALL"]:
        sub = d if t == "ALL" else d[d.task == t]
        if t != "ALL" and metric == "mse_hole" and "inpaint" not in t: continue
        cells = []
        base = sub[sub.arm == "baseline"][metric]
        if base.isna().all(): continue
        cells.append(f"{base.mean():.4f}" if metric != "psnr" else f"{base.mean():.2f}")
        for arm in ARMS[1:]:
            a = sub[sub.arm == arm].set_index(["task", "image", "seed"])[metric]
            bb = b[metric].reindex(a.index); diff = (a - bb).dropna()
            wins = int(((diff > 0) if better == "higher" else (diff < 0)).sum()); n = len(diff)
            val = a.mean(); cells.append(f"{val:.4f}" if metric != "psnr" else f"{val:.2f}")
            cells.append(f"{diff.mean():+.4f}±{diff.std():.4f} ({wins}/{n})" if metric != "psnr" else f"{diff.mean():+.2f}±{diff.std():.2f} ({wins}/{n})")
        md.append(f"| {t} | " + " | ".join(cells) + " |")
open(f"{T}/summary.md", "w").write("\n".join(md) + "\n"); print("\n".join(md))
# figure: PSNR per task per arm with std over cells
fig, ax = plt.subplots(figsize=(9, 3.8)); w = 0.26; x = np.arange(len(TASKS))
for i, arm in enumerate(ARMS):
    m = [d[(d.arm == arm) & (d.task == t)].psnr.mean() for t in TASKS]; s = [d[(d.arm == arm) & (d.task == t)].psnr.std() for t in TASKS]
    ax.bar(x + (i - 1) * w, m, w, yerr=s, capsize=2, label=arm)
ax.set_xticks(x); ax.set_xticklabels(TASKS, fontsize=8); ax.set_ylabel("PSNR (dB), 7 img x 3 seeds"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3)
ax.set_title("gamma^2 table comparison under spectral_class S, [2,2,1,1], all noise"); plt.tight_layout(); plt.savefig(f"{T}/psnr_per_task.png", dpi=120)
