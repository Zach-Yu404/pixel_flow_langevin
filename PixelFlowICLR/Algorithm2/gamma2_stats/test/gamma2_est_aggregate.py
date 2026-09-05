"""Aggregate results_est.csv: per (estimator, arm, task) mean over 7 images (+ ALL), paired Δ vs baseline per cell."""
import numpy as np, pandas as pd
T = "/CBIG-Standard-ECE/Zach/MSFlow/PixelFlowICLR/Algorithm2/gamma2_stats/test"
d = pd.read_csv(f"{T}/results_est.csv").drop_duplicates(["arm", "task", "image", "estimator"], keep="last")
ARMS = ["baseline", "gamma2_all", "gamma2_labelled"]; EST = ["single_avg10", "MMSE5", "MMSE10", "no_xi0", "no_noise"]
TASKS = ["box_inpainting", "random_inpainting", "gaussian_blur", "motion_blur", "superresolution"]
exp = 3 * 5 * 7 * 5
md = [f"# gamma2 tables × estimators under spectral_class S — {len(d)}/{exp} rows (3 arms × 5 tasks × 7 images × 5 estimators)\n",
      "Mean over 7 images; Δ = paired per-image difference vs baseline (same task/image/estimator); wins = images where the arm beats baseline.\n"]
for metric, better in [("psnr", "higher"), ("ssim", "higher"), ("lpips_alex", "lower")]:
    md += [f"\n## {metric} ({better} is better)\n", "| estimator | task | baseline | gamma2_all | Δ_all (wins) | gamma2_labelled | Δ_lab (wins) |", "|---|---|---|---|---|---|---|"]
    for est in EST:
        for t in TASKS + ["ALL"]:
            sub = d[d.estimator == est]; sub = sub if t == "ALL" else sub[sub.task == t]
            b = sub[sub.arm == "baseline"].set_index(["task", "image"])[metric]
            if not len(b): continue
            fmt = (lambda v: f"{v:.2f}") if metric == "psnr" else (lambda v: f"{v:.4f}")
            cells = [fmt(b.mean())]
            for arm in ARMS[1:]:
                a = sub[sub.arm == arm].set_index(["task", "image"])[metric]; diff = (a - b.reindex(a.index)).dropna()
                wins = int(((diff > 0) if better == "higher" else (diff < 0)).sum())
                cells += [fmt(a.mean()) if len(a) else "–", (f"{diff.mean():+.2f}±{diff.std():.2f} ({wins}/{len(diff)})" if metric == "psnr" else f"{diff.mean():+.4f}±{diff.std():.4f} ({wins}/{len(diff)})") if len(diff) else "–"]
            md.append(f"| {est} | {t} | " + " | ".join(cells) + " |")
open(f"{T}/summary_est.md", "w").write("\n".join(md) + "\n"); print("\n".join(md))
g = d.groupby(["estimator", "arm", "task"])[["psnr", "ssim", "lpips_alex", "lpips_piq", "mse_full", "mse_hole"]].agg(["mean", "std"]); g.columns = [f"{a}_{b}" for a, b in g.columns]
g.reset_index().to_csv(f"{T}/per_task_est.csv", index=False, float_format="%.4f")
