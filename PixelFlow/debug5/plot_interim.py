#!/usr/bin/env python
"""Plot interim results from debug5 sweep — run anytime to see current state."""
import re, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "sweep_stdout.log")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

def parse_log(path):
    results = []
    gt_hf = None
    with open(path) as f:
        for line in f:
            m = re.match(r"BOX INPAINTING\s+GT HF_unobs=(\S+)", line)
            if m:
                gt_hf = float(m.group(1))
            # Match result lines: "1   A0_raw   47  15.72  7.51  0.8936  0.646  0.072  267"
            m = re.match(
                r"\s*(\d+)\s+(\S+)\s+([\d.inf-]+)\s+([\d.inf-]+)\s+([\d.inf-]+)\s+"
                r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)", line)
            if m:
                results.append(dict(
                    idx=int(m.group(1)), name=m.group(2),
                    res=float(m.group(3)), psnr=float(m.group(4)),
                    psnr_u=float(m.group(5)), ssim=float(m.group(6)),
                    hf=float(m.group(7)), dhf=float(m.group(8)),
                    time=int(m.group(9)),
                ))
    return results, gt_hf

results, gt_hf = parse_log(LOG)
if not results:
    print("No results yet."); sys.exit(0)

names = [r["name"] for r in results]
psnrs = [r["psnr"] for r in results]
hfs   = [r["hf"] for r in results]
dhfs  = [r["dhf"] for r in results]
ssims = [r["ssim"] for r in results]
times = [r["time"] for r in results]

# Color by group
def group_color(name):
    prefix = name[0]
    colors = {"A": "#2196F3", "B": "#FF9800", "C": "#4CAF50", "D": "#F44336",
              "E": "#9C27B0", "F": "#00BCD4", "G": "#795548", "H": "#607D8B",
              "I": "#E91E63", "J": "#3F51B5", "K": "#FF5722", "L": "#009688"}
    return colors.get(prefix, "#999999")

colors = [group_color(n) for n in names]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. PSNR vs HF scatter
ax = axes[0, 0]
for i, r in enumerate(results):
    ax.scatter(r["hf"], r["psnr"], c=colors[i], s=60, zorder=3, edgecolors="k", linewidths=0.3)
    ax.annotate(r["name"], (r["hf"], r["psnr"]), fontsize=4, ha="center", va="bottom",
                xytext=(0, 3), textcoords="offset points")
ax.axvline(gt_hf, color="red", linestyle="--", linewidth=1, alpha=0.7, label=f"GT HF={gt_hf:.3f}")
ax.set_xlabel("HF_unobs (closer to GT is better)", fontsize=9)
ax.set_ylabel("PSNR_all (higher is better)", fontsize=9)
ax.set_title("PSNR vs HF — Pareto frontier", fontsize=10)
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# 2. PSNR vs |dHF| (closer to origin = better on both axes)
ax = axes[0, 1]
for i, r in enumerate(results):
    ax.scatter(r["dhf"], r["psnr"], c=colors[i], s=60, zorder=3, edgecolors="k", linewidths=0.3)
    ax.annotate(r["name"], (r["dhf"], r["psnr"]), fontsize=4, ha="center", va="bottom",
                xytext=(0, 3), textcoords="offset points")
ax.set_xlabel("|ΔHF| (lower is better — closer to GT)", fontsize=9)
ax.set_ylabel("PSNR_all (higher is better)", fontsize=9)
ax.set_title("PSNR vs HF gap — top-right corner is ideal", fontsize=10)
ax.grid(True, alpha=0.3)
# Mark the ideal quadrant
ax.axvspan(0, 0.02, color="green", alpha=0.05)

# 3. Bar chart: PSNR by config
ax = axes[1, 0]
x = np.arange(len(names))
bars = ax.barh(x, psnrs, color=colors, edgecolor="k", linewidth=0.3)
ax.set_yticks(x)
ax.set_yticklabels(names, fontsize=5)
ax.set_xlabel("PSNR_all (dB)", fontsize=9)
ax.set_title("PSNR by configuration", fontsize=10)
ax.invert_yaxis()
# Add value labels
for i, (p, d) in enumerate(zip(psnrs, dhfs)):
    if p > 0:
        ax.text(p + 0.1, i, f"{p:.1f} |Δ|={d:.3f}", fontsize=4, va="center")
ax.grid(True, axis="x", alpha=0.3)

# 4. SSIM vs HF scatter
ax = axes[1, 1]
for i, r in enumerate(results):
    ax.scatter(r["hf"], r["ssim"], c=colors[i], s=60, zorder=3, edgecolors="k", linewidths=0.3)
    ax.annotate(r["name"], (r["hf"], r["ssim"]), fontsize=4, ha="center", va="bottom",
                xytext=(0, 3), textcoords="offset points")
ax.axvline(gt_hf, color="red", linestyle="--", linewidth=1, alpha=0.7, label=f"GT HF={gt_hf:.3f}")
ax.set_xlabel("HF_unobs", fontsize=9)
ax.set_ylabel("SSIM (higher is better)", fontsize=9)
ax.set_title("SSIM vs HF", fontsize=10)
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# Legend for groups
from matplotlib.patches import Patch
legend_elements = []
seen = set()
for n, c in zip(names, colors):
    g = n[0]
    if g not in seen:
        seen.add(g)
        legend_elements.append(Patch(facecolor=c, label=f"Group {g}"))
fig.legend(handles=legend_elements, loc="upper right", fontsize=7, ncol=2,
           bbox_to_anchor=(0.99, 0.99))

plt.suptitle(f"debug5 interim results ({len(results)} configs)  GT HF={gt_hf:.3f}", fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f"{OUT}/interim_analysis.png", dpi=200)
plt.close()
print(f"Saved {OUT}/interim_analysis.png ({len(results)} configs)")

# Print summary table
print(f"\n{'='*90}")
print(f"GT HF_unobs = {gt_hf:.3f}")
print(f"\n{'Name':<24} {'PSNR':>7} {'PSNR_u':>7} {'SSIM':>7} {'HF':>6} {'|dHF|':>6} {'Time':>5}")
print("-"*70)
for r in sorted(results, key=lambda x: x["psnr"], reverse=True):
    print(f"{r['name']:<24} {r['psnr']:>7.2f} {r['psnr_u']:>7.2f} {r['ssim']:>7.4f} "
          f"{r['hf']:>6.3f} {r['dhf']:>6.3f} {r['time']:>5d}")

# Pareto analysis
print(f"\n--- Pareto optimal (PSNR vs |dHF|) ---")
pareto = []
for r in sorted(results, key=lambda x: x["psnr"], reverse=True):
    if not pareto or r["dhf"] < min(p["dhf"] for p in pareto):
        pareto.append(r)
        print(f"  {r['name']:<24} PSNR={r['psnr']:.2f}  |dHF|={r['dhf']:.3f}  SSIM={r['ssim']:.4f}")
