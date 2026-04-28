"""Aggregate sweep2 + cross-reference baselines.

Baselines pulled in for context:
  baseline (G_hx_uni3 + noise=1):  ../noise_uni3.json
  S1_h01_L15:                       ../S1_h01_L15.json   (h_x=0.1, L=15, lreg=50, no lprox, cfg=0)
  S2..S6 (if present):              ../S{2..6}_*.json
"""
import os, glob, json
HERE = os.path.dirname(os.path.abspath(__file__))
NU   = os.path.dirname(HERE)
RUN  = os.path.join(HERE, "runs")


def load_one(path, tag=""):
    if not os.path.exists(path): return None
    d = json.load(open(path))
    return dict(name=tag or d["name"], kw=d["kw"], psnr=d["psnr"], ssim=d["ssim"],
                lpips=d["lpips"], psnr_unobs=d["psnr_unobs"],
                dhf=d["dhf"], hf=d["hf"], time=d.get("time", 0))


rows = []

# Baselines
b = load_one(os.path.join(NU, "noise_uni3.json"), tag="(BASE) G_hx_uni3+n=1")
if b: rows.append(b)
b = load_one(os.path.join(NU, "S1_h01_L15.json"), tag="(SW1) S1_h01_L15")
if b: rows.append(b)
for nm in ["S2_h005_L30", "S3_lprox05", "S4_lprox20", "S5_skip_s3", "S6_combo"]:
    b = load_one(os.path.join(NU, f"{nm}.json"), tag=f"(SW1) {nm}")
    if b: rows.append(b)

# sweep2 configs
for f in sorted(glob.glob(os.path.join(RUN, "*.json"))):
    rows.append(load_one(f))


def fmt_kw(kw):
    parts = []
    for k in ("h_epsilon", "h_x", "num_langevin",
              "lambda_reg", "lambda_prox",
              "noise_scale", "guidance_scale", "terminal_replace_weight"):
        if k in kw: parts.append(f"{k}={kw[k]}")
    return "  ".join(parts)


# Sort by LPIPS for primary view
rows_lpips = sorted(rows, key=lambda r: r["lpips"])

print("\n# sweep2 + baselines — sorted by LPIPS\n")
print("| # | name | PSNR | SSIM | LPIPS | PSNR_u | |dHF| | t(s) |")
print("|---|------|------|------|-------|--------|-------|------|")
for i, r in enumerate(rows_lpips):
    print(f"| {i+1} | `{r['name']}` | {r['psnr']:.2f} | {r['ssim']:.4f} | "
          f"**{r['lpips']:.4f}** | {r['psnr_unobs']:.2f} | {r['dhf']:.3f} | {r['time']:.0f} |")

print("\n# sweep2 + baselines — sorted by PSNR\n")
print("| # | name | PSNR | LPIPS | SSIM | |dHF| |")
print("|---|------|------|-------|------|-------|")
for i, r in enumerate(sorted(rows, key=lambda r: -r["psnr"])):
    print(f"| {i+1} | `{r['name']}` | **{r['psnr']:.2f}** | {r['lpips']:.4f} | {r['ssim']:.4f} | {r['dhf']:.3f} |")

print("\n# Pareto front (PSNR↑ + LPIPS↓)\n")
pareto = []
for r in rows:
    dominated = False
    for s in rows:
        if (s["psnr"] >= r["psnr"] and s["lpips"] <= r["lpips"]
                and (s["psnr"] > r["psnr"] or s["lpips"] < r["lpips"])):
            dominated = True; break
    if not dominated: pareto.append(r)
print("| name | PSNR | LPIPS | SSIM | |dHF| | kw |")
print("|------|------|-------|------|-------|----|")
for r in sorted(pareto, key=lambda r: -r["psnr"]):
    print(f"| `{r['name']}` | {r['psnr']:.2f} | {r['lpips']:.4f} | {r['ssim']:.4f} | {r['dhf']:.3f} | `{fmt_kw(r['kw'])}` |")


# Save machine-readable
out = dict(all=rows, sorted_by_lpips=[r["name"] for r in rows_lpips],
           pareto=[r["name"] for r in pareto])
with open(os.path.join(HERE, "sweep2_summary.json"), "w") as f:
    json.dump(out, f, indent=2, default=str)
print(f"\nWrote: {HERE}/sweep2_summary.json   ({len(rows)} rows)")
