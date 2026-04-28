"""Aggregate the 36-config sweep into per-base 4x4 fd × tr matrices."""
import os, glob, json
HERE = os.path.dirname(os.path.abspath(__file__))
RUN  = os.path.join(HERE, "runs")


def load(path):
    return json.load(open(path))


rows = []
for f in sorted(glob.glob(os.path.join(RUN, "*.json"))):
    rows.append(load(f))


def parse_tag(name):
    # e.g. "balanced_perceptual_srs1e-4__fd0_tr1"
    base, var = name.split("__")
    fd = int(var.split("_")[0][2:])
    tr = int(var.split("_")[1][2:])
    return base, fd, tr


# Group by base
groups = {}
for r in rows:
    base, fd, tr = parse_tag(r["name"])
    groups.setdefault(base, {})[(fd, tr)] = r


# Header summary
print(f"# 36-config sweep: 9 bases × {{fd0/fd1}} × {{tr0/tr1}}\n")
print(f"Total runs: {len(rows)}/36")
print()

# Per-base 4-cell table (PSNR | LPIPS | |dHF|)
print("## Per-base matrix (rows = (fd,tr), columns = base)\n")

for level in ["LPIPS_king", "balanced_perceptual", "pareto_dual_king"]:
    print(f"\n### {level}\n")
    print("|        | fd=0,tr=0 | fd=0,tr=1 | fd=1,tr=0 | fd=1,tr=1 |")
    print("|--------|-----------|-----------|-----------|-----------|")
    for srs in ["1e-2", "1e-3", "1e-4"]:
        base = f"{level}_srs{srs}"
        if base not in groups:
            continue
        cells = []
        for fd, tr in [(0,0), (0,1), (1,0), (1,1)]:
            r = groups[base].get((fd, tr))
            if r is None:
                cells.append("—")
            else:
                cells.append(f"P={r['psnr']:.2f} L={r['lpips']:.4f} D={r['dhf']:.3f}")
        print(f"| **srs={srs}** | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")

# Effect tables
print("\n## fd=0 → fd=1 delta (matched on tr) — across all 9 bases\n")
print("| base | tr | ΔPSNR | ΔLPIPS | ΔSSIM | Δ\\|dHF\\| |")
print("|------|----|-------|--------|-------|-----------|")
fd_deltas = []
for base, by_var in sorted(groups.items()):
    for tr in (0, 1):
        a = by_var.get((0, tr)); b = by_var.get((1, tr))
        if not a or not b: continue
        dP = b["psnr"]  - a["psnr"]
        dL = b["lpips"] - a["lpips"]
        dS = b["ssim"]  - a["ssim"]
        dD = b["dhf"]   - a["dhf"]
        fd_deltas.append(dict(base=base, tr=tr, dP=dP, dL=dL, dS=dS, dD=dD))
        print(f"| `{base}` | {tr} | {dP:+.3f} | {dL:+.4f} | {dS:+.4f} | {dD:+.3f} |")

if fd_deltas:
    import numpy as np
    avg = lambda k: np.mean([d[k] for d in fd_deltas])
    print(f"| **mean** |  | **{avg('dP'):+.3f}** | **{avg('dL'):+.4f}** | **{avg('dS'):+.4f}** | **{avg('dD'):+.3f}** |")

print("\n## tr=0 → tr=1 delta (matched on fd) — across all 9 bases\n")
print("| base | fd | ΔPSNR | ΔLPIPS | ΔSSIM | Δ\\|dHF\\| |")
print("|------|----|-------|--------|-------|-----------|")
tr_deltas = []
for base, by_var in sorted(groups.items()):
    for fd in (0, 1):
        a = by_var.get((fd, 0)); b = by_var.get((fd, 1))
        if not a or not b: continue
        dP = b["psnr"]  - a["psnr"]
        dL = b["lpips"] - a["lpips"]
        dS = b["ssim"]  - a["ssim"]
        dD = b["dhf"]   - a["dhf"]
        tr_deltas.append(dict(base=base, fd=fd, dP=dP, dL=dL, dS=dS, dD=dD))
        print(f"| `{base}` | {fd} | {dP:+.3f} | {dL:+.4f} | {dS:+.4f} | {dD:+.3f} |")

if tr_deltas:
    import numpy as np
    avg = lambda k: np.mean([d[k] for d in tr_deltas])
    print(f"| **mean** |  | **{avg('dP'):+.3f}** | **{avg('dL'):+.4f}** | **{avg('dS'):+.4f}** | **{avg('dD'):+.3f}** |")

# Top-5 absolute winners by metric
print("\n## Top-5 across all 36 by LPIPS\n")
print("| name | PSNR | SSIM | LPIPS | |dHF| |")
print("|------|------|------|-------|-------|")
for r in sorted(rows, key=lambda r: r["lpips"])[:5]:
    print(f"| `{r['name']}` | {r['psnr']:.2f} | {r['ssim']:.4f} | **{r['lpips']:.4f}** | {r['dhf']:.3f} |")

print("\n## Top-5 across all 36 by PSNR\n")
print("| name | PSNR | LPIPS | SSIM | |dHF| |")
print("|------|------|-------|------|-------|")
for r in sorted(rows, key=lambda r: -r["psnr"])[:5]:
    print(f"| `{r['name']}` | **{r['psnr']:.2f}** | {r['lpips']:.4f} | {r['ssim']:.4f} | {r['dhf']:.3f} |")

# Save machine-readable
out = dict(rows=rows, n=len(rows))
with open(os.path.join(HERE, "summary.json"), "w") as f:
    json.dump(out, f, indent=2, default=str)
print(f"\nWrote: {HERE}/summary.json   ({len(rows)} rows)")
