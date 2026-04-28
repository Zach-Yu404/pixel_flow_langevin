"""Aggregate folder_woNoiseLast + cross-reference sweep2 baselines for sigma_ref_sq study."""
import os, glob, json
HERE = os.path.dirname(os.path.abspath(__file__))
LB   = os.path.dirname(HERE)
SW2  = os.path.join(LB, "noise_uni3", "sweep2", "runs")
NU   = os.path.join(LB, "noise_uni3")
RUN  = os.path.join(HERE, "runs")


def load(path, tag=""):
    if not os.path.exists(path): return None
    d = json.load(open(path))
    return dict(name=tag or d["name"], kw=d["kw"], psnr=d["psnr"], ssim=d["ssim"],
                lpips=d["lpips"], psnr_unobs=d["psnr_unobs"],
                dhf=d["dhf"], hf=d["hf"], time=d.get("time", 0))


rows = []

# folder_woNoiseLast (the new sigma_ref sweep)
for f in sorted(glob.glob(os.path.join(RUN, "*.json"))):
    rows.append(load(f))

# sweep2 default sigma_ref_sq=0.01 references — same grid minus the SRS axis
# Pull tr=0 only (matched to woNoiseLast's terminal_replace=0)
for f in sorted(glob.glob(os.path.join(SW2, "*_tr0.json"))):
    r = load(f, tag=os.path.basename(f).replace(".json", "") + " (sw2 srs=1e-2)")
    if r: rows.append(r)

# Baseline anchors
b = load(os.path.join(NU, "noise_uni3.json"), tag="(BASE) noise=1 default")
if b: rows.append(b)

print("\n# folder_woNoiseLast — sorted by LPIPS\n")
print("| # | name | h_eps | h_x | L | sigma_ref_sq | PSNR | SSIM | LPIPS | |dHF| |")
print("|---|------|-------|-----|---|--------------|------|------|-------|-------|")
for i, r in enumerate(sorted(rows, key=lambda r: r["lpips"])):
    kw = r["kw"]
    print(f"| {i+1} | `{r['name']}` | {kw.get('h_epsilon','-')} | {kw.get('h_x','-')} | "
          f"{kw.get('num_langevin','-')} | {kw.get('sigma_ref_sq','1e-2 (sw2 default)')} | "
          f"{r['psnr']:.2f} | {r['ssim']:.4f} | **{r['lpips']:.4f}** | {r['dhf']:.3f} |")

print("\n# folder_woNoiseLast — sorted by PSNR\n")
print("| # | name | sigma_ref_sq | PSNR | LPIPS | SSIM | |dHF| |")
print("|---|------|--------------|------|-------|------|-------|")
for i, r in enumerate(sorted(rows, key=lambda r: -r["psnr"])):
    kw = r["kw"]
    print(f"| {i+1} | `{r['name']}` | {kw.get('sigma_ref_sq','1e-2')} | "
          f"**{r['psnr']:.2f}** | {r['lpips']:.4f} | {r['ssim']:.4f} | {r['dhf']:.3f} |")

# Trend: matched-pairs across sigma_ref_sq
print("\n# Matched-pair effect of sigma_ref_sq (1e-2 sw2 baseline → 1e-3 → 1e-4)\n")
print("| (h_eps, h_x, L) | srs=1e-2 (sw2) | srs=1e-3 | srs=1e-4 |")
print("|-----------------|----------------|----------|----------|")
def find(rows, **kw):
    for r in rows:
        if all(r["kw"].get(k) == v for k, v in kw.items()): return r
    return None

for he in (0.01, 0.001):
    for hx in (0.1, 0.2):
        for L in (5, 10, 15):
            r2  = find(rows, h_epsilon=he, h_x=hx, num_langevin=L,
                       terminal_replace_weight=0.0)  # sigma_ref_sq absent => sw2 default
            r2  = next((r for r in rows if r["kw"].get("h_epsilon") == he
                        and r["kw"].get("h_x") == hx
                        and r["kw"].get("num_langevin") == L
                        and r["kw"].get("terminal_replace_weight") == 0.0
                        and "sigma_ref_sq" not in r["kw"]), None)
            r3 = find(rows, h_epsilon=he, h_x=hx, num_langevin=L, sigma_ref_sq=1e-3)
            r4 = find(rows, h_epsilon=he, h_x=hx, num_langevin=L, sigma_ref_sq=1e-4)
            def fmt(r):
                if r is None: return "—"
                return f"P={r['psnr']:.2f} L={r['lpips']:.4f} D={r['dhf']:.3f}"
            print(f"| ({he}, {hx}, L={L}) | {fmt(r2)} | {fmt(r3)} | {fmt(r4)} |")

# Save machine-readable
out = dict(all=rows)
with open(os.path.join(HERE, "summary.json"), "w") as f:
    json.dump(out, f, indent=2, default=str)
print(f"\nWrote: {HERE}/summary.json   ({len(rows)} rows)")
