"""One-shot reorganization of exploration/ into 6 folders, then delete .pt.

Target layout:
  exploration/
    configs/        all per-config .json files
    best_configs/   top-10 by balanced score (copies of corresponding json)
    imgs/           all per-config result .png files
    best_images/    top-10 result PNGs + top_comparison_panel.png + per-axis curves
    analysis/       all .md / .csv / summaries (analysis & summary files)
    codes/          .py / .sh / .log files

After: delete all .pt files in exploration/ subtree.
"""
import os, sys, json, glob, shutil

HERE = os.path.dirname(os.path.abspath(__file__))   # exploration/scripts
EXP  = os.path.dirname(HERE)
os.chdir(EXP)

NEW_DIRS = ["configs", "best_configs", "imgs", "best_images", "analysis", "codes"]
for d in NEW_DIRS:
    os.makedirs(os.path.join(EXP, d), exist_ok=True)

# ---- Step 1: gather all per-config json+png from runs / runs_p / runs_f --------
SRC_RUN_DIRS = ["runs", "runs_p", "runs_f"]
moved_json = 0
moved_png  = 0
for sd in SRC_RUN_DIRS:
    p = os.path.join(EXP, sd)
    if not os.path.isdir(p): continue
    for f in os.listdir(p):
        full = os.path.join(p, f)
        if f.endswith(".json"):
            shutil.move(full, os.path.join(EXP, "configs", f))
            moved_json += 1
        elif f.endswith(".png"):
            shutil.move(full, os.path.join(EXP, "imgs", f))
            moved_png += 1

# best_results/ also has duplicates of certain configs (winners + references).
# Keep the WINNER_*/anchor copies BUT only as part of best_images/best_configs.
# Other .json/.png in best_results that are not duplicates: move to configs/imgs.
br = os.path.join(EXP, "best_results")
if os.path.isdir(br):
    for f in os.listdir(br):
        full = os.path.join(br, f)
        target_dir = None
        if f.endswith(".json"):
            target_dir = os.path.join(EXP, "configs")
        elif f.endswith(".png"):
            target_dir = os.path.join(EXP, "imgs")
        elif f.endswith(".pt"):
            os.remove(full); continue   # we delete .pt anyway
        if target_dir is None: continue
        target = os.path.join(target_dir, f)
        if os.path.exists(target):
            os.remove(full)             # duplicate, drop
        else:
            shutil.move(full, target)

# ---- Step 2: pick top-10 by balanced score (LPIPS + PSNR-floor + |dHF|-cap) -----
def balanced(m):
    return (m["lpips"] + 0.02 * max(0.0, 17.0 - m["psnr"])
            + 0.5 * max(0.0, m.get("dhf", 0) - 0.05))

candidates = []
for jp in glob.glob(os.path.join(EXP, "configs", "*.json")):
    name = os.path.basename(jp)[:-5]
    if name.startswith("_") or name.startswith("WINNER_"):
        continue
    try:
        m = json.load(open(jp))
    except Exception:
        continue
    if "psnr" not in m or "lpips" not in m: continue
    candidates.append((name, m, jp))

candidates.sort(key=lambda r: balanced(r[1]))
top10 = candidates[:10]
print(f"[top-10 by balanced score] (anchor floor PSNR ≥ 17, |dHF| cap 0.05)")
for i, (n, m, _) in enumerate(top10):
    print(f"  #{i+1:2d}  {n:<26s}  P={m['psnr']:5.2f}  L={m['lpips']:.4f}  "
          f"dHF={m.get('dhf', 0):5.3f}  bal={balanced(m):.4f}")

# Write a top-10 manifest into best_configs/
manifest = []
for i, (n, m, jp) in enumerate(top10):
    rank = i + 1
    # Copy json -> best_configs/{rank}_{name}.json
    dst_cfg = os.path.join(EXP, "best_configs", f"{rank:02d}_{n}.json")
    shutil.copy(jp, dst_cfg)
    # Copy corresponding png -> best_images/{rank}_{name}.png
    src_png = os.path.join(EXP, "imgs", f"{n}.png")
    if os.path.exists(src_png):
        shutil.copy(src_png, os.path.join(EXP, "best_images", f"{rank:02d}_{n}.png"))
    manifest.append({"rank": rank, "name": n,
                     "psnr": m["psnr"], "lpips": m["lpips"],
                     "ssim": m.get("ssim", 0), "dhf": m.get("dhf", 0),
                     "balanced": balanced(m), "kw": m.get("kw", {})})

with open(os.path.join(EXP, "best_configs", "TOP10_manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2, default=str)

# ---- Step 3: comparison + curve images -> best_images ---------------------------
src_panel = os.path.join(EXP, "top_comparison_panel.png")
if os.path.exists(src_panel):
    shutil.move(src_panel, os.path.join(EXP, "best_images", "top_comparison_panel.png"))

cmp_dir = os.path.join(EXP, "comparisons")
if os.path.isdir(cmp_dir):
    for f in os.listdir(cmp_dir):
        full = os.path.join(cmp_dir, f)
        if f.endswith(".png"):
            shutil.move(full, os.path.join(EXP, "best_images", f))

# Also: summaries/exploration_grid_top.png is a comparison-style image
sd = os.path.join(EXP, "summaries")
egt = os.path.join(sd, "exploration_grid_top.png")
if os.path.exists(egt):
    shutil.move(egt, os.path.join(EXP, "best_images", "exploration_grid_top.png"))

# ---- Step 4: analysis & summaries ----------------------------------------------
for f in ["final_analysis.md", "parameter_effects_summary.md", "psnr_audit.md",
          "ranked_results.md", "ranked_results.csv", "best_config.json"]:
    src = os.path.join(EXP, f)
    if os.path.exists(src):
        shutil.move(src, os.path.join(EXP, "analysis", f))

if os.path.isdir(sd):
    for f in os.listdir(sd):
        full = os.path.join(sd, f)
        if os.path.isfile(full):
            shutil.move(full, os.path.join(EXP, "analysis", f))

# ---- Step 5: codes -- .py / .sh / .log -----------------------------------------
sc = os.path.join(EXP, "scripts")
codes_dir = os.path.join(EXP, "codes")
if os.path.isdir(sc):
    for f in os.listdir(sc):
        full = os.path.join(sc, f)
        if os.path.isfile(full) and (f.endswith(".py") or f.endswith(".sh")):
            shutil.move(full, os.path.join(codes_dir, f))

# Logs were inside runs / runs_p / runs_f
for sd_ in SRC_RUN_DIRS:
    p = os.path.join(EXP, sd_)
    if not os.path.isdir(p): continue
    for f in os.listdir(p):
        full = os.path.join(p, f)
        if os.path.isfile(full) and f.endswith(".log"):
            shutil.move(full, os.path.join(codes_dir, f"{sd_}_{f.lstrip('_')}"))

# ---- Step 6: delete all .pt files ----------------------------------------------
pt_files = []
for root, _, files in os.walk(EXP):
    for f in files:
        if f.endswith(".pt"):
            pt_files.append(os.path.join(root, f))
for p in pt_files:
    os.remove(p)
print(f"deleted {len(pt_files)} .pt files")

# ---- Step 7: remove now-empty source dirs --------------------------------------
for d in ["runs", "runs_p", "runs_f", "best_results", "comparisons",
          "summaries", "scripts"]:
    p = os.path.join(EXP, d)
    if os.path.isdir(p):
        # remove __pycache__ inside scripts if any
        pyc = os.path.join(p, "__pycache__")
        if os.path.isdir(pyc): shutil.rmtree(pyc)
        try:
            os.rmdir(p)
            print(f"removed empty dir {d}")
        except OSError:
            print(f"[skip] {d} not empty: {os.listdir(p)}")

# ---- Final inventory -----------------------------------------------------------
print("\n=== FINAL LAYOUT ===")
for d in NEW_DIRS:
    p = os.path.join(EXP, d)
    n = len([f for f in os.listdir(p) if os.path.isfile(os.path.join(p, f))])
    print(f"  {d}/  {n} files")
print(f"\nmoved {moved_json} .json + {moved_png} .png + best_results contents")
