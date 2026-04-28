"""Aggregate ALL three sweeps:
  larger_box/results/    (32 configs, h_eps × L)
  exploration/runs/      (45 configs, multi-axis combos)
  exploration/runs_p/    (78 configs, deep single-axis + CFG combos)

Outputs:
  exploration/summaries/full_summary.json
  exploration/ranked_results.csv          (overwrites)
  exploration/ranked_results.md           (overwrites, top-30 + group view)
  exploration/comparisons/curves_<param>.png  (per-axis LPIPS/PSNR/|dHF| curves)
"""
import os, sys, json, glob, csv
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))   # exploration/scripts
EXP  = os.path.dirname(HERE)
LB   = os.path.dirname(EXP)
DBG  = os.path.dirname(LB)
REPO = os.path.dirname(DBG)
sys.path.insert(0, HERE)
sys.path.insert(0, LB)
sys.path.insert(0, DBG)
sys.path.insert(0, REPO)
os.chdir(REPO)

SUMS = os.path.join(EXP, "summaries")
COMP = os.path.join(EXP, "comparisons")
os.makedirs(SUMS, exist_ok=True)
os.makedirs(COMP, exist_ok=True)


def gather(dirpath, source_tag):
    rows = []
    for jp in sorted(glob.glob(os.path.join(dirpath, "*.json"))):
        b = os.path.basename(jp)
        if b.startswith("_") or "summary" in b:
            continue
        try:
            m = json.load(open(jp))
        except Exception:
            continue
        if "name" not in m or "psnr" not in m:
            continue
        rows.append(dict(
            name=m["name"], kw=m.get("kw", {}),
            psnr=m["psnr"], ssim=m["ssim"], lpips=m["lpips"],
            psnr_unobs=m.get("psnr_unobs", float("nan")),
            hf=m.get("hf", float("nan")), dhf=m.get("dhf", float("nan")),
            time=m.get("time", float("nan")),
            source=source_tag, json_path=jp,
            pt_path=jp[:-5] + ".pt", png_path=jp[:-5] + ".png",
        ))
    return rows


def balanced_score(r):
    lpips = r["lpips"]; psnr = r["psnr"]; dhf = r.get("dhf", 0.0)
    psnr_pen = max(0.0, 17.0 - psnr) * 0.02
    dhf_pen  = max(0.0, dhf - 0.05) * 0.5
    return lpips + psnr_pen + dhf_pen


def main():
    rows  = gather(os.path.join(LB,  "results"),  "larger_box")
    rows += gather(os.path.join(EXP, "runs"),     "exploration_v1")
    rows += gather(os.path.join(EXP, "runs_p"),   "exploration_v2_deep")
    rows += gather(os.path.join(EXP, "runs_f"),   "exploration_v3_final")
    print(f"gathered {len(rows)} rows")
    if not rows: return

    for r in rows: r["balanced"] = balanced_score(r)
    rows_b = sorted(rows, key=lambda r: r["balanced"])
    rows_l = sorted(rows, key=lambda r: r["lpips"])
    rows_p = sorted(rows, key=lambda r: -r["psnr"])

    summary = dict(
        total=len(rows),
        winners_by_balanced=[r["name"] for r in rows_b[:15]],
        winners_by_lpips=[r["name"] for r in rows_l[:15]],
        winners_by_psnr=[r["name"] for r in rows_p[:15]],
        all_rows=rows_b,
    )
    with open(os.path.join(SUMS, "full_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # CSV
    csv_path = os.path.join(EXP, "ranked_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "name", "source", "psnr", "psnr_unobs", "ssim", "lpips",
                    "hf", "dhf", "balanced", "time_sec", "kw"])
        for i, r in enumerate(rows_b):
            w.writerow([i + 1, r["name"], r["source"],
                        f"{r['psnr']:.3f}", f"{r['psnr_unobs']:.3f}",
                        f"{r['ssim']:.4f}", f"{r['lpips']:.4f}",
                        f"{r['hf']:.4f}", f"{r['dhf']:.4f}",
                        f"{r['balanced']:.4f}", f"{r['time']:.0f}",
                        json.dumps(r["kw"], default=str)])
    print(f"wrote {csv_path}")

    # Top-30 markdown
    md = [f"# Combined ranked results — {len(rows)}-config corpus\n",
          f"`larger_box` (32) + `exploration/runs` (45) + `exploration/runs_p` (78) + `exploration/runs_f` (12) = **{len(rows)}** configs.",
          "Balanced = LPIPS + 0.02·max(0, 17−PSNR) + 0.5·max(0, |dHF|−0.05). Lower=better.\n",
          "## Top-30 by balanced score\n",
          "| Rank | Config | Source | PSNR | PSNRu | SSIM | LPIPS | |dHF| | Balanced |",
          "|------|--------|--------|------|-------|------|-------|-------|----------|"]
    for i, r in enumerate(rows_b[:30]):
        md.append(f"| {i+1} | `{r['name']}` | {r['source']} | "
                  f"{r['psnr']:.2f} | {r['psnr_unobs']:.2f} | {r['ssim']:.3f} | "
                  f"**{r['lpips']:.4f}** | {r['dhf']:.3f} | **{r['balanced']:.4f}** |")
    md.append("\n## Top-15 by LPIPS\n")
    md.append("| Rank | Config | Source | PSNR | LPIPS | |dHF| |")
    md.append("|------|--------|--------|------|-------|-------|")
    for i, r in enumerate(rows_l[:15]):
        md.append(f"| {i+1} | `{r['name']}` | {r['source']} | {r['psnr']:.2f} | **{r['lpips']:.4f}** | {r['dhf']:.3f} |")
    md.append("\n## Top-15 by PSNR\n")
    md.append("| Rank | Config | Source | PSNR | LPIPS | |dHF| |")
    md.append("|------|--------|--------|------|-------|-------|")
    for i, r in enumerate(rows_p[:15]):
        md.append(f"| {i+1} | `{r['name']}` | {r['source']} | **{r['psnr']:.2f}** | {r['lpips']:.4f} | {r['dhf']:.3f} |")

    md.append("\n## Same-batch F sweep (clean co-tenancy)\n")
    f_rows = sorted([r for r in rows if r["source"] == "exploration_v3_final"],
                    key=lambda r: r["balanced"])
    md.append("These 12 configs ran in one batch → metric deltas reflect real param effects, not co-tenancy noise.\n")
    md.append("| Config | PSNR | LPIPS | |dHF| | Balanced |")
    md.append("|--------|------|-------|-------|----------|")
    for r in f_rows:
        md.append(f"| `{r['name']}` | {r['psnr']:.2f} | **{r['lpips']:.4f}** | {r['dhf']:.3f} | **{r['balanced']:.4f}** |")

    md.append(f"\n## All {len(rows)} configs sorted by balanced score\n")
    md.append("| Config | Source | PSNR | LPIPS | |dHF| | Balanced |")
    md.append("|--------|--------|------|-------|-------|----------|")
    for r in rows_b:
        md.append(f"| `{r['name']}` | {r['source']} | {r['psnr']:.2f} | {r['lpips']:.4f} | {r['dhf']:.3f} | {r['balanced']:.4f} |")

    md_path = os.path.join(EXP, "ranked_results.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    print(f"wrote {md_path}")

    # ============================================================
    # Per-axis curves — extract single-axis sweeps from runs_p
    # Each curve plots LPIPS / PSNR / |dHF| vs the swept param value.
    # ============================================================
    def by_name(prefix):
        return [r for r in rows if r["name"].startswith(prefix)]

    def get_param(r, key, default=None):
        v = r["kw"].get(key, default)
        return v

    def plot_curve(rows_in, x_key, x_label, fname, *, anchor_value=None, log_x=True):
        xs, lps, ps, dhfs, names = [], [], [], [], []
        for r in rows_in:
            v = get_param(r, x_key)
            if v is None: continue
            if isinstance(v, (list, tuple)): continue
            xs.append(float(v)); lps.append(r["lpips"]); ps.append(r["psnr"])
            dhfs.append(r["dhf"]); names.append(r["name"])
        if len(xs) < 2: return
        order = np.argsort(xs)
        xs = np.array(xs)[order]; lps = np.array(lps)[order]
        ps = np.array(ps)[order]; dhfs = np.array(dhfs)[order]
        names = [names[i] for i in order]

        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        for a in ax:
            if log_x:
                try: a.set_xscale("log")
                except Exception: pass
            a.grid(True, alpha=0.3)
        ax[0].plot(xs, lps, "o-", color="C0"); ax[0].set_xlabel(x_label); ax[0].set_ylabel("LPIPS (lower better)")
        ax[1].plot(xs, ps,  "s-", color="C2"); ax[1].set_xlabel(x_label); ax[1].set_ylabel("PSNR (dB)")
        ax[2].plot(xs, dhfs,"^-", color="C3"); ax[2].set_xlabel(x_label); ax[2].set_ylabel("|dHF|")
        if anchor_value is not None:
            for a in ax:
                a.axvline(anchor_value, color="gray", linestyle="--", alpha=0.5, label=f"anchor={anchor_value}")
                a.legend(fontsize=7)
        for i, n in enumerate(names):
            ax[0].annotate(n.replace(prefix_strip, ""), (xs[i], lps[i]), fontsize=6, alpha=0.6)
        fig.suptitle(f"PRINCIPLE 128x128 box  •  axis = {x_label}", fontsize=11)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        path = os.path.join(COMP, fname)
        plt.savefig(path, dpi=140); plt.close()
        print(f"wrote {path}")

    prefix_strip = ""

    # h_epsilon (combine A_he* from larger_box + P_he_* + Q_*_he* + N_*)
    he_rows = []
    for r in rows:
        n = r["name"]
        v = r["kw"].get("h_epsilon")
        if v is None or isinstance(v, (list, tuple)): continue
        # Match: scalar h_eps, no CFG (group A and P_he), but anchor matters
        if n.startswith("A_he") or n.startswith("P_he"):
            he_rows.append(r)
        elif n in ("C_L5", "C_L10", "R_anchor_C_L5"):  # anchors
            he_rows.append(r)
    plot_curve(he_rows, "h_epsilon", "h_epsilon (log)", "curve_h_epsilon.png", anchor_value=1e-3, log_x=True)

    # num_langevin (C_L* + P_L*)
    L_rows = [r for r in rows if (r["name"].startswith("C_L") or r["name"].startswith("P_L"))
              and not isinstance(r["kw"].get("num_langevin"), (list, tuple))]
    plot_curve(L_rows, "num_langevin", "num_langevin (log)", "curve_num_langevin.png", anchor_value=5, log_x=True)

    # lambda_reg (F_lr* + P_lreg*)
    lreg_rows = [r for r in rows if (r["name"].startswith("F_lr") or r["name"].startswith("P_lreg"))]
    # Add anchors
    for r in rows:
        if r["name"] in ("C_L5", "R_anchor_N_cfg2"):
            lreg_rows.append(r)
    plot_curve(lreg_rows, "lambda_reg", "lambda_reg (log)", "curve_lambda_reg.png", anchor_value=50, log_x=True)

    # guidance_scale (P_cfg* + N_cfg2 + N_cfg2_n01 + R_anchor_N_cfg2)
    cfg_rows = []
    for r in rows:
        n = r["name"]
        v = r["kw"].get("guidance_scale", 0.0)
        if isinstance(v, (list, tuple)): continue
        if n.startswith("P_cfg") or n in ("N_cfg2", "N_cfg2_n01", "R_anchor_N_cfg2"):
            cfg_rows.append(r)
        # baseline = anchor C_L5 has cfg=0
        elif n in ("C_L5", "R_anchor_C_L5"):
            cfg_rows.append(r)
    # We'll plot vs guidance_scale, treating missing as 0
    for r in cfg_rows:
        if "guidance_scale" not in r["kw"]: r["kw"]["guidance_scale"] = 0.0
    plot_curve(cfg_rows, "guidance_scale", "guidance_scale (CFG)", "curve_guidance_scale.png", anchor_value=2.0, log_x=False)

    # h_x stage-3 magnitude (extract last element when h_x is list with first 3 = 0.1)
    hx_rows = []
    for r in rows:
        n = r["name"]
        v = r["kw"].get("h_x")
        if isinstance(v, list) and len(v) == 4 and v[0] == 0.1 and v[1] == 0.1 and v[2] == 0.1:
            r2 = dict(r); r2["kw"] = dict(r["kw"]); r2["kw"]["hx_s3"] = v[3]
            hx_rows.append(r2)
    plot_curve(hx_rows, "hx_s3", "h_x stage-3 magnitude", "curve_h_x_stage3.png", anchor_value=0.7, log_x=False)

    # h_x uniform (P_hxu_* + G_hx_uni3)
    hxu_rows = []
    for r in rows:
        n = r["name"]
        v = r["kw"].get("h_x")
        if isinstance(v, (int, float)):
            r2 = dict(r); r2["kw"] = dict(r["kw"]); r2["kw"]["hx_uni"] = float(v)
            hxu_rows.append(r2)
    plot_curve(hxu_rows, "hx_uni", "h_x uniform value", "curve_h_x_uniform.png", anchor_value=0.1, log_x=True)

    # lambda_prox (H_lprox* + P_lp_*)
    lp_rows = []
    for r in rows:
        v = r["kw"].get("lambda_prox", 0.0)
        if isinstance(v, (list, tuple)): continue
        if r["name"].startswith("H_lprox") or r["name"].startswith("P_lp_"):
            lp_rows.append(r)
        elif r["name"] in ("R_anchor_N_cfg2", "N_cfg2"):
            r2 = dict(r); r2["kw"] = dict(r["kw"]); r2["kw"]["lambda_prox"] = 0.0
            lp_rows.append(r2)
    plot_curve(lp_rows, "lambda_prox", "lambda_prox (log)", "curve_lambda_prox.png", anchor_value=0.0, log_x=False)

    # lambda_x (P_lx_*)
    lx_rows = []
    for r in rows:
        v = r["kw"].get("lambda_x", 0.01)
        if isinstance(v, (list, tuple)): continue
        if r["name"].startswith("P_lx_"):
            lx_rows.append(r)
        elif r["name"] in ("R_anchor_N_cfg2",):
            r2 = dict(r); r2["kw"] = dict(r["kw"]); r2["kw"]["lambda_x"] = 0.01
            lx_rows.append(r2)
    plot_curve(lx_rows, "lambda_x", "lambda_x (log)", "curve_lambda_x.png", anchor_value=0.01, log_x=True)

    print("\n=== TOP-10 by balanced ===")
    for i, r in enumerate(rows_b[:10]):
        src = r["source"][:6]
        print(f"  #{i+1:2d}  {src}  {r['name']:<22s}  P={r['psnr']:5.2f}  L={r['lpips']:.4f}  dHF={r['dhf']:5.3f}  bal={r['balanced']:.4f}")

    return rows_b


if __name__ == "__main__":
    main()
