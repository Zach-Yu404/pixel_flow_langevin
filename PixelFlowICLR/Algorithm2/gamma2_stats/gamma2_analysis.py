"""Analysis of the merged val-set gamma^2 tables vs the 7-image reference: per-stage curves with across-class
spread, per-class dispersion, fp32-vs-TF32 class groups, and a markdown summary."""
import os, json, csv, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
G = "/CBIG-Standard-ECE/Zach/MSFlow/PixelFlowICLR/Algorithm2/gamma2_stats"; A2 = os.path.dirname(G)
allt = json.load(open(f"{G}/gamma2_all.json")); lab = json.load(open(f"{G}/gamma2_labelled.json"))
ref = json.load(open(f"{A2}/gamma2_meas_alg4.json"))["table"]
stages = sorted(allt["table"], key=int); taus = {k: list(allt["table"][k]) for k in stages}
cls = lab["classes"]; syns = sorted(cls)
md = [f"# val-set gamma^2 vs 7-image reference — {allt['meta']['n_images']} images / {allt['meta']['n_classes']} classes\n",
      f"precision: {allt['meta']['precision']['classes_by_precision']} (see meta)\n",
      "| stage | tau | gamma2 val (all) | std across images | ref (7 img) | ratio val/ref | per-class median | per-class 5%–95% |", "|---|---|---|---|---|---|---|---|"]
rows = []
for k in stages:
    for t in taus[k]:
        v = allt["table"][k][t]; sd = allt["std_across_images"][k][t]; r = ref[k].get(t, float("nan"))
        pc = np.array([cls[s]["table"][k][t] for s in syns]); q5, q50, q95 = np.percentile(pc, [5, 50, 95])
        md.append(f"| {k} | {t} | {v:.4f} | {sd:.4f} | {r:.4f} | {v/r:.2f} | {q50:.4f} | {q5:.4f}–{q95:.4f} |")
        rows.append(dict(stage=k, tau=t, val=v, std_img=sd, ref=r, ratio=v / r, pc_median=q50, pc_q5=q5, pc_q95=q95))
# per-class dispersion summary at stage 3, last tau
k3 = stages[-1]; tl = taus[k3][-1]; t0 = taus[k3][0]
pc_last = {s: cls[s]["table"][k3][tl] for s in syns}; pc_first = {s: cls[s]["table"][k3][t0] for s in syns}
top = sorted(pc_last.items(), key=lambda x: -x[1])[:5]; bot = sorted(pc_last.items(), key=lambda x: x[1])[:5]
md += ["\n## per-class dispersion (stage 3)",
       f"- tau={tl}: median {np.median(list(pc_last.values())):.4f}, CV {np.std(list(pc_last.values()))/np.mean(list(pc_last.values())):.2f}; highest {[(s, round(v,4)) for s,v in top]}; lowest {[(s, round(v,4)) for s,v in bot]}",
       f"- tau={t0}: median {np.median(list(pc_first.values())):.4f}, CV {np.std(list(pc_first.values()))/np.mean(list(pc_first.values())):.2f}"]
# fp32 vs tf32 groups (different classes — descriptive only)
grp = {}
for s in syns:
    grp.setdefault(cls[s].get("precision", "fp32"), []).append(s)
if len(grp) > 1:
    md.append("\n## precision groups (different classes, descriptive only)")
    for p_, ss in grp.items():
        m = np.mean([cls[s]["table"][k3][tl] for s in ss]); md.append(f"- {p_}: {len(ss)} classes, stage3 tau={tl} mean {m:.4f}")
open(f"{G}/analysis.md", "w").write("\n".join(md) + "\n")
with open(f"{G}/analysis_rows.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
# figure: per-stage gamma2 vs tau, val mean with 5-95% class band, ref dots
fig, axes = plt.subplots(1, len(stages), figsize=(4.2 * len(stages), 3.6))
for i, k in enumerate(stages):
    ax = axes[i]; x = [float(t) for t in taus[k]]
    pcs = np.array([[cls[s]["table"][k][t] for t in taus[k]] for s in syns])
    ax.fill_between(x, np.percentile(pcs, 5, 0), np.percentile(pcs, 95, 0), alpha=.25, label="classes 5–95%")
    ax.plot(x, [allt["table"][k][t] for t in taus[k]], "o-", ms=3, label="val all (50k)")
    ax.plot(x, [ref[k].get(t, np.nan) for t in taus[k]], "s--", ms=3, label="ref (7 img)")
    ax.set_yscale("log"); ax.set_title(f"stage {k} ({32*2**int(k)}px)"); ax.set_xlabel("tau"); ax.grid(alpha=.3)
    if i == 0: ax.set_ylabel("gamma^2"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{G}/gamma2_vs_tau.png", dpi=120); plt.close()
plt.figure(figsize=(6, 3.5)); plt.hist(list(pc_last.values()), bins=50); plt.xlabel(f"per-class gamma^2, stage 3 tau={tl}"); plt.ylabel("classes"); plt.tight_layout(); plt.savefig(f"{G}/perclass_hist_stage3_last.png", dpi=120)
print("\n".join(md[:8])); print("..."); print("\n".join(md[-6:]))
