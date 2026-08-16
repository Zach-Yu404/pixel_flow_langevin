#!/usr/bin/env python
"""Consolidate onestep_mse results (PixelFlowICLR experiment 1 cleanup).

Rationale: the 5 task configs produce BITWISE-IDENTICAL predictions (verified:
max rel diff of every mse value = 0 across tasks; one-step never touches the
operator/y and the relevant LPIPS_king kw coincide). The original per-task
output (5 x 28 curve pngs + 5 overview pngs + 5 raw_mse.json) is therefore
5-fold redundant. This script:

  1. renders ONE summary figure  mse_vs_t_summary.png
     (4 stage panels; per-image thin lines + bold mean; WLS vs Model, log-y)
  2. keeps ONE raw_mse.json at the results root (rows from box_inpainting)
     + task_configs.json recording each task's config_path/kw for provenance
  3. deletes the 5 per-task directories (curves/, overview_mean.png, raw_mse.json)
  4. writes README.md describing the final layout

Safe to re-run; refuses to delete if the identity check fails.
"""

import json
import os
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "onestep_mse")
TASKS = ["box_inpainting", "random_inpainting", "gaussian_blur",
         "motion_blur", "superresolution"]


def main():
    data = {}
    for t in TASKS:
        p = os.path.join(OUT, t, "raw_mse.json")
        if os.path.isfile(p):
            data[t] = json.load(open(p))
    root_raw = os.path.join(OUT, "raw_mse.json")
    if not data:
        assert os.path.isfile(root_raw), "nothing to consolidate and no root raw_mse.json"
        data["_root"] = json.load(open(root_raw))

    # ── identity check across whatever task copies still exist ──
    tasks_present = [t for t in TASKS if t in data]
    if len(tasks_present) > 1:
        ref = data[tasks_present[0]]["rows"]
        for t in tasks_present[1:]:
            rows = data[t]["rows"]
            assert len(rows) == len(ref), f"row count mismatch {t}"
            for a, b in zip(ref, rows):
                assert a["mse_wls"] == b["mse_wls"] and a["mse_model"] == b["mse_model"], \
                    f"tasks NOT identical at {a['image']} stage{a['stage']} step{a['step']} — aborting, nothing deleted"
        print(f"[check] {len(tasks_present)} tasks bitwise identical ✓")

    ref_task = tasks_present[0] if tasks_present else "_root"
    rows = data[ref_task]["rows"]
    images = sorted({r["image"] for r in rows})
    stages = sorted({r["stage"] for r in rows})

    # ── 1. single summary figure ──
    fig, axes = plt.subplots(1, len(stages), figsize=(4.0 * len(stages), 3.4))
    for si in stages:
        ax = axes[si]
        rs = [r for r in rows if r["stage"] == si]
        taus = sorted({r["tau"] for r in rs})
        for name in images:  # thin per-image lines
            rw = sorted((r for r in rs if r["image"] == name), key=lambda r: r["tau"])
            ax.semilogy([r["tau"] for r in rw], [r["mse_wls"] for r in rw],
                        color="tab:blue", alpha=0.25, lw=0.8)
            ax.semilogy([r["tau"] for r in rw], [r["mse_model"] for r in rw],
                        color="tab:orange", alpha=0.25, lw=0.8)
        mw = [np.mean([r["mse_wls"] for r in rs if r["tau"] == t]) for t in taus]
        mm = [np.mean([r["mse_model"] for r in rs if r["tau"] == t]) for t in taus]
        ax.semilogy(taus, mw, "o-", color="tab:blue", lw=2, label=r"WLS $\hat{x}_1$")
        ax.semilogy(taus, mm, "s-", color="tab:orange", lw=2, label=r"Model $\hat{x}_1$")
        res = rs[0]["resolution"]
        ax.set_title(f"stage {si} ({res}px)", fontsize=11)
        ax.set_xlabel(r"$t$ (within-stage)")
        ax.grid(alpha=0.3)
        if si == 0:
            ax.set_ylabel(r"MSE vs GT $x_1^k$")
            ax.legend(fontsize=9)
    fig.suptitle("One-step $\\hat{x}_1$ prediction MSE vs $t$ — LPIPS_king, 7 playground images "
                 "(bold = mean, thin = per image; identical for all 5 tasks — one-step never "
                 "touches the operator)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(OUT, "mse_vs_t_summary.png"), dpi=160)
    plt.close(fig)
    print("[fig] mse_vs_t_summary.png written")

    # ── 2. provenance + single raw copy ──
    if tasks_present:
        json.dump(
            {t: dict(config_path=data[t]["config_path"], kw=data[t]["kw"]) for t in tasks_present},
            open(os.path.join(OUT, "task_configs.json"), "w"), indent=1)
        ref = dict(data[ref_task])
        ref["note"] = ("rows are identical for all 5 task configs (verified bitwise); "
                       "per-task copies removed — see task_configs.json for each task's kw")
        json.dump(ref, open(root_raw, "w"), indent=1)
        print("[raw] raw_mse.json + task_configs.json written")

        # ── 3. prune per-task dirs ──
        for t in tasks_present:
            shutil.rmtree(os.path.join(OUT, t))
        print(f"[prune] removed {len(tasks_present)} per-task dirs")

    # ── 4. README ──
    with open(os.path.join(OUT, "README.md"), "w") as f:
        f.write("""# onestep_mse — PixelFlowICLR experiment 1 (+1b visuals)

- `all_mse.csv` — raw MSE, 1400 rows = 5 tasks x 7 images x 4 stages x 10 t
  (kept with the task column for completeness; the 5 tasks are bitwise identical —
  one-step prediction never touches the operator/y).
- `raw_mse.json` — one canonical copy of the per-row data (+ note); per-task kw in
  `task_configs.json`.
- `mse_vs_t_summary.png` — THE curve figure (4 stage panels, WLS vs Model,
  per-image thin lines + bold mean). Replaces the former 140 per-(image,stage)
  pngs + 5 overviews.
- `onestep_predictions.mp4` — per-t one-step predictions video: 40 frames =
  4 stages x 10 steps, each frame rows [GT x1 | x_t | WLS | Model] x 7 images,
  per-image MSE under the WLS/Model rows. Produced by onestep_visual.py
  (A100 job) + ffmpeg.

Regenerate: `sbatch ../run_onestep_mse.sbatch`, `sbatch ../run_onestep_visual.sbatch`,
then `python ../consolidate_results.py`; encode: see .research experiment record.
""")
    print("[done] consolidation complete")


if __name__ == "__main__":
    main()
