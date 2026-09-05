#!/usr/bin/env python
"""Acceptance test: the clean project must reproduce the reference run of the finalized Algorithm 4
(PixelFlowICLR/Algorithm2 `python main4.py`, 2026-09-05 -> tests/reference_final.csv: 5 tasks x 7 images, seed 42).

    PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=0 python tests/acceptance.py [--tasks ...] [--images ...] [--tol 2e-4]

Runs run.py into results/acceptance and compares full MSE (post_mse) and hole MSE (post_hole) per cell.
Identical code paths on the same GPU agree to ~1e-5 in MSE (GPU run-to-run noise); the default tolerance 2e-4
absolute is 20x that and still ~100x below any modelling effect."""
import argparse, csv, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
ap = argparse.ArgumentParser(); ap.add_argument("--tasks", nargs="*"); ap.add_argument("--images", nargs="*")
ap.add_argument("--tol", type=float, default=2e-4); ap.add_argument("--out", default=os.path.join(ROOT, "results", "acceptance"))
ap.add_argument("--no-run", action="store_true", help="only compare an existing results/acceptance/final.csv"); a = ap.parse_args()
if not a.no_run:
    cmd = [sys.executable, os.path.join(ROOT, "run.py"), "--out", a.out, "--no-metrics"]
    if a.tasks: cmd += ["--tasks", *a.tasks]
    if a.images: cmd += ["--images", *a.images]
    subprocess.run(cmd, check=True, env={**os.environ, "PYTHONHASHSEED": "0"})
ref = {(r["task"], r["image"]): r for r in csv.DictReader(open(os.path.join(HERE, "reference_final.csv")))}
got = {(r["task"], r["image"]): r for r in csv.DictReader(open(os.path.join(a.out, "final.csv")))}
bad, n = [], 0
for key, g in got.items():
    if key not in ref: continue
    r = ref[key]; n += 1
    d_full = abs(float(g["mse_full"]) - float(r["post_mse"]))
    d_hole = abs(float(g["mse_hole"]) - float(r["post_hole"])) if r.get("post_hole") not in (None, "", "nan") and g["mse_hole"] != "nan" else 0.0
    flag = "OK " if max(d_full, d_hole) <= a.tol else "BAD"
    if flag == "BAD": bad.append(key)
    print(f"{flag} {key[0]:18s} {key[1]:18s} full {float(g['mse_full']):.5f} vs {float(r['post_mse']):.5f} (|d|={d_full:.1e})"
          + (f"  hole {float(g['mse_hole']):.5f} vs {float(r['post_hole']):.5f} (|d|={d_hole:.1e})" if d_hole else ""))
print(f"\n{n - len(bad)}/{n} cells within {a.tol:g}; " + ("ACCEPTANCE PASSED" if not bad else f"FAILED: {bad}"))
sys.exit(1 if bad else 0)
