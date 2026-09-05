#!/usr/bin/env python
"""Algorithm 4 — run the finalized sampler on the configured tasks x images.

    PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=0 python run.py [--config config.json] [--tasks ...] [--images ...]
                                                          [--seed 42] [--out results/run] [--save-png]

Writes <out>/final.csv (one row per task x image: full MSE, hole/observed MSE for inpainting, PSNR/SSIM/LPIPS on
[0,1], measurement residual, NFE), <out>/metrics.csv (per stage x step rows), <out>/nfe.json, and optionally the
reconstructions as PNG. Cells already present in final.csv are skipped (resume); remove the file to recompute.
PYTHONHASHSEED=0 is required (random-inpainting masks are seeded with hash()).
"""
import argparse
import csv
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np                                                            # noqa: E402
import torch                                                                  # noqa: E402

from alg4.data import load_demo_images                                        # noqa: E402
from alg4.model import load_model                                             # noqa: E402
from alg4.operators import build_measurement                                  # noqa: E402
from alg4.prior import ClassSpectralPrior                                     # noqa: E402
from alg4.sampler import run_algorithm4, count_nfe_hook, NFE                  # noqa: E402
from alg4.ops import mse_masked                                               # noqa: E402

FINAL_FIELDS = ["task", "image", "seed", "mse_full", "mse_hole", "mse_obs", "psnr", "ssim", "lpips_alex",
                "lpips_piq", "meas_resid", "nfe", "blk1_cg_bad", "secs"]
ROW_FIELDS = ["task", "image", "seed", "frame", "stage", "step", "tau", "sigma_tau", "s2", "gamma2", "mse_x1",
              "mse_hole", "mse_obs", "meas_resid", "x0_rms", "blk1_cg_iters", "blk1_cg_resid", "blk1_cg_converged"]


def resolve(p):
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(HERE, p))


def check_hash_seed():
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("PYTHONHASHSEED=0 is required: random-inpainting masks are seeded with hash(short_name). "
                           "Re-run as: PYTHONHASHSEED=0 python run.py ...")


def done_cells(path):
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as f:
        return {(r["task"], r["image"], int(r["seed"])) for r in csv.DictReader(f)}


def append_rows(path, fields, rows):
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


def save_png(x, path):
    from PIL import Image
    arr = ((x[0].clamp(-1, 1) + 1) / 2 * 255).round().byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(arr).save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--images", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=None, help="override algorithm.seed")
    ap.add_argument("--out", default=None, help="override config 'out' (relative to this directory)")
    ap.add_argument("--save-png", action="store_true")
    ap.add_argument("--no-metrics", action="store_true", help="skip PSNR/SSIM/LPIPS (MSE columns only)")
    args = ap.parse_args()
    check_hash_seed()

    cfg = json.load(open(args.config))
    paths = {k: resolve(v) for k, v in cfg["paths"].items()}
    ALG, KW, TS = cfg["algorithm"], cfg["sampler_kw"], cfg["tasks_setup"]
    tasks = args.tasks or cfg["tasks"]
    images = args.images or cfg["images"]
    seed = int(args.seed if args.seed is not None else ALG["seed"])
    out = resolve(args.out or cfg["out"])
    os.makedirs(out, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    demos = {d["short_name"]: d for d in load_demo_images(paths["demo_dir"])}
    config, model = load_model(paths["model_dir"], device)
    model.register_forward_pre_hook(count_nfe_hook)
    K = int(config.scheduler.num_stages)
    gamma2_tab = json.load(open(paths["gamma2_table"]))["table"]
    prior = ClassSpectralPrior(paths["s_stats_npz"], paths["synset_map"], K)
    print(f"[setup] device={device} model={paths['model_dir']} gamma2={paths['gamma2_table']} "
          f"S=per-class spectral ({paths['s_stats_npz']}) S_it={KW['num_langevin']} seed={seed}", flush=True)

    final_path, rows_path = os.path.join(out, "final.csv"), os.path.join(out, "metrics.csv")
    done = done_cells(final_path)
    if not args.no_metrics:
        from alg4 import metrics as MET
    nfe_stats = {}
    t_all = time.time()
    for task in tasks:
        spec = TS[task]
        for name in images:
            if (task, name, seed) in done:
                print(f"[skip] {task}/{name} seed {seed} already in {final_path}", flush=True)
                continue
            d = demos[name]
            M = build_measurement(task, spec["operator"] | {"measurement_mode": spec.get("measurement_mode", "measure")},
                                  d, spec["sigma_n"], 256, device, ALG["measurement_seed"])
            gt = d["gt"].unsqueeze(0).to(device)
            s2_fn = prior.bind(d["class_idx"])
            print(f"[cell] {task}/{name}: {s2_fn.describe()}", flush=True)
            NFE["n"] = 0
            t0 = time.time()
            x1, rows, _ = run_algorithm4(
                model, config, gt, M["y"], M["op"], float(spec["sigma_n"]), device,
                make_Ak_fns=M["make_Ak_fns"], s2_fn=s2_fn, gamma2_tab=gamma2_tab,
                num_langevin=KW["num_langevin"], ode_steps_per_stage=KW["ode_steps_per_stage"], shift=KW["shift"],
                guidance_scale=KW["guidance_scale"], class_label=int(d["class_idx"]),
                cg_tol=KW["cg_tol"], cg_max_iter=KW["cg_max_iter"], cg_max_iter_endpoint=ALG["cg_max_iter_endpoint"],
                sigma_min=ALG["sigma_min"], seed=seed, hole_mask=M["hole"],
                terminal_replace_weight=float(spec.get("terminal_replace_weight", 0.0)))
            secs = time.time() - t0
            fin = dict(task=task, image=name, seed=seed, mse_full=float(((x1 - gt) ** 2).mean()),
                       mse_hole=mse_masked(x1, gt, M["hole"]) if M["hole"] is not None else float("nan"),
                       mse_obs=mse_masked(x1, gt, 1.0 - M["hole"]) if M["hole"] is not None else float("nan"),
                       meas_resid=rows[-1]["meas_resid"], nfe=NFE["n"],
                       blk1_cg_bad=int(sum(1 for r in rows if not r["blk1_cg_converged"])), secs=round(secs, 1),
                       psnr=float("nan"), ssim=float("nan"), lpips_alex=float("nan"), lpips_piq=float("nan"))
            if not args.no_metrics:
                r01, g01 = MET.to01(x1), MET.to01(gt)
                fin.update(psnr=MET.psnr(r01, g01), ssim=MET.ssim(r01, g01),
                           lpips_alex=MET.lpips_alex(r01, g01, device), lpips_piq=MET.lpips_piq(r01, g01, device))
            for r in rows:
                r.update(task=task, image=name, seed=seed)
            append_rows(rows_path, ROW_FIELDS, rows)
            append_rows(final_path, FINAL_FIELDS, [fin])
            nfe_stats[f"{task}/{name}"] = NFE["n"]
            if args.save_png:
                save_png(x1, os.path.join(out, f"{task}__{name}__s{seed}.png"))
            hole_s = f"hole={fin['mse_hole']:.4f} " if M["hole"] is not None else ""
            print(f"[{task}/{name}] NFE={NFE['n']} {hole_s}mse={fin['mse_full']:.4f} psnr={fin['psnr']:.2f} "
                  f"resid={fin['meas_resid']:.3f} cg_bad={fin['blk1_cg_bad']} [{secs:.0f}s tot {time.time()-t_all:.0f}s]", flush=True)
    p = os.path.join(out, "nfe.json")
    old = json.load(open(p)) if os.path.exists(p) else {}
    json.dump({**old, **nfe_stats}, open(p, "w"), indent=1)
    print(f"[done] -> {out}", flush=True)


if __name__ == "__main__":
    main()
