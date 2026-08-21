#!/usr/bin/env python
"""Algorithm 4 experiments — single entry point.

    PYTHONHASHSEED=0 python main4.py [--config config_alg4.json] [--mode <mode>]

Algorithm 4 is the clean-endpoint sampler of
``results/algorithm.4pdf.pdf`` ("A clean-endpoint posterior sampler for
cascaded flow priors"; the note numbers its own listing "Algorithm 1", and its
Table 2 column headings read ``Alg. 1 | Alg. 2 | Alg. 1`` -- a typesetting bug,
the third column is this construction). The reading note is
``.research/references/2026-08-21-algorithm4-clean-endpoint-sampler.md``.

Nothing here re-implements method logic: every operator, solve and draw comes
from ``utils.py``, exactly as ``main.py`` does for Algorithm 2. This file is
config plumbing, instrumentation and plots.

    Alg 2 (main.py)                        Alg 4 (this file)
    -------------------------------------  -------------------------------------
    couple through N(x_tau; H x1, s^2 I)   couple through p(x1 | x_tau)      (5)
    p_tau survives -> Langevin on x0       p_tau cancels -> exact draw       (7)
    b carries H^T x_tau / sigma^2          b carries C^-1 x1_hat            (22)
    Block 2: x0 <- x0 - h0/2(...) + ...    Block 2: x_tau <- H x1 + sigma xi (23)
    ridge eps*I at tau=0                   no ridge, ever              Prop. 4(a)
    2 searched quantities (h0, and h1      0 searched quantities: gamma^2 is
      for Alg 1)                             measured (20), S is measured (Sec. 8.6)

ONE config file (config_alg4.json) is the ONLY source of configuration, with
the same strict key contract main.py enforces. Two of Algorithm 2's
"algorithm" keys are DELIBERATELY absent -- ``h0`` and ``ridge_rel`` -- so a
config copied over from Algorithm 2 fails loudly instead of silently passing a
step size that Algorithm 4 would ignore.

The one input Algorithm 2 does not have is ``S_prior``, the prior covariance
surrogate S of (12). It is a measured quantity, not a searched one; see
``--mode measure_s2``. Note that the Tweedie anchor ``lambda`` in
``run_posterior_reg_sampling_alg2`` is the same object: ``lambda*I == S^-1``,
i.e. ``s2 == 1/lambda``, so the lambda sweep in
``test/results_reg_alg2/`` is directly comparable (its best lambda=100 is
s2=0.01).

Modes:
  full_ip    : full Algorithm-4 sampling per task/image -- per-step MSE and
               measurement residual, final metrics (hole/observed split for
               inpainting), loss curves, trajectory frames
  measure_s2 : measure the per-pixel variance of the stage pyramid per stage
               and write s2_meas.json (the Sec. 8.6 quantity), no GPU needed
  verify     : CPU component audit A1-A7 (dense float64) of the claims that are
               specific to Algorithm 4 -- the shared operators are already
               covered by main.py --mode verify

GPU submission (no sbatch file needed):
  sbatch -A cbig-ece -p gpu --gres=gpu:a100:1 -c 8 --mem=64G -t 3:00:00 \\
    -o ../logs/%x-%j.out --export=ALL,PYTHONHASHSEED=0 \\
    --wrap "python main4.py --mode full_ip"
  (PYTHONHASHSEED=0 is required: demo_runner mask seeds still use hash().)

All math lives in utils.py; results land under results/alg4/.
"""

import os

ORIG_CWD = os.getcwd()

import argparse   # noqa: E402
import csv        # noqa: E402
import errno      # noqa: E402
import json       # noqa: E402
import time       # noqa: E402

from utils import (  # noqa: E402  (import first: sets IP_package sys.path + chdir)
    HERE, NFE, count_nfe_hook,
    apply_B, apply_N, apply_H_tau_inv, make_exact_AT, score_solve,
    make_endpoint_operator, clean_endpoint_solve, make_M_tau_den,
    measurement_residual, mse_masked, run_posterior_sampling_alg4,
)
import onestep_mse_vs_t as base  # noqa: E402

import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
from omegaconf import OmegaConf                                # noqa: E402
import matplotlib                                              # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402

from ms_posterior_sampling_article_version_final_utils import (  # noqa: E402
    apply_G, apply_H_tau, compute_sigma_tau, cg_solve, make_Ak_fns,
)
from pixelflow.utils import config as config_utils             # noqa: E402
from demo_runner import load_demo_images                        # noqa: E402
import measurement                                              # noqa: E402
from measurement import build_setup_and_measurement             # noqa: E402
from onestep_visual import to_img                              # noqa: E402

# Populated from config_alg4.json by main() before dispatch — no defaults in code.
PATHS = {}         # resolved "paths" section ({IP_PACKAGE}/{HERE} substituted)
ALG = {}           # "algorithm" section
SAMPLER_KW = {}    # shared "sampler_kw" section
TASKS_SETUP = {}   # per-task {"sigma_n", "operator", optional "kw" overrides}
S_PRIOR = {}       # "S_prior" section — the surrogate S of (12)
TRAJ_IMAGE = None  # "traj_image": which image gets trajectory frames (full_ip)

# Same share-root re-rooting main.py uses; see its resolve_path docstring.
SHARE_ROOTS = [
    "/sfs/ceph/standard/CBIG-Standard-ECE",
    "/standard/CBIG-Standard-ECE",
    "/CBIG-Standard-ECE",
]

# The ceph share this repo lives on intermittently fails a stat with EIO
# ("Remote I/O error") or ESTALE on a path that is perfectly fine a moment
# later; `git status`'s untracked scan fails the same way on this machine.
# os.path.exists() swallows OSError and answers False, so a transient EIO
# reads as "the checkpoint is missing" and resolve_path then raises
# FileNotFoundError listing paths that DO exist. Four of five arms of the
# first S_prior sweep died exactly this way. Retry instead, and only treat a
# genuine ENOENT as absence.
_RETRYABLE_ERRNO = frozenset(
    e for e in (getattr(errno, n, None) for n in ("EIO", "ESTALE", "EBUSY"))
    if e is not None)


def path_exists(p, attempts=5, delay=0.5):
    """os.path.exists, but a transient filesystem error is retried rather than
    silently reported as 'missing'. Returns False only for a real ENOENT."""
    for i in range(attempts):
        try:
            os.stat(p)
            return True
        except FileNotFoundError:
            return False
        except NotADirectoryError:
            return False
        except OSError as exc:
            if exc.errno not in _RETRYABLE_ERRNO or i == attempts - 1:
                raise
            print(f"[fs] transient {exc.strerror!r} on {p!r}, "
                  f"retry {i + 1}/{attempts - 1}", flush=True)
            time.sleep(delay * (i + 1))
    return False


def resolve_path(p):
    """Return an existing form of p, retrying it under every known share
    root; raise with the full candidate list if none is accessible."""
    if path_exists(p):
        return p
    tried = [p]
    for old in SHARE_ROOTS:
        if p.startswith(old + "/"):
            for new in SHARE_ROOTS:
                cand = new + p[len(old):]
                if cand == p:
                    continue
                if path_exists(cand):
                    return cand
                tried.append(cand)
    raise FileNotFoundError(
        "config path not found on this machine; tried: " + ", ".join(tried))


def task_kw(task):
    return {**SAMPLER_KW, **TASKS_SETUP[task].get("kw", {})}


def _resolve_out(out):
    # Anchored to this file's directory, not the caller's CWD (main.py's fix).
    return out if os.path.isabs(out) else os.path.join(HERE, out)


def _load_model(config, device):
    model = config_utils.instantiate_from_config(config.model).to(device)
    ckpt = torch.load(os.path.join(PATHS["model_dir"], "model.pt"),
                      map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt, strict=True)
    model.eval()
    return model


# ════════════════════════ the surrogate S of (12) ══════════════════════════
def make_s2_fn(spec, num_stages):
    """Build ``s2_fn(stage_idx, sigma_tau) -> float`` from config "S_prior".

    Four prescriptions, all from the draft, none of them a free parameter:

      isotropic     s2 = s2                      Sec. 4.2, the reduction of (13)
                    One number for every scale. ``s2 = 1.0`` is the
                    assumption-free choice for this repo: Popoviciu bounds the
                    per-pixel variance of a variable on [a,b] by (b-a)^2/4, and
                    demo_runner normalises images to [-1,1], so s2 <= 1. (The
                    draft writes 1/4 because it assumes [0,1].)
      per_stage     s2 = s2[k]                   (13) measured per scale k
      table         s2 = json[str(k)]            output of --mode measure_s2
      sigma_scaled  s2 = c^2 sigma_tau^2         (14), the no-statistics option

    Sec. 8.2 is explicit about which way to err: overestimating S is bounded in
    its effect (C degrades to the saturated bound of Lemma 3, still admissible),
    while underestimating it is not -- C -> 0 collapses Block 1 onto x1_hat,
    which IMPROVES reconstruction metrics while destroying sample diversity.
    That failure is invisible in MSE, so prefer a larger s2 and use the
    ker(A_k) spread diagnostic (Sec. 8.6) to check.
    """
    mode = spec["mode"]
    if mode == "isotropic":
        s2 = float(spec["s2"])
        return lambda k, sig: s2
    if mode == "per_stage":
        tab = [float(v) for v in spec["s2"]]
        if len(tab) != num_stages:
            raise ValueError(f'S_prior.per_stage: s2 has {len(tab)} entries, '
                             f"model has {num_stages} stages")
        return lambda k, sig: tab[k]
    if mode == "table":
        raw = json.load(open(resolve_path(_resolve_out(spec["path"]))))
        tab = raw["table"] if "table" in raw else raw
        miss = [k for k in range(num_stages) if str(k) not in tab]
        if miss:
            raise KeyError(f'S_prior.table {spec["path"]!r}: no entry for stages {miss}')
        return lambda k, sig: float(tab[str(k)])
    if mode == "sigma_scaled":
        c = float(spec["c"])
        # (14): C^-1 = [H^T H + c^-2 I]/sigma^2. The draft flags that the
        # tau-dependence is wrong (the exact ridge sigma^2/s2 shrinks with
        # sigma_tau, c^-2 does not); (24) bounds the resulting error.
        return lambda k, sig: (c * float(sig)) ** 2
    raise KeyError(f'S_prior.mode {mode!r} not in '
                   '("isotropic", "per_stage", "table", "sigma_scaled")')


# ═══════════════════════════ mode: measure_s2 ══════════════════════════════
def run_measure_s2(args):
    """Sec. 8.6's second reportable quantity: the prior covariance surrogate,
    "obtained from the training data at each scale".

    Measures the pooled per-pixel variance of the stage pyramid, i.e. of
    x1^k = D^{K-1-k} x1, over the configured image set. The pyramid is built by
    ``base.gt_stage_pyramid``, the same chain-of-bilinear-halving the training
    collate uses, so the statistic is of the quantity the model actually sees.

    CAVEAT, written into the output file as well: the images available here are
    the demo/eval set, not the training set. Using an eval-set statistic as the
    prior surrogate is leakage in the strict sense. It is small (a pooled
    second moment over 7 images) but it is not nothing, and the number should be
    replaced by a training-set measurement before anything is reported.
    """
    out = _resolve_out(args.out)
    os.makedirs(out, exist_ok=True)
    device = "cpu"

    demos_all = load_demo_images(resolution=256, demo_dir=PATHS["demo_dir"])
    by_short = {d["short_name"]: d for d in demos_all}
    demos = [by_short[s] for s in args.images]

    config = OmegaConf.load(os.path.join(PATHS["model_dir"], "config.yaml"))
    num_stages = int(config.scheduler.num_stages)

    # Pooled over images, channels and pixels at each stage: sum of squared
    # deviations from the pooled mean, divided by the total count.
    acc = {k: dict(n=0, s=0.0, s2=0.0) for k in range(num_stages)}
    for d in demos:
        gt = d["gt"].unsqueeze(0).to(device)
        pyr = base.gt_stage_pyramid(gt, num_stages)
        for k in range(num_stages):
            v = pyr[k].reshape(-1).double()
            acc[k]["n"] += int(v.numel())
            acc[k]["s"] += float(v.sum())
            acc[k]["s2"] += float((v * v).sum())

    table, report = {}, {}
    for k in range(num_stages):
        n, s, sq = acc[k]["n"], acc[k]["s"], acc[k]["s2"]
        mean = s / n
        var = sq / n - mean * mean
        table[str(k)] = var
        report[str(k)] = dict(n_pixels=n, mean=mean, per_pixel_var=var,
                              lambda_equivalent=(1.0 / var if var > 0 else None))

    payload = {
        "note": "S_prior isotropic surrogate: pooled per-pixel variance of the "
                "stage pyramid x1^k, per stage k. Draft (13) reduction "
                "S = s2*I; Sec. 8.6 asks for this to be reported, not tuned. "
                "1/s2 is the equivalent Tweedie-anchor lambda of "
                "run_posterior_reg_sampling_alg2.",
        "CAVEAT": "Measured on the demo/eval images, NOT on the training set. "
                  "Replace with a training-set measurement before reporting.",
        "images": list(args.images),
        "image_range": "[-1, 1] (demo_runner normalisation); Popoviciu bound "
                       "s2 <= (b-a)^2/4 = 1",
        "detail": report,
        "table": table,
    }
    path = os.path.join(out, "s2_meas.json")
    json.dump(payload, open(path, "w"), indent=1)
    print(f"[measure_s2] {len(demos)} images, {num_stages} stages -> {path}")
    for k in range(num_stages):
        v = table[str(k)]
        print(f"  stage {k}: s2 = {v:.6f}   (lambda-equivalent 1/s2 = {1.0/v:.2f})")
    return 0


# ═══════════════════ incremental, resumable result writing ════════════════
# main.py accumulates every row in memory and writes the CSVs only after the
# whole task x image loop finishes. On this ceph share that lost 21 completed
# GPU runs to a single transient EIO raised while motion_blur lazily imported
# DAPS. Results are therefore appended per cell here, and a re-run skips the
# cells already on disk.
METRIC_FIELDS = ["task", "image", "stage", "step", "tau", "sigma_tau",
                 "s2", "gamma2", "mse_x1", "meas_resid", "x0_rms",
                 "mse_hole", "mse_obs"]


def _final_fields(num_stages):
    return (["task", "image", "trw", "pre_mse", "post_mse", "meas_resid",
             "post_hole", "post_obs"]
            + [f"stage{k}_hole" for k in range(num_stages)])


def _append_rows(path, fields, rows):
    """Append rows, writing the header only when the file is new. restval=''
    keeps blur/SR (no hole) and inpainting in one table."""
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=fields, restval="",
                              extrasaction="ignore")
        if new:
            wcsv.writeheader()
        wcsv.writerows(rows)


def _completed_cells(path):
    """(task, image) pairs already present in a final CSV, for resume."""
    if not os.path.exists(path):
        return set()
    try:
        with open(path, newline="") as f:
            return {(r["task"], r["image"]) for r in csv.DictReader(f)
                    if r.get("task") and r.get("image")}
    except (OSError, KeyError) as exc:
        print(f"[resume] could not read {path} ({exc}); starting fresh",
              flush=True)
        return set()


def _with_retries(what, fn, attempts=6, delay=20):
    """Run one cell, retrying transient filesystem faults.

    The ceph share EIOs during Python's import machinery as well as during
    stats, and a lazily imported dependency (DAPS' forward_operator) is reached
    only when the first motion_blur cell runs -- i.e. 40 minutes into a run.
    A cell is cheap to redo; the run is not.
    """
    for i in range(attempts):
        try:
            return fn()
        except OSError as exc:
            if exc.errno not in _RETRYABLE_ERRNO or i == attempts - 1:
                raise
            print(f"[fs] transient {exc.strerror!r} during {what}, "
                  f"retry {i + 1}/{attempts - 1}", flush=True)
            time.sleep(delay)
    raise RuntimeError("unreachable")


# ═════════════════════════════ mode: full_ip ═══════════════════════════════
def run_full_ip(args):
    out = _resolve_out(args.out)
    frames_dir = os.path.join(out, "frames_tmp")
    os.makedirs(frames_dir, exist_ok=True)
    device = "cuda:0"

    demos_all = load_demo_images(resolution=256, demo_dir=PATHS["demo_dir"])
    by_short = {d["short_name"]: d for d in demos_all}
    demos = [by_short[s] for s in args.images]
    tasks = list(args.tasks)
    if args.smoke:
        tasks, demos = tasks[:1], [by_short[TRAJ_IMAGE]]

    config = OmegaConf.load(os.path.join(PATHS["model_dir"], "config.yaml"))
    model = _load_model(config, device)
    model.register_forward_pre_hook(count_nfe_hook)
    print("[setup] model loaded", flush=True)

    gamma2_tab = json.load(open(PATHS["gamma2_table"]))["table"]
    s2_fn = make_s2_fn(S_PRIOR, int(config.scheduler.num_stages))
    print(f"[setup] S_prior={S_PRIOR} -> "
          f"s2(k=0..3, sigma=0.5) = "
          f"{[round(s2_fn(k, 0.5), 6) for k in range(int(config.scheduler.num_stages))]}",
          flush=True)

    num_stages = int(config.scheduler.num_stages)
    metrics_path = os.path.join(out, "full_ip_metrics.csv")
    final_path = os.path.join(out, "full_ip_final.csv")
    final_fields = _final_fields(num_stages)
    done = _completed_cells(final_path)
    if done:
        print(f"[resume] {len(done)} cell(s) already on disk, skipping them",
              flush=True)

    all_rows, final_rows, nfe_stats, trajs = [], [], {}, {}
    t0 = time.time()
    for task in tasks:
        op_cfg = TASKS_SETUP[task]["operator"]
        kw = task_kw(task)
        if args.smoke:
            kw["num_langevin"] = 2
        sigma_n = float(TASKS_SETUP[task]["sigma_n"])
        for d in demos:
            name = d["short_name"]
            if (task, name) in done:
                print(f"[skip] {task}/{name} (already in {final_path})", flush=True)
                continue
            op, mask, y, _, _, make_Ak_fns_fn, _ = _with_retries(
                f"{task}/{name} setup",
                lambda: build_setup_and_measurement(
                    task, op_cfg, d, sigma_n, 256, device))
            if task in ("gaussian_blur", "motion_blur"):
                inner = make_Ak_fns_fn

                def make_Ak_fns_fn(operator, y_, ss, dev, _inner=inner):
                    Af, _ = _inner(operator, y_, ss, dev)
                    return Af, make_exact_AT(Af, tuple(ss))
            gt = d["gt"].unsqueeze(0).to(device)
            record = (name == TRAJ_IMAGE)
            # The hole/observed split is THE metric every inpainting result in
            # this repo is compared on (test/results_reg_alg2/comparison.csv).
            # main.py only reports it when terminal_replace_weight > 0, which
            # ties a metric to a projection setting; here it is reported
            # whenever the operator actually has a hole.
            hole_full = (1.0 - mask).to(device).float()
            has_hole = float(hole_full.sum()) > 0
            hole_mask = hole_full if has_hole else None

            NFE["n"] = 0
            ts = time.time()
            x1_final, rows, traj = _with_retries(
                f"{task}/{name} sampling",
                lambda: run_posterior_sampling_alg4(
                    model, config, gt, y, op, sigma_n, device,
                    gamma2_tab=gamma2_tab, s2_fn=s2_fn, hole_mask=hole_mask,
                    make_Ak_fns_fn=make_Ak_fns_fn,
                    seed=int(kw.get("seed", ALG["seed"])), record_trajectory=record,
                    sigma_min=ALG["sigma_min"],
                    cg_max_iter_endpoint=ALG["cg_max_iter_endpoint"],
                    terminal_replace_weight=TASKS_SETUP[task]["terminal_replace_weight"],
                    **{k_: v for k_, v in kw.items()
                       if k_ not in ("class_label", "seed")},
                    class_label=int(d["class_idx"])))
            for r in rows:
                r.update(task=task, image=name)
            all_rows += rows
            # rows/traj are recorded pre-projection inside the sampler, so the
            # terminal projection is invisible there; record the returned
            # (post-projection) reconstruction separately.
            trw = float(TASKS_SETUP[task]["terminal_replace_weight"])
            err_f = (x1_final - gt) ** 2
            fin = dict(task=task, image=name, trw=trw,
                       pre_mse=rows[-1]["mse_x1"], post_mse=float(err_f.mean()),
                       meas_resid=rows[-1]["meas_resid"])
            if has_hole:                      # inpainting: split by the mask
                fin["post_hole"] = mse_masked(x1_final, gt, hole_full)
                fin["post_obs"] = mse_masked(x1_final, gt, 1.0 - hole_full)
                # stage-end hole, the columns results_reg_alg2 tabulates
                for k in sorted({r["stage"] for r in rows}):
                    last = [r for r in rows if r["stage"] == k][-1]
                    fin[f"stage{k}_hole"] = last.get("mse_hole")
            final_rows.append(fin)
            # Written now, not at the end of the loop: a cell that completed is
            # a cell that survives whatever the filesystem does next.
            _append_rows(metrics_path, METRIC_FIELDS, rows)
            _append_rows(final_path, final_fields, [fin])
            nfe_stats[f"{task}/{name}"] = NFE["n"]
            if record:
                trajs[task] = traj
            hole_s = (f"hole={fin['post_hole']:.4f} obs={fin['post_obs']:.5f} "
                      if has_hole else "")
            print(f"[{task}/{name}] NFE={NFE['n']} {hole_s}"
                  f"mse={fin['post_mse']:.4f} resid={fin['meas_resid']:.3f} "
                  f"[{time.time()-ts:.0f}s tot {time.time()-t0:.0f}s]", flush=True)

    # nfe.json is merged rather than overwritten so a resumed run keeps the
    # counts of the cells it skipped.
    nfe_path = os.path.join(out, "nfe.json")
    if os.path.exists(nfe_path):
        try:
            nfe_stats = {**json.load(open(nfe_path)), **nfe_stats}
        except (OSError, ValueError):
            pass
    json.dump(nfe_stats, open(nfe_path, "w"), indent=1)
    print(f"[done-sampling] new rows={len(all_rows)} new final={len(final_rows)}",
          flush=True)

    # Plot from disk, not from memory, so a resumed run draws every cell and
    # not just the ones this process happened to compute.
    if os.path.exists(metrics_path):
        with open(metrics_path, newline="") as f:
            all_rows = [{k: (float(v) if k not in ("task", "image") and v != ""
                             else v)
                         for k, v in r.items()}
                        for r in csv.DictReader(f)]
        for r in all_rows:
            r["stage"] = int(r["stage"])

    # ── loss curves + the Sec. 8.6 measurement residual, per task panel ──
    if all_rows and not args.smoke:
        plot_tasks = [t for t in tasks if any(r["task"] == t for r in all_rows)]
        fig, axes = plt.subplots(2, len(plot_tasks),
                                 figsize=(4.0 * len(plot_tasks), 6.4),
                                 squeeze=False)
        for ti, task in enumerate(plot_tasks):
            rs = [r for r in all_rows if r["task"] == task]
            xs = sorted({r["stage"] + r["tau"] for r in rs})
            for row_i, (key, lab, color) in enumerate(
                    [("mse_x1", r"MSE$(\hat{x}_1^k, x_1^k)$", "tab:purple"),
                     ("meas_resid", r"$\|A_kx_1-y\|/(\eta\sqrt{m})$", "tab:orange")]):
                ax = axes[row_i][ti]
                if rs:
                    mean = [np.mean([r[key] for r in rs
                                     if r["stage"] + r["tau"] == x]) for x in xs]
                    ax.semilogy(xs, mean, "-", color=color, lw=1.8, label="Alg4")
                if row_i == 1:
                    # Sec. 8.6: this should settle near 1.
                    ax.axhline(1.0, color="k", lw=0.9, ls=":", alpha=0.8)
                for bdy in (1, 2, 3):
                    ax.axvline(bdy, color="k", lw=0.7, ls="--", alpha=0.5)
                ax.set_title(task if row_i == 0 else "", fontsize=10)
                ax.set_xlabel("global $t$")
                ax.grid(alpha=0.3)
                if ti == 0:
                    ax.set_ylabel(lab)
                    ax.legend(fontsize=8)
        fig.suptitle("Algorithm 4 (clean-endpoint) full sampling — "
                     f"no step size, no ridge; S_prior={S_PRIOR}", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(os.path.join(out, "loss_curves.png"), dpi=160)
        plt.close(fig)

    # ── frames: per task x step, cols = x_tau | x1 | x1_hat ──
    # NOTE the third column differs from main.py's: Algorithm 4 solves (19) for
    # the clean endpoint and never solves (18), so there is no x0_hat here.
    col_labels = [r"$x_\tau^k$", r"$x_1^k$ (drawn)", r"$\hat{x}_1^k$ (19)"]
    frame_idx = 0
    for task in tasks:
        if task not in trajs:
            continue
        for stp, tr in enumerate(trajs[task]):
            fig, axes = plt.subplots(1, 3, figsize=(3 * 2.2, 2.7))
            for ci in range(3):
                ax = axes[ci]
                ax.imshow(to_img(tr[ci].to(device)))
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(col_labels[ci], fontsize=10)
            fig.suptitle(f"Alg4 · {task} · {TRAJ_IMAGE} · "
                         f"step {stp + 1}/{len(trajs[task])}", fontsize=11)
            fig.tight_layout(rect=[0, 0, 1, 0.93])
            fig.savefig(os.path.join(frames_dir, f"frame_{frame_idx:04d}.png"), dpi=110)
            plt.close(fig)
            frame_idx += 1
    print(f"[done] {frame_idx} frames -> {frames_dir}", flush=True)


# ══════════════════════════════ mode: verify ═══════════════════════════════
def run_verify(args):
    """CPU component audit A1-A7 (dense float64) of the ALGORITHM-4-SPECIFIC
    claims. The shared operators (G/H/B/N self-adjointness, sigma_tau, CFG,
    Ak adjoints, power iteration, score_solve) are audited by
    ``main.py --mode verify`` V1-V7 and are not repeated."""
    torch.set_default_dtype(torch.float64)
    RES = int(args.res)
    N_PIX = RES * RES
    SHAPE = (1, 1, RES, RES)

    def dense(op, n=N_PIX, shape=SHAPE):
        M = np.zeros((n, n))
        for j in range(n):
            e = torch.zeros(shape)
            e.view(-1)[j] = 1.0
            M[:, j] = op(e).reshape(-1).numpy()
        return M

    checks = {"ok": True}

    def report(name, val, tol, unit="max|Δ|"):
        ok = bool(val < tol)
        checks["ok"] &= ok
        print(f"  [{name:<52}] {unit} = {val:.3e}  "
              f"{'PASS' if ok else '** FAIL **'} (tol {tol:g})")

    si, s_k, e_k, tau = 1, 0.142857, 0.5, 0.4
    sigma_tau = compute_sigma_tau(tau, s_k, e_k)
    g = torch.Generator().manual_seed(0)
    x1_true = torch.randn(SHAPE, generator=g)
    x0_true = torch.randn(SHAPE, generator=g)
    x_tau = apply_H_tau(x1_true, tau, s_k, e_k, si) + sigma_tau * x0_true
    d_exact = apply_B(x1_true, s_k, e_k, si) - (e_k - s_k) * x0_true

    print("== A1: make_endpoint_operator == score_solve's inline operator ==")
    # The two are written out separately (see make_endpoint_operator's
    # docstring); this is what keeps them from drifting apart.
    for g2 in (0.0, 0.02, 0.5):
        Mine = dense(make_endpoint_operator(s_k, e_k, tau, g2, si))
        Nm = dense(lambda x: apply_N(x, s_k, e_k, si))
        H = dense(lambda x: apply_H_tau(x, tau, s_k, e_k, si))
        Ref = Nm @ Nm + g2 * (H @ H)
        report(f"[N^2+g^2H^2] closure vs dense, gamma2={g2}",
               np.abs(Mine - Ref).max() / max(np.abs(Ref).max(), 1e-30), 1e-12, "rel")

    print("== A2: Prop. 7 — (19) == H_tau^-1 (x_tau - sigma_tau x0_hat), tau>0 ==")
    for g2 in (0.0, 0.01, 0.2):
        x0_hat = score_solve(x_tau, d_exact, s_k, e_k, tau, g2, si, 1e-13, 6000)
        x1_ref = apply_H_tau_inv(x_tau - sigma_tau * x0_hat, tau, s_k, e_k, si)
        x1_19 = clean_endpoint_solve(x_tau, d_exact, sigma_tau, s_k, e_k, tau,
                                     g2, si, 1e-13, 6000)
        report(f"(19) vs (18)+H^-1, gamma2={g2}",
               float((x1_19 - x1_ref).abs().max()), 1e-6)

    print("== A3: limits of (19) (Appendix A.3) + the gamma=0 identity ==")
    # gamma -> 0 : N x1_hat = (e-s) x_tau + sigma_tau v, i.e. (17)
    x1_g0 = clean_endpoint_solve(x_tau, d_exact, sigma_tau, s_k, e_k, tau,
                                 0.0, si, 1e-13, 6000)
    lhs = apply_N(x1_g0, s_k, e_k, si)
    rhs = (e_k - s_k) * x_tau + sigma_tau * d_exact
    report("gamma->0 reduces to (17)", float((lhs - rhs).abs().max()), 1e-8)
    # and with the EXACT displacement the endpoint must be the true x1
    report("exactness: v=d_exact, gamma2=0 -> x1_hat == x1",
           float((x1_g0 - x1_true).abs().max()), 1e-6)
    # gamma -> infinity : x1_hat = H_tau^-1 x_tau (the flat-prior deconvolution
    # the interpolant-coupled samplers centre on)
    x1_ginf = clean_endpoint_solve(x_tau, d_exact, sigma_tau, s_k, e_k, tau,
                                   1e10, si, 1e-14, 20000)
    x1_deconv = apply_H_tau_inv(x_tau, tau, s_k, e_k, si)
    report("gamma->inf reduces to H_tau^-1 x_tau",
           float((x1_ginf - x1_deconv).abs().max()) /
           float(x1_deconv.abs().max()), 1e-4, "rel")

    print("== A4: Prop. 4 — no ridge is needed, at tau=0 included ==")

    class MaskOp:
        def __init__(self, mask):
            self.mask = mask

        def get_mask(self, x=None):
            return self.mask

    mask = (torch.rand(SHAPE, generator=g) > 0.3).double()
    y_probe = torch.randn(SHAPE, generator=g)
    Ak, ATk = make_Ak_fns(MaskOp(mask), y_probe, SHAPE, "cpu")
    eta, s2 = 0.05, 0.25
    for tau0 in (0.0, 0.4):
        sig0 = compute_sigma_tau(tau0, s_k, e_k)
        M_den, Cinv, inv_e2, inv_s2, inv_S = make_M_tau_den(
            Ak, ATk, eta, sig0, tau0, s_k, e_k, si, s2)
        Md = dense(M_den)
        lam_min = float(np.linalg.eigvalsh(Md).min())
        ok = lam_min >= 1.0 / s2 - 1e-9
        checks["ok"] &= ok
        print(f"  [Prop. 4(a) M^den >= S^-1 at tau={tau0:<28}] "
              f"lambda_min = {lam_min:.4f} >= 1/s2 = {1.0/s2:.4f}  "
              f"{'PASS' if ok else '** FAIL **'}")
        # Prop. 4(b): C <= sigma^2 H^-2, i.e. C^-1 - H^T H/sigma^2 = S^-1 >= 0
        Cd = dense(Cinv)
        Hd = dense(lambda x: apply_H_tau(x, tau0, s_k, e_k, si))
        gapd = Cd - (1.0 / sig0 ** 2) * (Hd @ Hd)
        report(f"Prop. 4(b) C^-1 - H^TH/sigma^2 == S^-1 (tau={tau0})",
               np.abs(gapd - np.eye(N_PIX) / s2).max(), 1e-10)
    # At tau=0 the interpolant term must be singular — that is the whole reason
    # Algorithm 2 needs a ridge there and Algorithm 4 does not.
    sig0 = compute_sigma_tau(0.0, s_k, e_k)
    H0 = dense(lambda x: apply_H_tau(x, 0.0, s_k, e_k, si))
    lam0 = float(np.linalg.eigvalsh(H0.T @ H0 / sig0 ** 2).min())
    print(f"  [context: lambda_min(H_0^T H_0/sigma^2) = {lam0:.3e} "
          f"-> Alg. 2 needs its ridge here, Alg. 4 does not]")

    print("== A5: Lemma 9 — Cov(zeta) == M_tau^den exactly ==")
    M_den, Cinv, inv_e2, inv_s2, inv_S = make_M_tau_den(
        Ak, ATk, eta, sigma_tau, tau, s_k, e_k, si, s2)
    Ad = dense(Ak)
    Hd = dense(lambda x: apply_H_tau(x, tau, s_k, e_k, si))
    # zeta = A^T xi_y/eta + H^T xi_h/sigma + xi_s/sqrt(s2)
    Cov = (Ad.T @ Ad) / eta ** 2 + (Hd.T @ Hd) / sigma_tau ** 2 + np.eye(N_PIX) / s2
    report("Cov(zeta) vs M^den", np.abs(Cov - dense(M_den)).max()
           / np.abs(Cov).max(), 1e-12, "rel")

    print("== A6: Block 2 (23) + l.16 return exactly xi_0 ==")
    xi_0 = torch.randn(SHAPE, generator=g)
    x1_drawn = torch.randn(SHAPE, generator=g)
    x_tau_b2 = apply_H_tau(x1_drawn, tau, s_k, e_k, si) + sigma_tau * xi_0
    x0_back = (x_tau_b2 - apply_H_tau(x1_drawn, tau, s_k, e_k, si)) / sigma_tau
    report("l.16 x0 == the xi_0 drawn at l.14",
           float((x0_back - xi_0).abs().max()), 1e-12)

    print("== A7: measurement_residual is the Sec. 8.6 quantity ==")
    resid = measurement_residual(Ak, x1_true, y_probe, eta)
    ref = float(((Ak(x1_true) - y_probe) ** 2).sum().sqrt()) / (
        eta * np.sqrt(y_probe.numel()))
    report("||A x1 - y||/(eta sqrt(m))", abs(resid - ref), 1e-12)

    print("\n" + ("ALL CHECKS PASSED" if checks["ok"] else "** SOME CHECKS FAILED **"))
    return 0 if checks["ok"] else 1


# ═════════════════════════════ dispatcher ══════════════════════════════════
RUNNERS = {"full_ip": run_full_ip, "measure_s2": run_measure_s2,
           "verify": run_verify}

# Exact key contract for config_alg4.json — same discipline as main.py.
# "algorithm" deliberately has NO h0 and NO ridge_rel: Algorithm 4 has no step
# size (Sec. 6.3) and needs no ridge (Prop. 4(a)), so an Algorithm-2 config
# copied over here fails on those keys instead of silently passing dead values.
CONFIG_SCHEMA = {
    "mode": None,
    "paths": {"model_dir", "demo_dir", "gamma2_table"},
    "algorithm": {"sigma_min", "seed", "measurement_seed", "cg_max_iter_endpoint"},
    "sampler_kw": {"num_langevin", "ode_steps_per_stage", "shift", "guidance_scale",
                   "g_bypass_stage3", "cg_tol", "cg_max_iter"},
    "S_prior": None,          # mode-dependent, checked by _check_s_prior
    "tasks_setup": None, "tasks": None, "images": None, "traj_image": None,
    "full_ip": {"out", "smoke"},
    "measure_s2": {"out"},
    "verify": {"res"},
}
TASK_KEYS = {"sigma_n", "operator", "terminal_replace_weight", "measurement_mode",
             "kw"}
TASK_OPERATOR_KEYS = {
    "box_inpainting": {"mask_len_range", "center"},
    "random_inpainting": {"mask_prob"},
    "gaussian_blur": {"kernel_size", "kernel_std"},
    "motion_blur": {"kernel_size", "kernel_intensity", "kernel_seed",
                    "use_daps_kernel"},
    "superresolution": {"scale_factor", "antialias"},
}
S_PRIOR_KEYS = {
    "isotropic": {"mode", "s2"},
    "per_stage": {"mode", "s2"},
    "table": {"mode", "path"},
    "sigma_scaled": {"mode", "c"},
}
DEAD_ALG2_KEYS = {"h0", "h1", "ridge_rel", "cg_max_iter_l14",
                  "anchor_lambda", "anchor_P", "block1_noise", "block2_noise"}


def _check_s_prior(spec):
    if "mode" not in spec:
        raise KeyError('config_alg4.json "S_prior": missing "mode" '
                       f"(one of {sorted(S_PRIOR_KEYS)})")
    mode = spec["mode"]
    if mode not in S_PRIOR_KEYS:
        raise KeyError(f'config_alg4.json "S_prior".mode {mode!r} not in '
                       f"{sorted(S_PRIOR_KEYS)}")
    got, allowed = set(spec), S_PRIOR_KEYS[mode]
    if got != allowed:
        raise KeyError(f'config_alg4.json "S_prior" (mode {mode!r}): '
                       f"missing={sorted(allowed - got)} unknown={sorted(got - allowed)}")


def _check_config_keys(cfg):
    missing = set(CONFIG_SCHEMA) - set(cfg)
    extra = set(cfg) - set(CONFIG_SCHEMA)
    if missing or extra:
        dead = sorted(set(extra) & DEAD_ALG2_KEYS)
        hint = (f"  ({dead} belong to Algorithm 2; Algorithm 4 has no step size "
                "and no ridge)" if dead else "")
        raise KeyError(f"config_alg4.json top level: missing={sorted(missing)} "
                       f"unknown={sorted(extra)}{hint}")
    for sect, allowed in CONFIG_SCHEMA.items():
        if allowed is None:
            continue
        got = set(cfg[sect])
        if got != allowed:
            dead = sorted(got & DEAD_ALG2_KEYS)
            hint = (f"  ({dead} belong to Algorithm 2; Algorithm 4 has no step "
                    "size (Sec. 6.3) and needs no ridge (Prop. 4(a)))"
                    if dead else "")
            raise KeyError(f'config_alg4.json "{sect}": missing={sorted(allowed - got)} '
                           f"unknown={sorted(got - allowed)}{hint}")
    _check_s_prior(cfg["S_prior"])
    required = {"sigma_n", "operator", "terminal_replace_weight",
                "measurement_mode"}
    for task, spec in cfg["tasks_setup"].items():
        got = set(spec)
        if not got <= TASK_KEYS or not required <= got:
            raise KeyError(f'config_alg4.json "tasks_setup"."{task}": '
                           f"missing={sorted(TASK_KEYS - got - {'kw'})} "
                           f"unknown={sorted(got - TASK_KEYS)}")
        if task not in TASK_OPERATOR_KEYS:
            raise KeyError(f'config_alg4.json "tasks_setup": unknown task {task!r} '
                           f"(known: {sorted(TASK_OPERATOR_KEYS)})")
        op_got = set(spec["operator"])
        if op_got != TASK_OPERATOR_KEYS[task]:
            raise KeyError(
                f'config_alg4.json "tasks_setup"."{task}"."operator": '
                f"missing={sorted(TASK_OPERATOR_KEYS[task] - op_got)} "
                f"unknown={sorted(op_got - TASK_OPERATOR_KEYS[task])}")
        kw_got = set(spec.get("kw", {}))          # merged over sampler_kw
        if not kw_got <= CONFIG_SCHEMA["sampler_kw"]:
            raise KeyError(f'config_alg4.json "tasks_setup"."{task}"."kw": '
                           f"unknown={sorted(kw_got - CONFIG_SCHEMA['sampler_kw'])}")
    for name in cfg["tasks"]:
        if name not in cfg["tasks_setup"]:
            raise KeyError(f'config_alg4.json "tasks": {name!r} has no tasks_setup entry')
    if cfg["traj_image"] not in cfg["images"]:
        raise KeyError(f'config_alg4.json "traj_image" {cfg["traj_image"]!r} '
                       "is not in images")


def _check_hash_seed(mode):
    """demo_runner's mask seeds go through Python's hash(), so the masks are
    only reproducible under PYTHONHASHSEED=0 (the documented contract). verify
    builds no masks and is exempt."""
    if mode != "verify" and os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError(
            "PYTHONHASHSEED=0 is required: demo_runner derives mask seeds from "
            f"hash(), so mode {mode!r} would draw a different mask per process. "
            "Re-run as: PYTHONHASHSEED=0 python main4.py ...")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None,
                    help='single config: {"mode": ..., "<mode>": {overrides}} '
                         '(default: <Algorithm2>/config_alg4.json)')
    ap.add_argument("--mode", default=None, help="override the config's mode")
    cli = ap.parse_args()
    # Same rule as main.py: omitting --config uses this directory's config;
    # any value passed on the command line resolves against the caller's CWD.
    if cli.config is None:
        cfg_path = os.path.join(HERE, "config_alg4.json")
    elif os.path.isabs(cli.config):
        cfg_path = cli.config
    else:
        cfg_path = os.path.join(ORIG_CWD, cli.config)
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"--config {cli.config!r} -> {cfg_path} not found")
    cfg = json.load(open(cfg_path))
    _check_config_keys(cfg)
    mode = cli.mode or cfg["mode"]

    if mode not in RUNNERS:
        raise KeyError(f"mode {mode!r} not in {sorted(RUNNERS)}")
    _check_hash_seed(mode)

    global PATHS, ALG, SAMPLER_KW, TASKS_SETUP, S_PRIOR, TRAJ_IMAGE
    TRAJ_IMAGE = cfg["traj_image"]
    subst = {"{IP_PACKAGE}": base.IP_PACKAGE, "{HERE}": HERE}
    PATHS = {k: v for k, v in cfg["paths"].items()}
    for k, v in PATHS.items():
        for token, real in subst.items():
            v = v.replace(token, real)
        PATHS[k] = resolve_path(v)
    ALG = dict(cfg["algorithm"])
    SAMPLER_KW = dict(cfg["sampler_kw"])
    TASKS_SETUP = dict(cfg["tasks_setup"])
    S_PRIOR = dict(cfg["S_prior"])
    measurement.configure(TASKS_SETUP, ALG["measurement_seed"])

    params = dict(cfg[mode])                       # strict: no code defaults
    if mode in ("full_ip", "measure_s2"):
        params.setdefault("images", cfg["images"])
    if mode == "full_ip":
        params.setdefault("tasks", cfg["tasks"])
    print(f"[main4] mode={mode}  config={cfg_path}\n[main4] paths={PATHS}\n"
          f"[main4] algorithm={ALG}\n[main4] S_prior={S_PRIOR}\n"
          f"[main4] params={params}", flush=True)
    return RUNNERS[mode](argparse.Namespace(**params))


if __name__ == "__main__":
    raise SystemExit(main())
