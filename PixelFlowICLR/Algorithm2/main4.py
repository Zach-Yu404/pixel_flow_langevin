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
the same strict key contract main.py enforces. Two values are derived rather
than tuned and their derivations live in the diagnosis report (Sec. 11-12):
``cg_max_iter = 300`` covers the measured worst case of the line-13 solve
(motion_blur stage 0 needs 174 plain-CG iterations), and ``sigma_min`` (config: 1e-8 -- every frame runs; the 0.39 rule below is historical)
stops each stage where the velocity weight of (19), b = N sigma/(N^2+g2 h^2),
falls below 1 -- sigma_tau >= N_k ~= 0.4, minus float32 tolerance so the
sigma_tau = 0.4 boundary frame still runs. Past that point the clean-endpoint
step is provably network-free (verify A8) and measured to only add error. Two of Algorithm 2's
"algorithm" keys are DELIBERATELY absent -- ``h0`` and ``ridge_rel`` -- so a
config copied over from Algorithm 2 fails loudly instead of silently passing a
step size that Algorithm 4 would ignore.

The one input Algorithm 2 does not have is the prior covariance surrogate S
of (12). It is FIXED, not a config option: the per-class spectral S from
s_stats/ (``S_STATS`` / ``default_s2_fn`` below; ImageNet-val statistics,
s_stats/compute_s_stats.py), designated on 2026-09-03 after the S4 test
(s_stats/test/s4_per_task.md). ``--mode measure_s2`` and the scalar
``make_s2_fn`` prescriptions remain only as ablation arms for ``--mode
diversity``. Note that the Tweedie anchor ``lambda`` in
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
  diagnose   : THE box-inpainting diagnosis entry point. One image, full
               per-frame trajectory metrics, Block-1/Block-2 error
               decomposition, schedule and precision terms, critical-frame
               detection, S_it comparison, plots and a montage
  diversity  : repeated posterior samples at fixed y across S arms, so a
               reconstruction gain can be told apart from Block 1 collapsing
               onto x1_hat (draft Sec. 8.2 / 8.6)
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
import math       # noqa: E402
import time       # noqa: E402

from utils import (  # noqa: E402  (import first: sets IP_package sys.path + chdir)
    HERE, NFE, count_nfe_hook,
    apply_B, apply_N, apply_H_tau_inv, make_exact_AT, score_solve,
    make_endpoint_operator, clean_endpoint_solve, make_M_tau_den,
    measurement_residual, mse_masked, run_posterior_sampling_alg4,
    make_jacobi_precond, pcg_solve, SOperator, SpectralSOp,
)
import onestep_mse_vs_t as base  # noqa: E402
import alg4_diag  # noqa: E402

import numpy as np                                             # noqa: E402
import copy                                                    # noqa: E402
import torch                                                   # noqa: E402
from omegaconf import OmegaConf                                # noqa: E402
import matplotlib                                              # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402

from ms_posterior_sampling_article_version_final_utils import (  # noqa: E402
    apply_G, apply_H_tau, compute_sigma_tau, cg_solve, make_Ak_fns,
    make_velocity_fn,
)
import torch.nn.functional as F                                # noqa: E402
from pixelflow.scheduling_pixelflow import PixelFlowScheduler  # noqa: E402
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
# The sampler's prior covariance S of (12)/(22) — FIXED, not a config option.
# S_k = F^H diag(P_k) F (utils.SpectralSOp), P_k = floored (1e-8 max), centred
# mean power spectrum of the image's own ImageNet synset at stage k (1000
# classes x 50 val images, s_stats/compute_s_stats.py). Designated by the user
# on 2026-09-03 after the S4 test (s_stats/test/s4_per_task.md): best of the four
# val-set S (MMSE10 24.86 dB vs 24.72 spectral_all; the scalar val-set tables
# collapse to 14.5 dB on noisy single samples) and on par with the old leaked
# 14-image spectral S. Class-dependent, so the s2_fn is bound per image
# (_bind_s2) once the demo's class_idx is known.
S_STATS = {
    "spectral_npz": os.path.join(HERE, "s_stats", "spectral_power_labelled.npz"),
    "synset_map": "/CBIG-Standard-ECE/Zach_dataset/Zach_dataset/imageNet256/"
                  "LOC_synset_mapping.txt",
}
S_DESC = "spectral_class(s_stats/spectral_power_labelled.npz)"
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
# first S sweep died exactly this way. Retry instead, and only treat a
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
    """Build a SCALAR ``s2_fn(stage_idx, sigma_tau) -> float`` from an arm
    spec -- ablation arms for ``--mode diversity`` only. The sampler's S is
    fixed: ``default_s2_fn`` (per-class spectral S from s_stats/).

    Four scalar prescriptions, all from the draft, none of them a free parameter:

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
    raise KeyError(f'S arm mode {mode!r} not in '
                   '("isotropic", "per_stage", "table", "sigma_scaled")')


def default_s2_fn(num_stages):
    """The sampler's S (see S_STATS): per-class spectral S from s_stats/,
    UNBOUND until ``_bind_s2(s2_fn, class_idx)`` (done inside _run_once and
    the full_ip loop once the demo's class is known)."""
    with open(resolve_path(S_STATS["synset_map"])) as f:
        synsets = [ln.split()[0] for ln in f if ln.strip()]
    if len(synsets) != 1000:
        raise ValueError(f'{S_STATS["synset_map"]}: expected 1000 synsets, '
                         f"got {len(synsets)}")
    return _SpectralS2(resolve_path(S_STATS["spectral_npz"]), num_stages, synsets)


class _SpectralS2:
    """The fixed S as an s2_fn: one utils.SpectralSOp per stage (S diagonal in
    the orthonormal 2-D Fourier basis, power P_k as stored in the npz --
    already floored at 1e-8 max). With ``synsets`` the power is the image's
    own class's, so the object starts UNBOUND; ``_bind_s2(s2_fn, class_idx)``
    returns a cheap bound view (shared npz handle + operator cache) that the
    sampler can call. ``synsets=None`` selects the all-classes spectrum
    (keys ``stage{k}``). Same construction as s_stats/test/run_s4_test.py::
    make_s2fn (validated 2026-09-03: identical power tensors at all stages;
    full runs agree to the GPU's own run-to-run noise, max|dx| ~1e-3)."""

    def __init__(self, npz_path, num_stages, synsets=None):
        self.npz_path, self.K, self.synsets = npz_path, int(num_stages), synsets
        self.npz = np.load(npz_path)     # lazy per key: only this class's 4 arrays are read
        self._cache = {}
        self.class_idx = None

    def _prefix(self):
        if self.synsets is None:
            return ""
        if self.class_idx is None:
            raise RuntimeError("the per-class spectral S needs the image's class: "
                               "call _bind_s2(s2_fn, class_idx) after task setup")
        return f"{self.synsets[self.class_idx]}_"

    def _ops(self, prefix):
        if prefix not in self._cache:
            miss = [k for k in range(self.K) if f"{prefix}stage{k}" not in self.npz.files]
            if miss:
                raise KeyError(f"{self.npz_path}: no '{prefix}stage{{k}}' for stages {miss}")
            self._cache[prefix] = {
                k: SpectralSOp(torch.from_numpy(self.npz[f"{prefix}stage{k}"]),
                               meta=dict(source=os.path.basename(self.npz_path),
                                         key=f"{prefix}stage{k}"))
                for k in range(self.K)}
        return self._cache[prefix]

    def for_class(self, class_idx):
        if self.synsets is None:
            return self
        class_idx = int(class_idx)
        if not 0 <= class_idx < len(self.synsets):
            raise ValueError(f"class_idx {class_idx} outside 0..{len(self.synsets)-1}")
        b = copy.copy(self)
        b.class_idx = class_idx
        return b

    def __call__(self, k, sig):
        return self._ops(self._prefix())[int(k)]

    def describe(self):
        if self.synsets is None:
            tag = "all"
        elif self.class_idx is None:
            return "spectral[class=<unbound until task setup>]"
        else:
            tag = f"class={self.synsets[self.class_idx]}"
        ops = self._ops(self._prefix())
        return (f"spectral[{tag}] Tr(S_k)/D = "
                f"{[round(float(ops[k].scalar_equiv), 6) for k in range(self.K)]}")


def _bind_s2(s2_fn, class_idx):
    """Bind a class-dependent s2_fn (spectral_class) to the image's ImageNet
    class index; every other s2_fn is returned unchanged."""
    return s2_fn.for_class(int(class_idx)) if hasattr(s2_fn, "for_class") else s2_fn


def _describe_s2(s2_fn, num_stages):
    if hasattr(s2_fn, "describe"):
        return s2_fn.describe()
    return (f"s2(k=0..{num_stages - 1}, sigma=0.5) = "
            f"{[round(float(s2_fn(k, 0.5)), 6) for k in range(num_stages)]}")


def _S_inv_sqrt(s_op, x):
    """S^-1/2 x for a scalar s2 or an SOperator (the Lemma-9 R3^T factor)."""
    return (s_op.apply_S_inv_sqrt(x) if isinstance(s_op, SOperator)
            else x / math.sqrt(float(s_op)))


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
# Keep this in sync with the row dict in run_posterior_sampling_alg4:
# _append_rows uses extrasaction="ignore", so a column missing HERE is dropped
# from full_ip's CSV without a word. That is exactly how the blk1_cg_* columns
# -- added to stop a solve failing silently -- were themselves silently lost.
METRIC_FIELDS = ["task", "image", "stage", "step", "tau", "sigma_tau",
                 "s2", "gamma2", "mse_x1", "meas_resid", "x0_rms",
                 "mse_hole", "mse_obs",
                 "blk1_cg_iters", "blk1_cg_resid", "blk1_cg_converged"]


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

    # Every one of these touches the ceph share, and each has been observed to
    # EIO on this machine. Retrying in-process is much cheaper than letting the
    # outer loop restart and reload the 2.7GB checkpoint again.
    demos_all = _with_retries("load_demo_images", lambda: load_demo_images(
        resolution=256, demo_dir=PATHS["demo_dir"]))
    by_short = {d["short_name"]: d for d in demos_all}
    demos = [by_short[s] for s in args.images]
    tasks = list(args.tasks)
    if args.smoke:
        tasks, demos = tasks[:1], [by_short[TRAJ_IMAGE]]

    config = _with_retries("load model config", lambda: OmegaConf.load(
        os.path.join(PATHS["model_dir"], "config.yaml")))
    model = _with_retries("load model weights", lambda: _load_model(config, device))
    model.register_forward_pre_hook(count_nfe_hook)
    print("[setup] model loaded", flush=True)

    gamma2_tab = _with_retries(
        "read gamma2 table",
        lambda: json.load(open(PATHS["gamma2_table"]))["table"])
    s2_fn = _with_retries(
        "build s2_fn", lambda: default_s2_fn(int(config.scheduler.num_stages)))
    print(f"[setup] S={S_DESC} -> "
          f"{_describe_s2(s2_fn, int(config.scheduler.num_stages))}", flush=True)

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

            print(f"[cell] {task}/{name}: {_describe_s2(_bind_s2(s2_fn, d['class_idx']), num_stages)}", flush=True)
            NFE["n"] = 0
            ts = time.time()
            x1_final, rows, traj = _with_retries(
                f"{task}/{name} sampling",
                lambda: run_posterior_sampling_alg4(
                    model, config, gt, y, op, sigma_n, device,
                    gamma2_tab=gamma2_tab, s2_fn=_bind_s2(s2_fn, d["class_idx"]),
                    hole_mask=hole_mask,
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
    bad = [r for r in all_rows if str(r.get("blk1_cg_converged", "1")) == "0"]
    if bad:
        worst = max(float(r["blk1_cg_resid"]) for r in bad)
        print(f"[WARN] Block-1 solve did not converge on {len(bad)}/{len(all_rows)} "
              f"steps (worst relative residual {worst:.2e}). Lemma 9 only gives an "
              f"exact draw for the exact solution -- raise cg_max_iter or improve "
              f"the preconditioner.", flush=True)
    else:
        print("[ok] Block-1 solve converged on every step", flush=True)

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
                     f"no step size, no ridge; S={S_DESC}", fontsize=11)
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


# ═══════════════════ mode: diagnose (box_inpainting only) ══════════════════
def _task_setup(task, image, device, config):
    """One (task, image) cell. The diagnosis modes run on box_inpainting; the
    CG audit takes a task because the Block-1 solve's conditioning depends
    entirely on A_k, and blur is a different operator from a mask."""
    if task not in TASKS_SETUP:
        raise KeyError(f'tasks_setup has no "{task}" entry')
    demos_all = _with_retries("load_demo_images", lambda: load_demo_images(
        resolution=256, demo_dir=PATHS["demo_dir"]))
    by_short = {d["short_name"]: d for d in demos_all}
    if image not in by_short:
        raise KeyError(f"image {image!r} not in demo set {sorted(by_short)}")
    d = by_short[image]
    spec = TASKS_SETUP[task]
    sigma_n = float(spec["sigma_n"])
    op, mask, y, _, _, mkA, _ = _with_retries(
        f"{task}/{image} setup",
        lambda: build_setup_and_measurement(
            task, spec["operator"], d, sigma_n, 256, device))
    if task in ("gaussian_blur", "motion_blur"):
        # same exact-adjoint substitution run_full_ip makes, so the audit sees
        # the operator the sampler actually solves against
        inner = mkA

        def mkA(operator, y_, ss, dev, _inner=inner):
            Af, _ = _inner(operator, y_, ss, dev)
            return Af, make_exact_AT(Af, tuple(ss))
    gt = d["gt"].unsqueeze(0).to(device)
    hole = (1.0 - mask).to(device).float()
    return dict(task=task, demo=d, op=op, mask=mask, y=y, mkA=mkA, gt=gt,
                hole=hole, sigma_n=sigma_n, kw=task_kw(task), spec=spec)


def _box_setup(image, device, config):
    """The cell every box-inpainting diagnosis mode runs on."""
    return _task_setup("box_inpainting", image, device, config)


def _run_once(model, config, S, device, *, s2_fn, gamma2_tab, seed,
              num_langevin=None, diag=None, record_trajectory=False):
    s2_fn = _bind_s2(s2_fn, S["demo"]["class_idx"])   # spectral_class: per-image S
    kw = dict(S["kw"])
    if num_langevin is not None:
        kw["num_langevin"] = int(num_langevin)
    return run_posterior_sampling_alg4(
        model, config, S["gt"], S["y"], S["op"], S["sigma_n"], device,
        gamma2_tab=gamma2_tab, s2_fn=s2_fn, hole_mask=S["hole"],
        make_Ak_fns_fn=S["mkA"], seed=int(seed), diag=diag,
        record_trajectory=record_trajectory,
        sigma_min=ALG["sigma_min"],
        cg_max_iter_endpoint=ALG["cg_max_iter_endpoint"],
        terminal_replace_weight=S["spec"]["terminal_replace_weight"],
        **{k: v for k, v in kw.items() if k not in ("class_label", "seed")},
        class_label=int(S["demo"]["class_idx"]))


def run_diagnose(args):
    out = _resolve_out(args.out)
    os.makedirs(out, exist_ok=True)
    device = "cuda:0"

    config = _with_retries("load model config", lambda: OmegaConf.load(
        os.path.join(PATHS["model_dir"], "config.yaml")))
    model = _with_retries("load model weights", lambda: _load_model(config, device))
    model.register_forward_pre_hook(count_nfe_hook)
    gamma2_tab = _with_retries(
        "read gamma2 table",
        lambda: json.load(open(PATHS["gamma2_table"]))["table"])
    s2_fn = default_s2_fn(int(config.scheduler.num_stages))
    S = _box_setup(args.image, device, config)
    print(f"[diagnose] box_inpainting/{args.image}  S={S_DESC}  "
          f"S_it={S['kw']['num_langevin']}", flush=True)

    # ── the instrumented run ───────────────────────────────────────────────
    n_frames = 4 * int(S["kw"]["ode_steps_per_stage"])
    diag = alg4_diag.Alg4Diagnostics(
        gt_full=S["gt"], hole_full=S["hole"],
        capture_frames=range(n_frames))          # capture everything: one image
    NFE["n"] = 0
    t0 = time.time()
    x1, rows, _ = _run_once(model, config, S, device, s2_fn=s2_fn,
                            gamma2_tab=gamma2_tab, seed=ALG["seed"], diag=diag)
    print(f"[diagnose] instrumented run done, NFE={NFE['n']} "
          f"[{time.time()-t0:.0f}s]", flush=True)

    paths = diag.write_csv(out)
    critical = alg4_diag.find_critical_frames(diag.frames)
    with open(os.path.join(out, "critical_frames.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["kind", "stage", "frame", "tau",
                                          "value", "note"])
        w.writeheader(); w.writerows(critical)
    print("[diagnose] critical frames:")
    for c in critical:
        print(f"    {c['kind']:<16} frame {c['frame']:>3}  stage {c['stage']}  "
              f"tau={c['tau']:.3f}  mse_full={c['value']:.4f}  ({c['note']})")

    # ── the diagnostics must not perturb the sampler ───────────────────────
    # Compared against the committed full_ip run of the same cell, which used
    # the same seed and config and had no diag object and no diagnostics code
    # path at all. Any disagreement beyond GPU non-determinism means the hooks
    # are not read-only.
    cmp_path = _resolve_out(args.compare_to)
    ident = {}
    if os.path.exists(cmp_path):
        with open(cmp_path, newline="") as f:
            ref = [r for r in csv.DictReader(f)
                   if r["task"] == "box_inpainting" and r["image"] == args.image]
        if ref:
            got = dict(post_mse=float(((x1 - S["gt"]) ** 2).mean()),
                       post_hole=mse_masked(x1, S["gt"], S["hole"]),
                       meas_resid=rows[-1]["meas_resid"])
            for k, v in got.items():
                rv = float(ref[0][k])
                ident[k] = dict(instrumented=v, committed=rv,
                                rel=abs(v - rv) / max(abs(rv), 1e-12))
            print("[diagnose] read-only check vs committed full_ip run:")
            for k, d in ident.items():
                print(f"    {k:<12} {d['instrumented']:.6f} vs {d['committed']:.6f}"
                      f"   rel {d['rel']:.2e}")
    json.dump(ident, open(os.path.join(out, "readonly_check.json"), "w"), indent=1)

    # ── S_it comparison (Sec. 4): is the decay inner-loop feedback? ─────────
    sit_rows = []
    default_sit = int(S["kw"]["num_langevin"])
    for sit in [default_sit] + [int(v) for v in args.sit_variants]:
        if any(r["S_it"] == sit for r in sit_rows):
            continue
        if sit == default_sit:
            frames = diag.frames
        else:
            d2 = alg4_diag.Alg4Diagnostics(gt_full=S["gt"], hole_full=S["hole"])
            _run_once(model, config, S, device, s2_fn=s2_fn,
                      gamma2_tab=gamma2_tab, seed=ALG["seed"],
                      num_langevin=sit, diag=d2)
            frames = d2.frames
            with open(os.path.join(out, f"trajectory_metrics_Sit{sit}.csv"),
                      "w", newline="") as f:
                fields = list(dict.fromkeys(k for r in frames for k in r))
                w = csv.DictWriter(f, fieldnames=fields, restval="")
                w.writeheader(); w.writerows(frames)
        st3 = [r for r in frames if r["stage"] == 3]
        sit_rows.append(dict(
            S_it=sit,
            mse_full_final=frames[-1]["mse_full"],
            psnr_full_final=frames[-1]["psnr_full"],
            hole_full_final=frames[-1].get("hole_full", ""),
            obs_full_final=frames[-1].get("obs_full", ""),
            mse_full_stage3_start=st3[0]["mse_full"],
            mse_full_stage3_growth=(st3[-1]["mse_full"] / st3[0]["mse_full"]
                                    if st3[0]["mse_full"] > 0 else float("nan")),
            meas_resid_final=frames[-1]["meas_resid"]))
        print(f"[diagnose] S_it={sit:<3} final mse_full="
              f"{sit_rows[-1]['mse_full_final']:.4f}  stage3 growth="
              f"{sit_rows[-1]['mse_full_stage3_growth']:.2f}x", flush=True)
    with open(os.path.join(out, "sit_comparison.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sit_rows[0].keys()))
        w.writeheader(); w.writerows(sit_rows)

    # ── plots ──────────────────────────────────────────────────────────────
    alg4_diag.plot_loss_vs_frame(diag.frames, critical,
                                 os.path.join(out, "loss_vs_frame.png"))
    alg4_diag.plot_h_sigma(diag.precision, diag.frames, critical,
                           os.path.join(out, "h_sigma_vs_frame.png"))
    alg4_diag.plot_block_decomposition(
        diag.inner, diag.frames, critical,
        os.path.join(out, "block_error_decomposition.png"))
    alg4_diag.plot_inner_feedback(diag.inner,
                                  os.path.join(out, "inner_loop_feedback.png"))
    ids = alg4_diag.montage_frames(diag.frames, critical)
    alg4_diag.plot_montage(diag.captured, diag.frames, ids, S["gt"], S["hole"],
                           os.path.join(out, "trajectory_montage.png"),
                           device=device)
    alg4_diag.plot_final_output(diag.captured, diag.frames, S["gt"], S["hole"],
                                os.path.join(out, "final_output.png"),
                                device=device)
    json.dump(dict(montage_frames=ids), open(
        os.path.join(out, "montage_frames.json"), "w"), indent=1)
    print(f"[diagnose] wrote {sorted(os.listdir(out))}", flush=True)
    return 0


# ═════════════ mode: diversity (scalar S arms x seeds, fixed y) ═════════════
def run_diversity(args):
    """Sec. 8.2 / 8.6: a smaller s2 improves reconstruction, but the draft says
    that is also what collapsing Block 1 onto x1_hat looks like. The two are
    told apart by the spread of repeated posterior samples inside ker(A_k) --
    for box inpainting, the hole.

    y is built once and reused for every seed (measurement_seed is independent
    of the sampler seed), which is the repo's standing rule for this kind of
    comparison.
    """
    out = os.path.join(_resolve_out(args.out), "diversity")
    sens = os.path.join(_resolve_out(args.out), "s_prior_sensitivity")
    os.makedirs(out, exist_ok=True); os.makedirs(sens, exist_ok=True)
    device = "cuda:0"

    config = _with_retries("load model config", lambda: OmegaConf.load(
        os.path.join(PATHS["model_dir"], "config.yaml")))
    model = _with_retries("load model weights", lambda: _load_model(config, device))
    model.register_forward_pre_hook(count_nfe_hook)
    gamma2_tab = _with_retries(
        "read gamma2 table",
        lambda: json.load(open(PATHS["gamma2_table"]))["table"])
    num_stages = int(config.scheduler.num_stages)
    S = _box_setup(args.image, device, config)

    per_run = os.path.join(out, "per_run.csv")
    done = set()
    if os.path.exists(per_run):
        with open(per_run, newline="") as f:
            done = {(r["arm"], int(r["seed"])) for r in csv.DictReader(f)}
        print(f"[diversity] resuming; {len(done)} run(s) already on disk", flush=True)

    summary = []
    for arm in args.arms:
        name = arm["name"]
        spec = {k: v for k, v in arm.items() if k != "name"}
        _check_s_prior(spec)
        s2_fn = make_s2_fn(spec, num_stages)
        samples = []
        for seed in args.seeds:
            tag = f"{name}_seed{seed}"
            npy = os.path.join(out, f"{tag}.npy")
            if (name, int(seed)) in done and os.path.exists(npy):
                samples.append(torch.from_numpy(np.load(npy)).to(device))
                print(f"[skip] {tag}", flush=True)
                continue
            t0 = time.time()
            x1, rows, _ = _with_retries(tag, lambda: _run_once(
                model, config, S, device, s2_fn=s2_fn, gamma2_tab=gamma2_tab,
                seed=seed))
            np.save(npy, x1.detach().cpu().numpy())
            samples.append(x1.detach())
            row = dict(arm=name, seed=int(seed),
                       s2_stage3=float(s2_fn(num_stages - 1, 0.4)),
                       mse=float(((x1 - S["gt"]) ** 2).mean()),
                       hole=mse_masked(x1, S["gt"], S["hole"]),
                       obs=mse_masked(x1, S["gt"], 1.0 - S["hole"]),
                       meas_resid=rows[-1]["meas_resid"])
            _append_rows(per_run, ["arm", "seed", "s2_stage3", "mse", "hole",
                                   "obs", "meas_resid"], [row])
            print(f"[{tag}] hole={row['hole']:.4f} obs={row['obs']:.5f} "
                  f"resid={row['meas_resid']:.3f} [{time.time()-t0:.0f}s]",
                  flush=True)

        # spread across seeds, per pixel, then pooled over the hole / observed
        st = torch.stack(samples, 0)                       # [n_seed,1,3,H,W]
        sd = st.std(dim=0, unbiased=True)                  # [1,3,H,W]
        hole = S["hole"]
        n_h = (hole.sum() * 3).clamp(min=1)
        n_o = ((1 - hole).sum() * 3).clamp(min=1)
        summary.append(dict(
            arm=name, n_seeds=len(samples),
            s2_stage3=float(s2_fn(num_stages - 1, 0.4)),
            lambda_eq=1.0 / float(s2_fn(num_stages - 1, 0.4)),
            hole_spread=float((sd * hole).sum() / n_h),
            obs_spread=float((sd * (1 - hole)).sum() / n_o),
            hole_mse_mean=float(np.mean(
                [mse_masked(x, S["gt"], hole) for x in samples])),
            obs_mse_mean=float(np.mean(
                [mse_masked(x, S["gt"], 1 - hole) for x in samples]))))
        r = summary[-1]
        print(f"[diversity] {name:<10} lambda_eq={r['lambda_eq']:>8.1f}  "
              f"hole_mse={r['hole_mse_mean']:.4f}  "
              f"hole_spread={r['hole_spread']:.5f}  "
              f"obs_spread={r['obs_spread']:.5f}", flush=True)

    with open(os.path.join(sens, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)
    # All plotting for the diagnosis modes lives in alg4_diag.
    alg4_diag.plot_s_prior_vs_diversity(
        summary, os.path.join(sens, "s_prior_vs_diversity.png"))
    print(f"[diversity] wrote {sens} and {out}", flush=True)
    return 0


# ═════════ mode: contraction (controlled one-step probe, box only) ═════════
def _stage_schedule(config, si, ode_steps, shift, device):
    """The sampler's own scheduler state for one stage, so the probe runs on
    exactly the (s_k, e_k, tau, T) the trajectory saw."""
    sc = PixelFlowScheduler(config.scheduler.num_train_timesteps,
                            num_stages=int(config.scheduler.num_stages), gamma=-1 / 3)
    sc.set_timesteps(int(ode_steps), si, device=device, shift=shift)
    return sc


def run_contraction(args):
    """Is the stage-3 runaway inevitable, and does the network still pull back?

    The trajectory alone cannot separate "the inner loop is unstable here" from
    "one unlucky Block-2 draw kicked it out". So probe the map directly: plant a
    KNOWN amount of error in the hole, run exactly one clean-endpoint -> Block 1
    -> Block 2 pass at a given frame, and read off where the error goes. Sweeping
    the planted error traces the one-step map and its fixed points.

    Nothing here is a new hyperparameter: the planted error is the probe's
    independent variable, not a setting the sampler ever uses.
    """
    out = os.path.join(_resolve_out(args.out), "contraction")
    os.makedirs(out, exist_ok=True)
    device = "cuda:0"

    config = _with_retries("load model config", lambda: OmegaConf.load(
        os.path.join(PATHS["model_dir"], "config.yaml")))
    model = _with_retries("load model weights", lambda: _load_model(config, device))
    gamma2_tab = _with_retries("read gamma2 table",
                               lambda: json.load(open(PATHS["gamma2_table"]))["table"])
    num_stages = int(config.scheduler.num_stages)
    s2_fn = default_s2_fn(num_stages)
    S = _box_setup(args.image, device, config)
    s2_fn = _bind_s2(s2_fn, S["demo"]["class_idx"])
    kw = S["kw"]
    ode_steps = int(kw["ode_steps_per_stage"])
    pyr = base.gt_stage_pyramid(S["gt"], num_stages)

    rows = []
    gen = torch.Generator(device="cpu").manual_seed(int(ALG["seed"]))
    for frame in args.frames:
        si, step = divmod(int(frame), ode_steps)
        sc = _stage_schedule(config, si, ode_steps, float(kw["shift"]), device)
        s_k, e_k = float(sc.start_t[si]), float(sc.end_t[si])
        tau = float(sc.t[step]); T = sc.Timesteps[step]
        sigma_tau = compute_sigma_tau(tau, s_k, e_k)
        gamma2 = float(gamma2_tab[str(si)].get(
            f"{round(tau, 6)}", list(gamma2_tab[str(si)].values())[step]))
        s_op = s2_fn(si, float(sigma_tau))
        s2 = float(s_op.scalar_equiv) if isinstance(s_op, SOperator) else float(s_op)
        gt_k = pyr[si]
        h, w = gt_k.shape[-2:]
        hole_k = F.interpolate(S["hole"], size=(h, w), mode="nearest")
        Ak, ATk = S["mkA"](S["op"], S["y"], (1, 3, h, w), device)
        # eff_si=None: real G at every stage (user decision 2026-08-24)
        M_den, Cinv, inv_e2, inv_s2, inv_S = make_M_tau_den(
            Ak, ATk, S["sigma_n"], sigma_tau, tau, s_k, e_k, None, s_op)
        inv_S_half = 1.0 / math.sqrt(s2)
        pe = torch.tensor([int(S["demo"]["class_idx"])], dtype=torch.int32, device=device)
        do_cfg = float(kw["guidance_scale"]) > 0
        emb = (torch.cat([int(model.num_classes) * torch.ones_like(pe), pe], 0)
               if do_cfg else pe)
        size_tensor, rope_pos = base.rope_for(model, h, w, device)
        vfn = make_velocity_fn(model, T, emb, size_tensor, rope_pos, do_cfg,
                               float(kw["guidance_scale"]), si)

        # The planted-noise sweep, plus (optionally) a REPLAY arm: the actual x1
        # the sampler ended up with, injected at this frame. White corruption of
        # the same MSE and the sampler's own structured error are not the same
        # object, and the probe is only informative if it says which one the
        # denoiser can undo.
        probes = [("gauss", d) for d in args.deltas]
        replay = None
        if getattr(args, "replay_x1", None):
            rp = _resolve_out(args.replay_x1)
            if os.path.exists(rp):
                replay = torch.from_numpy(np.load(rp)).to(device)
                probes.append(("replay", float("nan")))
            else:
                print(f"[contraction] replay_x1 {rp} not found, skipping", flush=True)
        for kind, delta in probes:
            for rep in range(int(args.repeats)):
                if kind == "replay":
                    # downsample the full-res sample to this stage, hole only:
                    # the observed region is kept at GT so the two arms differ
                    # ONLY in the structure of the hole error.
                    rk = F.interpolate(replay, size=(h, w), mode="area")
                    x1 = gt_k * (1.0 - hole_k) + rk * hole_k
                else:
                    n = torch.randn(gt_k.shape, generator=gen).to(device)
                    x1 = gt_k + math.sqrt(delta) * n * hole_k
                d_in = mse_masked(x1, gt_k, hole_k)
                xi0 = torch.randn(gt_k.shape, generator=gen).to(device)
                x_tau = apply_H_tau(x1, tau, s_k, e_k, None) + sigma_tau * xi0
                with torch.no_grad():
                    v = vfn(x_tau)
                x1_hat = clean_endpoint_solve(x_tau, v, sigma_tau, s_k, e_k, tau,
                                              gamma2, None, float(kw["cg_tol"]),
                                              ALG["cg_max_iter_endpoint"],
                                              x1_warm=x1.clone())
                d_hat = mse_masked(x1_hat, gt_k, hole_k)
                xi_y = torch.randn(S["y"].shape, generator=gen).to(device)
                xi_h = torch.randn(gt_k.shape, generator=gen).to(device)
                xi_s = torch.randn(gt_k.shape, generator=gen).to(device)
                b_t = (inv_e2 * ATk(S["y"]) + Cinv(x1_hat)
                       + (1.0 / S["sigma_n"]) * ATk(xi_y)
                       + (1.0 / float(sigma_tau)) * apply_H_tau(xi_h, tau, s_k, e_k, None)
                       + _S_inv_sqrt(s_op, xi_s))
                x1_out = cg_solve(M_den, b_t, x0=x1.clone(),
                                  tol=float(kw["cg_tol"]),
                                  max_iter=int(kw["cg_max_iter"]))
                d_out = mse_masked(x1_out, gt_k, hole_k)
                rows.append(dict(frame=int(frame), stage=si, step=step, tau=tau,
                                 kind=kind,
                                 sigma_tau=float(sigma_tau), gamma2=gamma2, s2=s2,
                                 h_tau=(1 - tau) * s_k + tau * e_k,
                                 C_inv=inv_s2 * ((1 - tau) * s_k + tau * e_k) ** 2 + inv_S,
                                 rep=rep, delta_planted=delta,
                                 d_in=d_in, d_hat=d_hat, d_out=d_out,
                                 endpoint_gain=d_hat - d_in,
                                 block1_gain=d_out - d_hat,
                                 net_gain=d_out - d_in))
                print(f"  frame {frame:>2} {kind:<6} delta={delta:<6} rep{rep}: "
                      f"in={d_in:.4f} -> hat={d_hat:.4f} -> out={d_out:.4f}",
                      flush=True)

    path = os.path.join(out, "contraction_map.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    alg4_diag.plot_contraction_map(rows, os.path.join(out, "contraction_map.png"))
    print(f"[contraction] wrote {path}", flush=True)
    return 0


# ═══════════ mode: cg_audit (does line 13 actually converge?) ══════════════
def run_cg_audit(args):
    """Line 13 is only an exact draw if its solve converges (Lemma 9).

    This measures, at the REAL resolution and with the real box mask, how many
    CG iterations line 13 needs, what the residual still is at the configured
    cap, and how far the truncated answer sits from the converged one -- with
    and without the Jacobi preconditioner. No network is needed: line 13 uses
    only A_k, H_tau and S.
    """
    out = os.path.join(_resolve_out(args.out), "cg_audit")
    os.makedirs(out, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    config = _with_retries("load model config", lambda: OmegaConf.load(
        os.path.join(PATHS["model_dir"], "config.yaml")))
    num_stages = int(config.scheduler.num_stages)
    s2_fn = default_s2_fn(num_stages)
    S = _task_setup(getattr(args, "task", "box_inpainting"), args.image,
                    device, config)
    s2_fn = _bind_s2(s2_fn, S["demo"]["class_idx"])
    kw = S["kw"]
    ode_steps = int(kw["ode_steps_per_stage"])
    cap = int(kw["cg_max_iter"])
    tol = float(kw["cg_tol"])
    gt = S["gt"]
    pyr = base.gt_stage_pyramid(gt, num_stages)
    g = torch.Generator(device="cpu").manual_seed(int(ALG["seed"]))

    rows = []
    for frame in args.frames:
        si, step = divmod(int(frame), ode_steps)
        sc = _stage_schedule(config, si, ode_steps, float(kw["shift"]), device)
        s_k, e_k = float(sc.start_t[si]), float(sc.end_t[si])
        tau = float(sc.t[step])
        sigma_tau = compute_sigma_tau(tau, s_k, e_k)
        if sigma_tau < ALG["sigma_min"]:
            continue
        s_op = s2_fn(si, float(sigma_tau))
        s2 = float(s_op.scalar_equiv) if isinstance(s_op, SOperator) else float(s_op)
        h, w = pyr[si].shape[-2:]
        Ak, ATk = S["mkA"](S["op"], S["y"], (1, 3, h, w), device)
        # eff_si=None: real G at every stage (user decision 2026-08-24)
        M_den, Cinv, inv_e2, inv_s2, inv_S = make_M_tau_den(
            Ak, ATk, S["sigma_n"], sigma_tau, tau, s_k, e_k, None, s_op)
        # a right-hand side with the same structure line 13 builds
        xi_y = torch.randn(S["y"].shape, generator=g).to(device)
        xi_h = torch.randn(pyr[si].shape, generator=g).to(device)
        xi_s = torch.randn(pyr[si].shape, generator=g).to(device)
        b = (inv_e2 * ATk(S["y"]) + Cinv(pyr[si])
             + (1.0 / S["sigma_n"]) * ATk(xi_y)
             + (1.0 / float(sigma_tau)) * apply_H_tau(xi_h, tau, s_k, e_k, si)
             + _S_inv_sqrt(s_op, xi_s))
        warm = pyr[si].clone()

        probe = int(getattr(args, "max_probe", 4000))
        ref, it_ref, rel_ref = pcg_solve(M_den, b, None, x0=warm.clone(),
                                         tol=1e-10, max_iter=probe)
        x_cap, it_cap, rel_cap = pcg_solve(M_den, b, None, x0=warm.clone(),
                                           tol=tol, max_iter=cap)
        x_tolc, it_tolc, _ = pcg_solve(M_den, b, None, x0=warm.clone(),
                                       tol=tol, max_iter=probe)
        Minv = make_jacobi_precond(M_den, pyr[si].shape, device)
        if Minv is None:
            it_pc, rel_pc, err_pc = -1, float("nan"), float("nan")
        else:
            x_pc, it_pc, rel_pc = pcg_solve(M_den, b, Minv, x0=warm.clone(),
                                            tol=tol, max_iter=probe)
            err_pc = float((x_pc - ref).norm() / ref.norm())
        rows.append(dict(
            task=S["task"], frame=int(frame), stage=si, step=step, tau=tau,
            sigma_tau=float(sigma_tau), stage_res=h, s2=s2,
            cap=cap, tol=tol,
            iters_plain_to_tol=it_tolc,
            converged_within_cap=int(it_tolc <= cap),
            rel_resid_at_cap=rel_cap,
            err_at_cap=float((x_cap - ref).norm() / ref.norm()),
            iters_jacobi_to_tol=it_pc,
            rel_resid_jacobi=rel_pc,
            err_jacobi=err_pc))
        r = rows[-1]
        print(f"  frame {frame:>2} st{si} res{h:>3} sigma={sigma_tau:.4f}  "
              f"plain {it_tolc:>4} iters (cap {cap}{'' if r['converged_within_cap'] else ' EXCEEDED'})"
              f"  err@cap {r['err_at_cap']:.2e}   jacobi {it_pc:>3} iters"
              f"  err {err_pc:.2e}", flush=True)

    path = os.path.join(out, "cg_audit.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    bad = [r for r in rows if not r["converged_within_cap"]]
    print(f"[cg_audit] {len(bad)}/{len(rows)} frames do NOT converge within "
          f"cg_max_iter={cap}", flush=True)
    if bad:
        print(f"[cg_audit]   worst relative error of the truncated draw: "
              f"{max(r['err_at_cap'] for r in bad):.3e}", flush=True)
    print(f"[cg_audit] wrote {path}", flush=True)
    return 0


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

    print("== A8: sigma_tau -> 0 drives (19) to H_tau^-1 x_tau, network excluded ==")
    # Using N = sigma B + (e-s) H (draft (3)), sigma -> 0 gives N -> (e-s) H, so
    #     [N^2+g2H^2] x1_hat = N[(e-s)x_tau + sigma v] + g2 H x_tau
    # collapses to [(e-s)^2+g2] H^2 x1_hat = [(e-s)^2+g2] H x_tau, i.e.
    #     x1_hat = H_tau^-1 x_tau,   for ANY G and independently of v_theta.
    # The velocity enters (19) only through the term sigma * N * v, so its weight
    # is O(sigma_tau). This is the sigma->0 counterpart of the draft's gamma->inf
    # limit (A.3); the draft does not state it, and it is what empties the
    # clean-endpoint estimate of network information at the end of the last stage.
    s3, e3 = 0.6, 1.0                      # the only stage with e_k = 1
    v1 = torch.randn(SHAPE, generator=g)
    v2 = torch.randn(SHAPE, generator=g)
    dv = float((v1 - v2).norm())
    print(f"  {'tau':>7}{'sigma_tau':>11}{'|x1hat-H^-1 x_tau|':>21}"
          f"{'d|x1hat|/d|v| (=b)':>20}")
    for tau3 in (0.5, 0.9, 0.99, 0.999, 0.9999):
        sig3 = compute_sigma_tau(tau3, s3, e3)
        xt = torch.randn(SHAPE, generator=g)
        g2 = 0.02
        a1 = clean_endpoint_solve(xt, v1, sig3, s3, e3, tau3, g2, None, 1e-13, 6000)
        a2 = clean_endpoint_solve(xt, v2, sig3, s3, e3, tau3, g2, None, 1e-13, 6000)
        ref = apply_H_tau_inv(xt, tau3, s3, e3, None)
        gap = float((a1 - ref).abs().max())
        bcoef = float((a1 - a2).norm()) / dv
        print(f"  {tau3:>7.4f}{sig3:>11.6f}{gap:>21.3e}{bcoef:>20.6f}")
        if tau3 == 0.9999:
            report("sigma->0: (19) == H_tau^-1 x_tau", gap, 1e-3)
            report("sigma->0: velocity weight b -> 0", bcoef, 1e-3, "b")

    print("\n" + ("ALL CHECKS PASSED" if checks["ok"] else "** SOME CHECKS FAILED **"))
    return 0 if checks["ok"] else 1


# ═════════════════════════════ dispatcher ══════════════════════════════════
RUNNERS = {"full_ip": run_full_ip, "measure_s2": run_measure_s2,
           "diagnose": run_diagnose, "diversity": run_diversity,
           "contraction": run_contraction, "cg_audit": run_cg_audit,
           "verify": run_verify}

# Exact key contract for config_alg4.json — same discipline as main.py.
# "algorithm" deliberately has NO h0 and NO ridge_rel: Algorithm 4 has no step
# size (Sec. 6.3) and needs no ridge (Prop. 4(a)), so an Algorithm-2 config
# copied over here fails on those keys instead of silently passing dead values.
CONFIG_SCHEMA = {
    "mode": None,
    "paths": {"model_dir", "demo_dir", "gamma2_table"},
    "algorithm": {"sigma_min", "seed", "measurement_seed", "cg_max_iter_endpoint"},
    # g_bypass_stage3 is deliberately absent: the stage-3 identity bypass was
    # removed from apply_G (2026-08-24) — G is the same projection at every
    # stage — so it is not a variable here and a config that sets it fails
    # loudly.
    "sampler_kw": {"num_langevin", "ode_steps_per_stage", "shift", "guidance_scale",
                   "cg_tol", "cg_max_iter"},
    "tasks_setup": None, "tasks": None, "images": None, "traj_image": None,
    "full_ip": {"out", "smoke"},
    "measure_s2": {"out"},
    "diagnose": {"out", "image", "sit_variants", "compare_to"},
    "diversity": {"out", "image", "arms", "seeds"},
    "contraction": {"out", "image", "frames", "deltas", "repeats", "replay_x1"},
    "cg_audit": {"out", "image", "frames", "task", "max_probe"},
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
                  "anchor_lambda", "anchor_P", "block1_noise", "block2_noise",
                  "g_bypass_stage3"}


def _check_s_prior(spec):
    if "mode" not in spec:
        raise KeyError('diversity arm: missing "mode" '
                       f"(one of {sorted(S_PRIOR_KEYS)})")
    mode = spec["mode"]
    if mode not in S_PRIOR_KEYS:
        raise KeyError(f'diversity arm mode {mode!r} not in '
                       f"{sorted(S_PRIOR_KEYS)}")
    got, allowed = set(spec), S_PRIOR_KEYS[mode]
    if got != allowed:
        raise KeyError(f'diversity arm (mode {mode!r}): '
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

    global PATHS, ALG, SAMPLER_KW, TASKS_SETUP, TRAJ_IMAGE
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
    measurement.configure(TASKS_SETUP, ALG["measurement_seed"])

    params = dict(cfg[mode])                       # strict: no code defaults
    if mode in ("full_ip", "measure_s2"):
        params.setdefault("images", cfg["images"])
    if mode == "full_ip":
        params.setdefault("tasks", cfg["tasks"])
    print(f"[main4] mode={mode}  config={cfg_path}\n[main4] paths={PATHS}\n"
          f"[main4] algorithm={ALG}\n[main4] S={S_DESC}\n"
          f"[main4] params={params}", flush=True)
    return RUNNERS[mode](argparse.Namespace(**params))


if __name__ == "__main__":
    raise SystemExit(main())
