#!/usr/bin/env python
"""Prior-covariance S for Algorithm 4 (box inpainting): the two retained
constructions, pooled_junco (the incumbent scalar table) and spectral.

History: six constructions were compared on 2026-08-25 (pooled, centered_trace,
channel_diag, two_band_G, spectral, upper_bound vs the incumbent); only the
incumbent and spectral survived — see results/alg4_box_s_prior_methods/report.md
for the full comparison and the reasons. The losing constructions' code and
result directories were removed on user request; report.md keeps the record.

The ONLY variable is how S of (12)/(22) is built; Eq. (19), Block 1, Block 2,
the schedule, S_it and every other sampler knob stay as configured in
config_alg4.json. Both constructions flow through the SOperator interface in
utils.py (apply_S_inv / apply_S_inv_sqrt) and the one Block-1 implementation.

Usage (from Algorithm2/, PYTHONHASHSEED=0 required):

    python s_prior_methods.py measure                # s_statistics.json + spectral power
    python s_prior_methods.py onestep                # frozen-state Block 1, both arms
    python s_prior_methods.py trajectory [--arms pooled_junco,spectral]
    python s_prior_methods.py spread     [--arms ...]
    python s_prior_methods.py analyze                # crossover csv + plots

Everything lands in results/alg4_box_s_prior_methods/.

DATA CAVEAT — DEMO_ONLY / LEAKAGE: spectral statistics come from the 6
non-junco demo images (junco, the evaluation image, is excluded to avoid
direct leakage), but the demo set is still the eval family, not a
training/calibration set. pooled_junco is the junco-only pooled variance
(maximal leakage, kept as the incumbent baseline). No paper statistics.
"""
import argparse
import csv
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402

import main4                                                   # noqa: E402
import alg4_diag                                               # noqa: E402
import utils as U                                              # noqa: E402
from utils import (                                            # noqa: E402
    SOperator, ScalarSOp, SpectralSOp,
    make_M_tau_den, make_jacobi_precond, pcg_solve)
from omegaconf import OmegaConf                                # noqa: E402

OUT = os.path.join(HERE, "results", "alg4_box_s_prior_methods")
EVAL_IMAGE = "junco"
ONESTEP_FRAMES = (32, 35, 37, 38)   # normal / ratchet window / tau=0.777 / late low-sigma
ONESTEP_DRAWS = 16
ONESTEP_SEED0 = 8000                # draw d uses generator seed ONESTEP_SEED0+d
SPECTRAL_FLOOR_REL = 1e-8           # floor = SPECTRAL_FLOOR_REL * max(P_k); fixed, not searched
SPREAD_SEEDS = (42, 43, 44, 45)
ALL_ARMS = ("pooled_junco", "spectral")


# ───────────────────────────── shared setup ────────────────────────────────
def _init_globals():
    cfg = json.load(open(os.path.join(HERE, "config_alg4.json")))
    main4._check_config_keys(cfg)
    main4._check_hash_seed("diagnose")
    main4.TRAJ_IMAGE = cfg["traj_image"]
    subst = {"{IP_PACKAGE}": main4.base.IP_PACKAGE, "{HERE}": main4.HERE}
    for k, v in cfg["paths"].items():
        for t, r in subst.items():
            v = v.replace(t, r)
        main4.PATHS[k] = main4.resolve_path(v)
    main4.ALG.update(cfg["algorithm"])
    main4.SAMPLER_KW.update(cfg["sampler_kw"])
    main4.TASKS_SETUP.update(cfg["tasks_setup"])
    main4.S_PRIOR.update(cfg["S_prior"])
    main4.measurement.configure(main4.TASKS_SETUP, main4.ALG["measurement_seed"])
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(OUT, "configs"), exist_ok=True)
    return cfg


def _load_sampling(device="cuda:0"):
    config = main4._with_retries("load model config", lambda: OmegaConf.load(
        os.path.join(main4.PATHS["model_dir"], "config.yaml")))
    model = main4._with_retries("load model weights",
                                lambda: main4._load_model(config, device))
    gamma2_tab = main4._with_retries(
        "read gamma2 table",
        lambda: json.load(open(main4.PATHS["gamma2_table"]))["table"])
    S = main4._box_setup(EVAL_IMAGE, device, config)
    return config, model, gamma2_tab, S


def _stats_path():
    return os.path.join(OUT, "s_statistics.json")


def _spectral_path():
    return os.path.join(OUT, "configs", "spectral_power.npz")


# ───────────────────────────── measure ─────────────────────────────────────
def cmd_measure(_args):
    """Spectral per-stage statistics (6 non-junco demo images). DEMO_ONLY."""
    _init_globals()
    config = OmegaConf.load(os.path.join(main4.PATHS["model_dir"], "config.yaml"))
    K = int(config.scheduler.num_stages)
    demos_all = main4.load_demo_images(resolution=256,
                                       demo_dir=main4.PATHS["demo_dir"])
    demos = [d for d in demos_all if d["short_name"] != EVAL_IMAGE]
    names = [d["short_name"] for d in demos]
    pyrs = [main4.base.gt_stage_pyramid(d["gt"].unsqueeze(0), K) for d in demos]
    N = len(pyrs)
    spectral_npz, per_stage = {}, {}
    for k in range(K):
        xs = torch.cat([p[k] for p in pyrs], dim=0).double()      # (N,3,H,W)
        _, C, H, W = xs.shape
        mu = xs.mean(dim=0, keepdim=True)
        z = xs - mu
        centered = float((z ** 2).sum() / (N * C * H * W))         # Tr[Cov]/D
        P = torch.fft.fft2(z.float(), norm="ortho").abs().pow(2).mean(dim=(0, 1))
        floor = SPECTRAL_FLOOR_REL * float(P.max())
        n_floored = int((P < floor).sum())
        P = P.clamp_min(floor)
        spectral_npz[f"stage{k}"] = P.numpy()
        per_stage[str(k)] = dict(
            resolution=[H, W],
            spectral=dict(mean_power=float(P.mean()),
                          max_power=float(P.max()),
                          floor=floor, n_floored_bins=n_floored,
                          consistency_vs_centered_trace=float(P.mean()) / centered))
        print(f"[measure] stage {k} ({H}x{W}): spec_mean={float(P.mean()):.4f} "
              f"(=Tr[Cov]/D check: {float(P.mean())/centered:.4f})")
    incumbent = json.load(open(os.path.join(
        HERE, "results", "alg4", "s2_meas.json")))["table"]
    stats = {
        "CAVEAT": "DEMO_ONLY / LEAKAGE: spectral measured on the 6 non-junco "
                  "demo images (eval family, not a training set); pooled_junco "
                  "is the junco-only incumbent (maximal leakage).",
        "calibration_images": names,
        "n_images": N,
        "formulas": {
            "pooled_junco": "incumbent: junco-only pooled E[x^2]-E[x]^2 "
                            "(results/alg4/s2_meas.json)",
            "spectral": "S_k(w) = mean_{i,c} |F z_{i,c}(w)|^2 (fft2 norm=ortho), "
                        f"floored at {SPECTRAL_FLOOR_REL} * max(S_k)",
        },
        "pooled_junco": {k: float(v) for k, v in incumbent.items()},
        "per_stage": per_stage,
    }
    json.dump(stats, open(_stats_path(), "w"), indent=1)
    np.savez(_spectral_path(), **spectral_npz)
    print(f"[measure] -> {_stats_path()} + spectral_power.npz")
    return 0


# ─────────────────────── S construction per arm ────────────────────────────
def _build_s_ops(arm, stats, num_stages):
    """Return s2_fn(k, sigma) for the arm. pooled_junco returns plain floats
    (the original scalar path); spectral returns cached SOperator instances."""
    if arm == "pooled_junco":
        tab = stats["pooled_junco"]
        return lambda k, sig: float(tab[str(k)])
    if arm == "spectral":
        npz = np.load(_spectral_path())
        per = stats["per_stage"]
        ops = {k: SpectralSOp(torch.from_numpy(npz[f"stage{k}"]),
                              meta=dict(floor=per[str(k)]["spectral"]["floor"]))
               for k in range(num_stages)}
        return lambda k, sig: ops[k]
    raise KeyError(f"arm {arm!r} not in {ALL_ARMS}")


# ───────────────────────────── onestep ─────────────────────────────────────
class _Capture:
    """Duck-typed diag recorder: stores frame constants and the inner-0
    (x1_in, x1_hat) state at the target frames. Read-only, RNG-free."""

    def __init__(self, targets):
        self.targets = set(targets)
        self.setup = {}
        self.state = {}

    def on_frame_setup(self, **kw):
        if kw["frame"] in self.targets:
            self.setup[kw["frame"]] = {k: kw[k] for k in
                                       ("stage", "step", "tau", "s_k", "e_k",
                                        "sigma_tau", "gamma2", "s2", "eta")}
            self.setup[kw["frame"]]["A_fn"] = kw["A_fn"]
            self.setup[kw["frame"]]["AT_fn"] = kw["AT_fn"]

    def on_inner(self, **kw):
        if kw["frame"] in self.targets and kw["inner"] == 0:
            self.state[kw["frame"]] = dict(
                x1_in=kw["x1_in"].detach().clone(),
                x1_hat=kw["x1_hat"].detach().clone())

    def on_frame_end(self, **kw):
        pass


def cmd_onestep(_args):
    _init_globals()
    device = "cuda:0"
    config, model, gamma2_tab, S = _load_sampling(device)
    stats = json.load(open(_stats_path()))
    K = int(config.scheduler.num_stages)
    s2_fn_base = main4.make_s2_fn(main4.S_PRIOR, K)     # incumbent (capture run)

    cap = _Capture(ONESTEP_FRAMES)
    t0 = time.time()
    main4._run_once(model, config, S, device, s2_fn=s2_fn_base,
                    gamma2_tab=gamma2_tab, seed=main4.ALG["seed"], diag=cap)
    print(f"[onestep] capture run done [{time.time()-t0:.0f}s]; "
          f"frames {sorted(cap.state)}")

    hole = S["hole"].to(device).float()
    gt = S["gt"].to(device)
    y = S["y"].to(device)
    cg_tol = float(main4.SAMPLER_KW["cg_tol"])
    cg_cap = int(main4.SAMPLER_KW["cg_max_iter"])

    def mseh(a, b):
        return U.mse_masked(a, b, hole)

    rows = []
    for arm in ALL_ARMS:
        s2_fn = _build_s_ops(arm, stats, K)
        for f in ONESTEP_FRAMES:
            c = cap.setup[f]
            st = cap.state[f]
            x1_in = st["x1_in"].to(device)
            x1_hat = st["x1_hat"].to(device)
            s_ret = s2_fn(c["stage"], c["sigma_tau"])
            s_op = s_ret if isinstance(s_ret, SOperator) else ScalarSOp(float(s_ret))
            M_den, Cinv, inv_e2, _, _ = make_M_tau_den(
                c["A_fn"], c["AT_fn"], c["eta"], c["sigma_tau"], c["tau"],
                c["s_k"], c["e_k"], None, s_op)
            M_inv = make_jacobi_precond(M_den, x1_in.shape, device)
            mse_hat = mseh(x1_hat, gt)
            outs = []
            for d in range(ONESTEP_DRAWS):
                g = torch.Generator(device="cpu").manual_seed(ONESTEP_SEED0 + d)
                xi_y = torch.randn(y.shape, generator=g).to(device)
                xi_h = torch.randn(x1_in.shape, generator=g).to(device)
                xi_s = torch.randn(x1_in.shape, generator=g).to(device)
                b_tilde = (inv_e2 * c["AT_fn"](y) + Cinv(x1_hat)
                           + (1.0 / c["eta"]) * c["AT_fn"](xi_y)
                           + (1.0 / c["sigma_tau"]) * U.apply_H_tau(
                               xi_h, c["tau"], c["s_k"], c["e_k"], None)
                           + s_op.apply_S_inv_sqrt(xi_s))
                x1_out, it, rel = pcg_solve(M_den, b_tilde, M_inv,
                                            x0=x1_in.clone(), tol=cg_tol,
                                            max_iter=cg_cap)
                outs.append(x1_out)
                rows.append(dict(
                    arm=arm, frame=f, tau=c["tau"], sigma_tau=c["sigma_tau"],
                    draw=d, mse_hole_x1hat=mse_hat,
                    mse_hole_out=mseh(x1_out, gt),
                    delta_block1=mseh(x1_out, gt) - mse_hat,
                    inj_hole=mseh(x1_out, x1_hat),
                    cg_iters=it, cg_resid=rel,
                    s2_scalar_equiv=float(s_op.scalar_equiv)))
            stack = torch.stack(outs)
            spread = float((stack.std(dim=0, unbiased=True) * hole).sum()
                           / hole.sum() / stack.shape[2])
            for r in rows[-ONESTEP_DRAWS:]:
                r["hole_spread"] = spread
            m = [r["mse_hole_out"] for r in rows[-ONESTEP_DRAWS:]]
            print(f"[onestep] {arm:<14} f{f}: hat={mse_hat:.4f} "
                  f"out={np.mean(m):.4f}±{np.std(m):.4f} "
                  f"inj={np.mean([r['inj_hole'] for r in rows[-ONESTEP_DRAWS:]]):.4f} "
                  f"spread={spread:.4f}", flush=True)
    p = os.path.join(OUT, "one_step_metrics.csv")
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[onestep] -> {p}")
    return 0


# ─────────────────────── trajectory + spread ───────────────────────────────
def _run_arm(arm, seed, config, model, gamma2_tab, S, device, diag=None):
    stats = json.load(open(_stats_path()))
    K = int(config.scheduler.num_stages)
    s2_fn = _build_s_ops(arm, stats, K)
    return main4._run_once(model, config, S, device, s2_fn=s2_fn,
                           gamma2_tab=gamma2_tab, seed=seed, diag=diag)


def cmd_trajectory(args):
    _init_globals()
    device = "cuda:0"
    config, model, gamma2_tab, S = _load_sampling(device)
    arms = args.arms.split(",")
    combined = []
    for arm in arms:
        d = os.path.join(OUT, f"traj_{arm}")
        os.makedirs(d, exist_ok=True)
        diag = alg4_diag.Alg4Diagnostics(gt_full=S["gt"], hole_full=S["hole"])
        t0 = time.time()
        x1, rows, _ = _run_arm(arm, main4.ALG["seed"], config, model,
                               gamma2_tab, S, device, diag=diag)
        diag.write_csv(d)
        np.save(os.path.join(d, f"final_x1_seed{main4.ALG['seed']}.npy"),
                x1.detach().cpu().numpy())
        for r in diag.frames:
            combined.append(dict(r, arm=arm))
        last = diag.frames[-1]
        print(f"[trajectory] {arm:<14} hole={float(last['hole_full']):.4f} "
              f"full={float(last['mse_full']):.4f} "
              f"obs={float(last['obs_full']):.4f} [{time.time()-t0:.0f}s]",
              flush=True)
    p = os.path.join(OUT, "trajectory_metrics.csv")
    fields = list(dict.fromkeys(k for r in combined for k in r))
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
        w.writeheader(); w.writerows(combined)
    print(f"[trajectory] -> {p}")
    return 0


def cmd_spread(args):
    _init_globals()
    device = "cuda:0"
    config, model, gamma2_tab, S = _load_sampling(device)
    hole = S["hole"].to(device).float()
    gt = S["gt"].to(device)
    rows = []
    for arm in args.arms.split(","):
        d = os.path.join(OUT, f"traj_{arm}")
        os.makedirs(d, exist_ok=True)
        finals = []
        for seed in SPREAD_SEEDS:
            p = os.path.join(d, f"final_x1_seed{seed}.npy")
            if os.path.exists(p):
                finals.append(torch.from_numpy(np.load(p)).to(device))
                continue
            x1, _, _ = _run_arm(arm, seed, config, model, gamma2_tab, S, device)
            np.save(p, x1.detach().cpu().numpy())
            finals.append(x1)
            print(f"[spread] {arm} seed {seed} done", flush=True)
        stack = torch.stack([f.to(device) for f in finals])
        spread = float((stack.std(dim=0, unbiased=True) * hole).sum()
                       / hole.sum() / stack.shape[2])
        mses = [U.mse_masked(f, gt, hole) for f in finals]
        rows.append(dict(arm=arm, n_seeds=len(finals), hole_spread=spread,
                         mse_hole_mean=float(np.mean(mses)),
                         mse_hole_std=float(np.std(mses))))
        print(f"[spread] {arm:<14} spread={spread:.4f} "
              f"mse={np.mean(mses):.4f}±{np.std(mses):.4f}", flush=True)
    p = os.path.join(OUT, "spread_metrics.csv")
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[spread] -> {p}")
    return 0


# ───────────────────────────── analyze ─────────────────────────────────────
PALETTE = ["#2a78d6", "#eb6834"]     # pooled_junco, spectral


def cmd_analyze(_args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _init_globals()
    stats = json.load(open(_stats_path()))

    geom = list(csv.DictReader(open(os.path.join(
        OUT, "traj_pooled_junco", "precision_terms.csv"))))
    npz = np.load(_spectral_path())
    xrows = []
    for r in geom:
        f, k = int(r["frame"]), int(r["stage"])
        sig = float(r["sigma_tau"])
        if sig <= 0:
            continue
        d_range = float(r["h_range"]) ** 2 / sig ** 2
        d_ker = float(r["h_ker"]) ** 2 / sig ** 2
        s2 = stats["pooled_junco"][str(k)]
        P = npz[f"stage{k}"]
        xrows.append(dict(
            frame=f, stage=k, tau=float(r["tau"]), sigma_tau=sig,
            data_prec_range=d_range, data_prec_ker=d_ker,
            ratio_ker_pooled_junco=d_ker * s2,          # >1 = data dominates
            spectral_frac_data_dom=float((P > 1.0 / max(d_ker, 1e-30)).mean())))
    p = os.path.join(OUT, "precision_crossover.csv")
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(xrows[0].keys()))
        w.writeheader(); w.writerows(xrows)
    cross = [r for r in xrows if r["ratio_ker_pooled_junco"] >= 1.0]
    if cross:
        first = min(cross, key=lambda r: r["frame"])
        print(f"[analyze] pooled_junco ker crossover: f{first['frame']} "
              f"(stage {first['stage']}, tau={first['tau']:.3f})")

    ones = list(csv.DictReader(open(os.path.join(OUT, "one_step_metrics.csv"))))
    arms = [a for a in ALL_ARMS
            if any(r["arm"] == a for r in ones)]
    frames = sorted({int(r["frame"]) for r in ones})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    width = 0.8 / max(len(arms), 1)
    for ai, arm in enumerate(arms):
        col = PALETTE[ai % len(PALETTE)]
        dmeans, derrs, imeans = [], [], []
        for f in frames:
            sel = [float(r["delta_block1"]) for r in ones
                   if r["arm"] == arm and int(r["frame"]) == f]
            inj = [float(r["inj_hole"]) for r in ones
                   if r["arm"] == arm and int(r["frame"]) == f]
            dmeans.append(np.mean(sel)); derrs.append(np.std(sel))
            imeans.append(np.mean(inj))
        x = np.arange(len(frames)) + (ai - len(arms) / 2 + 0.5) * width
        axes[0].bar(x, dmeans, width * 0.9, yerr=derrs, color=col,
                    label=arm, error_kw=dict(lw=0.8))
        axes[1].bar(x, imeans, width * 0.9, color=col, label=arm)
    for ax, title in ((axes[0], "Block-1 ΔMSE_hole (out − x̂₁), 16 draws"),
                      (axes[1], "Block-1 injection ‖x₁ᵒᵘᵗ−x̂₁‖²_hole")):
        ax.set_xticks(np.arange(len(frames)))
        ax.set_xticklabels([f"f{f}" for f in frames])
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.25); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "one_step_comparison.png"), dpi=160)

    traj = list(csv.DictReader(open(os.path.join(OUT, "trajectory_metrics.csv"))))
    tarms = list(dict.fromkeys(r["arm"] for r in traj))
    fig2, ax = plt.subplots(figsize=(9, 4.2))
    for ai, arm in enumerate(tarms):
        sel = [r for r in traj if r["arm"] == arm]
        ax.plot([int(r["frame"]) for r in sel],
                [float(r["mse_hole"]) for r in sel],
                color=PALETTE[ai % len(PALETTE)], lw=1.6, label=arm)
    for b in (10, 20, 30):
        ax.axvline(b, color="#999", lw=0.7, ls=":")
    ax.set_xlabel("frame"); ax.set_ylabel("hole MSE")
    ax.set_title("box/junco trajectory by S construction (S_it=2)", fontsize=10)
    ax.grid(alpha=0.25); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUT, "trajectory_comparison.png"), dpi=160)

    if os.path.exists(os.path.join(OUT, "spread_metrics.csv")):
        spr = list(csv.DictReader(open(os.path.join(OUT, "spread_metrics.csv"))))
        fig3, ax = plt.subplots(figsize=(5.4, 4.4))
        for ai, r in enumerate(spr):
            ax.scatter(float(r["hole_spread"]), float(r["mse_hole_mean"]),
                       s=55, color=PALETTE[ai % len(PALETTE)], zorder=3)
            ax.annotate(r["arm"], (float(r["hole_spread"]),
                                   float(r["mse_hole_mean"])),
                        textcoords="offset points", xytext=(6, 4), fontsize=8)
        ax.set_xlabel("hole spread (pixelwise std over seeds)")
        ax.set_ylabel("final hole MSE (mean over seeds)")
        ax.set_title("posterior spread vs reconstruction", fontsize=10)
        ax.margins(x=0.25, y=0.15)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))
        ax.grid(alpha=0.25); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig3.tight_layout()
        fig3.savefig(os.path.join(OUT, "spread_vs_mse.png"), dpi=160)

    shutil.copy(os.path.join(HERE, "config_alg4.json"),
                os.path.join(OUT, "configs", "config_alg4.json"))
    json.dump(dict(arms=list(ALL_ARMS), onestep_frames=list(ONESTEP_FRAMES),
                   onestep_draws=ONESTEP_DRAWS, onestep_seed0=ONESTEP_SEED0,
                   spread_seeds=list(SPREAD_SEEDS),
                   spectral_floor_rel=SPECTRAL_FLOOR_REL,
                   eval_image=EVAL_IMAGE),
              open(os.path.join(OUT, "configs", "s_prior_config.json"), "w"),
              indent=1)
    print("[analyze] wrote precision_crossover.csv + plots + configs/")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["measure", "onestep", "trajectory",
                                    "spread", "analyze"])
    ap.add_argument("--arms", default=",".join(ALL_ARMS))
    args = ap.parse_args()
    return {"measure": cmd_measure, "onestep": cmd_onestep,
            "trajectory": cmd_trajectory, "spread": cmd_spread,
            "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
