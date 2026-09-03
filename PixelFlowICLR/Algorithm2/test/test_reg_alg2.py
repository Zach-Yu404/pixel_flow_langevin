#!/usr/bin/env python
"""Tweedie pseudo-observation anchor for Algorithm 2 (run_posterior_reg_sampling_alg2).

Hypothesis under test: Algorithm 2's hole MEAN may already be reasonable, and
what breaks the reconstruction is the SAMPLE VARIANCE along weakly constrained
(nullspace) directions. Treating the network's clean estimate as one Gaussian
pseudo-observation puts a floor under that precision,

    M_tau = M_tau^(0) + lambda P,     b_tau = b_tau^(0) + lambda P x1_model

with P the measurement nullspace projection (the hole, for inpainting).

    --check   P is an orthogonal projection into ker(A_k); the scalar variance
              prediction Var = 1/(m + lambda) at representative (stage, tau)
    --run     lambda sweep 0/5/10/25/50/100/200/400/800/1600 with P = I, plus
              lambda=25 with P = the measurement-nullspace projection
    --report  comparison.csv + trajectories.png

Every lambda draws xi_a whether or not it is used, so all arms consume the same
count/shape/order of random numbers -- the sweep is a paired-noise comparison.

    PYTHONHASHSEED=0 python test_reg_alg2.py --check
    PYTHONHASHSEED=0 python test_reg_alg2.py --run lambda_25
"""

import argparse
import csv
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ALG2 = os.path.dirname(HERE)
if ALG2 not in sys.path:
    sys.path.insert(0, ALG2)

import torch                                                    # noqa: E402
import torch.nn.functional as F                                 # noqa: E402
import matplotlib                                               # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                 # noqa: E402

import utils                                                    # noqa: E402
from utils import (apply_G, compute_sigma_tau, make_M_tau, mse_masked,
                   _make_anchor_P)                            # noqa: E402
from test_wv_alg2 import build, img, SCHED                     # noqa: E402

OUT = os.path.join(HERE, "results_reg_alg2")
LAMBDAS = [0.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0]
# lambda_<N>_null (the nullspace P) is still selectable by name but is no
# longer part of the default sweep -- it never beat P = I; see report.md.
ARMS = [f"lambda_{int(l)}" for l in LAMBDAS] + ["ref_0", "ref_25"]


def arm_spec(arm):
    """arm name -> (lambda, kwargs overrides).

      lambda_<N>            P = I  (the original 74a0c39 anchor; the default)
      lambda_<N>_null       P = the hole projection (kept, not swept)
      lambda_<N>_nob1       block1_noise=False   (drop xi_y / xi_h)
      lambda_<N>_nob2       block2_noise=False   (drop sqrt(h0) xi_0)
      lambda_<N>_nob12      both off
      lambda_<N>_h0-<x>     h0 override
      lambda_<N>_S-<n>      num_langevin override
      ref_<N>               the verbatim 74a0c39 sampler
    """
    parts = arm.split("_")
    lam = float(parts[1])
    ov = dict(anchor_P="identity")
    for p in parts[2:]:
        if p == "null":
            ov["anchor_P"] = "nullspace"
        elif p == "nob1":
            ov["block1_noise"] = False
        elif p == "nob2":
            ov["block2_noise"] = False
        elif p == "nob12":
            ov["block1_noise"] = ov["block2_noise"] = False
        elif p.startswith("h0-"):
            ov["h0"] = float(p[3:])
        elif p.startswith("S-"):
            ov["num_langevin"] = int(p[2:])
        else:
            raise ValueError(f"unknown arm suffix {p!r} in {arm!r}")
    return lam, ov


# ══════════════════════════════════ checks ══════════════════════════════════
def check(device="cuda:0"):
    cfg, demo, gt, op, mask, y, mkA, model, mcfg, gamma2_tab, spec = build(device)
    eta = float(spec["sigma_n"])
    rows, ok = [], True
    print("== P: orthogonal projection (hard) + ker(A_k) containment (reported) ==")
    for k, (s_k, e_k) in enumerate(SCHED):
        h = w = 256 // 2 ** (3 - k)
        P_fn, kind = _make_anchor_P(op, y, (1, 3, h, w), device, "nullspace")
        Ak, ATk = mkA(op, y, (1, 3, h, w), device)
        x = torch.randn(1, 3, h, w, device=device)
        z = torch.randn(1, 3, h, w, device=device)
        idem = float((P_fn(P_fn(x)) - P_fn(x)).abs().max())
        adj = abs(float((P_fn(x) * z).sum()) - float((x * P_fn(z)).sum()))
        adj /= max(float(x.norm() * z.norm()), 1e-12)
        leak = float(Ak(P_fn(x)).abs().max()) / max(float(x.abs().max()), 1e-12)
        frac = float(P_fn(torch.ones_like(x)).mean())
        p = idem < 1e-6 and adj < 1e-6           # projection: the hard requirement
        ok &= p
        rows.append(dict(check="P", stage=k, res=h, kind=kind, idem=idem,
                         selfadj=adj, ker_leak=leak, hole_frac=frac, passed=p))
        print(f"  [stage {k} {h}x{h}] P^2-P {idem:.2e}  |<Px,z>-<x,Pz>| {adj:.2e}  "
              f"{'PASS' if p else 'FAIL'}   max|A_k(Px)|/max|x| = {leak:.2e}  "
              f"hole frac {frac:.3f}")
    print("  NOTE A_k = A . U with an INTERPOLATING U, so a vector supported on the")
    print("       down-sampled hole is exactly in ker(A_k) only at stage 3 (no U).")
    print("       At stages 0-2 P is still an exact orthogonal projection onto the")
    print("       hole -- sqrt(lambda) P xi_a has covariance lambda P exactly -- but")
    print("       the anchored subspace leaks slightly outside ker(A_k).")

    print("== scalar variance prediction  Var = 1/(m + lambda)  by probe direction ==")
    print("   stage tau    probe          m=<u,M0 u>   lam=0 std   lam=25 std  lam=100 std")
    for k, (s_k, e_k) in enumerate(SCHED):
        h = w = 256 // 2 ** (3 - k)
        P_fn, _ = _make_anchor_P(op, y, (1, 3, h, w), device, "nullspace")
        Ak, ATk = mkA(op, y, (1, 3, h, w), device)
        for tau in (0.0, 0.999):
            sig = compute_sigma_tau(tau, s_k, e_k)
            M0, _, _, _ = make_M_tau(Ak, ATk, eta, sig, tau, s_k, e_k, k,
                                     (1, 3, h, w), device, 0.0)
            base = torch.randn(1, 3, h, w, device=device)
            Gb = apply_G(base, stage_idx=k)
            for name, u in (("hole", P_fn(base)), ("hole&range(G)", P_fn(Gb)),
                            ("hole&ker(G)", P_fn(base - Gb))):
                if float(u.norm()) < 1e-8:
                    continue
                u = u / u.norm()
                m = float((u * M0(u)).sum())
                s0 = m ** -0.5 if m > 0 else float("inf")
                rows.append(dict(check="var", stage=k, res=h, tau=tau, probe=name,
                                 m=m, std_0=s0, std_25=(m + 25.0) ** -0.5,
                                 std_100=(m + 100.0) ** -0.5, passed=True))
                print(f"     {k}   {tau:.3f}  {name:14s} {m:11.4g}  {s0:10.4g}  "
                      f"{(m + 25.0) ** -0.5:11.4g}  {(m + 100.0) ** -0.5:10.4g}")
    os.makedirs(OUT, exist_ok=True)
    keys = sorted({kk for r in rows for kk in r})
    with open(os.path.join(OUT, "checks.csv"), "w", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=keys); w_.writeheader()
        for r in rows:
            w_.writerow({kk: r.get(kk, "") for kk in keys})
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


# ═══════════════════════════════════ runs ═══════════════════════════════════
def run(arm, device="cuda:0"):
    lam, ov = arm_spec(arm)
    cfg, demo, gt, op, mask, y, mkA, model, mcfg, gamma2_tab, spec = build(device)
    print(f"[{arm}] model loaded (lambda={lam}, {ov})", flush=True)
    kw = {**cfg["sampler_kw"], **spec.get("kw", {})}
    alg = cfg["algorithm"]
    eta = float(spec["sigma_n"])
    # h0 / num_langevin are ordinary sampler kwargs, not anchor kwargs: take them
    # out of ov so they do not collide with the explicit ones below.
    if "num_langevin" in ov:
        kw["num_langevin"] = ov.pop("num_langevin")
    h0 = ov.pop("h0", alg["h0"])
    if arm.startswith("ref_"):
        import _deprecated_anchor_ref as ref
        kw2 = dict(kw); kw2["class_label"] = int(demo["class_idx"])
        x1, rows, traj = ref.sample_alg2(
            model, mcfg, gt, y, dict(mkA=mkA, op=op, y=y), kw2, eta, gamma2_tab,
            device, seed=alg["seed"], record=True, h0=h0, anchor=lam)
        traj = [(t[0], t[1], t[2], None, None, None) for t in traj]
        for r in rows:
            r["anchor_P"] = "identity (verbatim 74a0c39)"
        return finish(arm, lam, x1, rows, traj, gt, y, op, mask, mkA, mcfg, device)
    x1, rows, traj = utils.run_posterior_reg_sampling_alg2(
        model, mcfg, gt, y, op, eta, device,
        anchor_lambda=lam, **ov,
        gamma2_tab=gamma2_tab, make_Ak_fns_fn=mkA, h0=h0,
        sigma_min=alg["sigma_min"], ridge_rel=alg["ridge_rel"],
        cg_max_iter_l14=alg["cg_max_iter_l14"],
        terminal_replace_weight=float(spec["terminal_replace_weight"]),
        class_label=int(demo["class_idx"]), seed=alg["seed"],
        record_trajectory=True,
        **{k: v for k, v in kw.items() if k not in ("class_label", "seed")})
    return finish(arm, lam, x1, rows, traj, gt, y, op, mask, mkA, mcfg, device)


def finish(arm, lam, x1, rows, traj, gt, y, op, mask, mkA, mcfg, device):
    hole = (1.0 - mask).to(device).float()
    pyr = utils.base.gt_stage_pyramid(gt, int(mcfg.scheduler.num_stages))
    out, keep = [], {}
    for gi, (r, t) in enumerate(zip(rows, traj)):
        k = r["stage"]
        xk = t[1].unsqueeze(0).to(device)
        m_k = F.interpolate(hole, size=xk.shape[-2:], mode="nearest")
        Ak, _ = mkA(op, y, tuple(xk.shape), device)
        row = dict(arm=arm, lam=lam, P=r["anchor_P"], gstep=gi, stage=k,
                   step=r["step"], tau=r["tau"], sigma_tau=r["sigma_tau"],
                   mse_total=float(((xk - pyr[k]) ** 2).mean()),
                   mse_hole=mse_masked(xk, pyr[k], m_k),
                   mse_vis=mse_masked(xk, pyr[k], 1.0 - m_k),
                   meas_resid=float(((Ak(xk) - y) ** 2).mean()))
        if t[3] is not None:
            row["x1model_hole"] = mse_masked(t[3].unsqueeze(0).to(device), pyr[k], m_k)
        if t[4] is not None:
            mu = t[4].unsqueeze(0).to(device)
            row["mu_hole"] = mse_masked(mu, pyr[k], m_k)
            row["mu_vis"] = mse_masked(mu, pyr[k], 1.0 - m_k)
            row["noise_rms_hole"] = math.sqrt(mse_masked(xk, mu, m_k))
        out.append(row)
        keep.setdefault(k, None)
        # sigma_tau < sigma_min skips Block 1 entirely, so the last step of a
        # stage can have no mu / x1_model (stage 3 tau=0.999: sigma_tau=4e-4).
        # Panel the last step that actually solved, else the last step at all.
        if t[4] is not None or keep[k] is None:
            keep[k] = (k, r["step"], r["tau"], t, xk, m_k, row)

    for k in sorted(keep):
        if keep[k] is not None:
            kk, st, tau_, t_, xk_, mk_, row_ = keep[k]
            panel(arm, kk, st, tau_, pyr[kk], y, t_, xk_, mk_, row_)

    fin = dict(arm=arm, lam=lam, P=rows[-1]["anchor_P"],
               hole=mse_masked(x1, gt, hole), vis=mse_masked(x1, gt, 1.0 - hole),
               full=float(((x1 - gt) ** 2).mean()),
               meas_resid=float(((op(x1) - y) ** 2).mean()),
               # last step has sigma_tau < sigma_min (no mu), so take the
               # last step that actually has a Block-1 solve
               noise_rms_hole=next((r["noise_rms_hole"] for r in reversed(out)
                                    if "noise_rms_hole" in r), ""))
    for k in range(4):
        st = [r for r in out if r["stage"] == k]
        fin[f"stage{k}_hole"] = st[-1]["mse_hole"]
    d = os.path.join(OUT, arm)
    os.makedirs(d, exist_ok=True)
    keys = ["arm", "lam", "P", "gstep", "stage", "step", "tau"] + \
        sorted({kk for r in out for kk in r} -
               {"arm", "lam", "P", "gstep", "stage", "step", "tau"})
    with open(os.path.join(d, "trajectory.csv"), "w", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=keys); w_.writeheader()
        for r in out:
            w_.writerow({kk: r.get(kk, "") for kk in keys})
    json.dump(fin, open(os.path.join(d, "final.json"), "w"), indent=1)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    for A, (t_, ttl) in zip(ax, [(gt, "GT"), (y, "measurement y"),
                                 (x1, f"{arm}\nhole {fin['hole']:.4f} "
                                      f"vis {fin['vis']:.5f}")]):
        A.imshow(img(t_)); A.set_title(ttl, fontsize=9); A.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(d, "final.png"), dpi=140)
    plt.close(fig)
    print(f"[{arm}] FINAL {json.dumps(fin)}", flush=True)


def panel(arm, k, step, tau, gt_k, y, t, xk, m_k, row):
    items = [(gt_k, "GT (stage)"), (y, "measurement y"),
             (None if t[3] is None else t[3].unsqueeze(0),
              f"x1_model\nhole {row.get('x1model_hole', float('nan')):.4f}"),
             (None if t[4] is None else t[4].unsqueeze(0),
              f"Block-1 mean\nhole {row.get('mu_hole', float('nan')):.4f}"),
             (xk, f"sampled x1\nhole {row['mse_hole']:.4f}")]
    fig, ax = plt.subplots(1, len(items) + 1, figsize=(3.1 * (len(items) + 1), 3.4))
    for A, (t_, ttl) in zip(ax, items):
        if t_ is None:
            A.text(.5, .5, "n/a", ha="center", va="center")
        else:
            A.imshow(img(t_.to(xk.device)))
        A.set_title(ttl, fontsize=9); A.axis("off")
    err = (((xk - gt_k) ** 2).mean(1, keepdim=True) * m_k)[0, 0].detach().cpu()
    im = ax[-1].imshow(err, cmap="magma"); ax[-1].set_title("hole sq-err", fontsize=9)
    ax[-1].axis("off"); fig.colorbar(im, ax=ax[-1], fraction=0.046)
    fig.suptitle(f"{arm} — stage {k} step {step} tau={tau:.4f}", fontsize=11)
    fig.tight_layout()
    d = os.path.join(OUT, arm); os.makedirs(d, exist_ok=True)
    fig.savefig(os.path.join(d, f"stage{k}_step{step}.png"), dpi=130)
    plt.close(fig)


# ══════════════════════════════════ report ══════════════════════════════════
def report():
    fins, allrows = [], []
    for a in ARMS:
        p = os.path.join(OUT, a, "final.json")
        if not os.path.exists(p):
            continue
        fins.append(json.load(open(p)))
        allrows += list(csv.DictReader(open(os.path.join(OUT, a, "trajectory.csv"))))
    keys = ["arm", "lam", "P", "hole", "vis", "full", "meas_resid",
            "noise_rms_hole", "stage0_hole", "stage1_hole", "stage2_hole",
            "stage3_hole"]
    with open(os.path.join(OUT, "comparison.csv"), "w", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=keys); w_.writeheader()
        for d in fins:
            w_.writerow({k: d.get(k, "") for k in keys})
    print(f"{'arm':16s} {'hole':>10s} {'visible':>11s} {'full':>10s} "
          f"{'holeRMS':>9s}  stage-end hole")
    for d in fins:
        se = "  ".join(f"{d[f'stage{k}_hole']:.4f}" for k in range(4))
        rms = d.get("noise_rms_hole", "")
        rms = f"{float(rms):9.4f}" if rms not in ("", None) else " " * 9
        print(f"{d['arm']:16s} {d['hole']:10.6f} {d['vis']:11.7f} "
              f"{d['full']:10.6f} {rms}  {se}")

    def g(r, k):
        v = r.get(k, "")
        try:
            return float(v)
        except ValueError:
            return float("nan")
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    for a in ARMS:
        rs = [r for r in allrows if r["arm"] == a]
        if not rs:
            continue
        gs = [int(r["gstep"]) for r in rs]
        ax[0].semilogy(gs, [g(r, "mse_hole") for r in rs], marker=".", ms=2, label=a)
        ax[1].semilogy(gs, [g(r, "mu_hole") for r in rs], marker=".", ms=2, label=a)
        ax[2].semilogy(gs, [g(r, "noise_rms_hole") for r in rs], marker=".", ms=2,
                       label=a)
    for A, ttl in zip(ax, ["sampled x1 — HOLE MSE", "Block-1 mean — HOLE MSE",
                           "hole sampling noise rms(x1 - mu)"]):
        for b in (10, 20, 30):
            A.axvline(b, color="gray", lw=.6, alpha=.5)
        A.set_title(ttl, fontsize=10); A.grid(alpha=.3)
        A.legend(fontsize=7); A.set_xlabel("global step")
    fig.suptitle("Tweedie anchor lambda sweep — box_inpainting / junco / seed 42")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "trajectories.png"), dpi=140)
    print(f"[report] -> {OUT}")


def grid_report():
    """Scan OUT for every arm that finished and summarise the (lambda, h0, S) grid."""
    import glob
    rows = []
    for fp in sorted(glob.glob(os.path.join(OUT, "*", "final.json"))):
        d = json.load(open(fp))
        arm = os.path.basename(os.path.dirname(fp))
        _, ov = arm_spec(arm)
        rows.append(dict(arm=arm, lam=d["lam"], h0=ov.get("h0", ""),
                         S=ov.get("num_langevin", ""), hole=d["hole"],
                         vis=d["vis"], full=d["full"],
                         meas_resid=d["meas_resid"],
                         noise_rms_hole=d.get("noise_rms_hole", ""),
                         **{f"stage{k}_hole": d[f"stage{k}_hole"] for k in range(4)}))
    if not rows:
        print(f"[grid] nothing finished under {OUT}")
        return
    rows.sort(key=lambda r: (r["lam"], r["h0"] or 0, r["S"] or 0))
    keys = list(rows[0].keys())
    with open(os.path.join(OUT, "comparison.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    best = min(rows, key=lambda r: r["hole"])
    print(f"{'arm':34s} {'hole':>10s} {'visible':>11s} {'meas_res':>10s}")
    for r in rows:
        mark = "  <-- best" if r is best else ""
        print(f"{r['arm']:34s} {r['hole']:10.6f} {r['vis']:11.7f} "
              f"{r['meas_resid']:10.7f}{mark}")

    lams = sorted({r["lam"] for r in rows})
    h0s = sorted({r["h0"] for r in rows if r["h0"] != ""})
    Ss = sorted({r["S"] for r in rows if r["S"] != ""})
    if len(lams) > 1 and len(h0s) > 1 and len(Ss) > 1:
        fig, ax = plt.subplots(1, len(Ss), figsize=(5.2 * len(Ss), 4.4))
        ax = [ax] if len(Ss) == 1 else list(ax)
        vals = [r["hole"] for r in rows]
        vmin, vmax = min(vals), max(vals)
        for a, S in zip(ax, Ss):
            grid_v = [[next((r["hole"] for r in rows
                             if r["lam"] == L and r["h0"] == H and r["S"] == S),
                            float("nan")) for L in lams] for H in h0s]
            im = a.imshow(grid_v, cmap="viridis_r", vmin=vmin, vmax=vmax,
                          aspect="auto")
            a.set_xticks(range(len(lams))); a.set_xticklabels([f"{L:g}" for L in lams])
            a.set_yticks(range(len(h0s))); a.set_yticklabels([f"{H:g}" for H in h0s])
            a.set_xlabel("$\\lambda$"); a.set_ylabel("$h_0$")
            a.set_title(f"S = {S}", fontsize=11)
            for i, H in enumerate(h0s):
                for j, L in enumerate(lams):
                    v = grid_v[i][j]
                    if v == v:
                        a.text(j, i, f"{v:.4f}", ha="center", va="center",
                               fontsize=8,
                               color="white" if v > (vmin + vmax) / 2 else "black")
            fig.colorbar(im, ax=a, fraction=0.046)
        fig.suptitle("hole MSE over ($\\lambda$, $h_0$, S) — random terms kept "
                     "(box_inpainting / junco / seed 42)", fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "grid_hole.png"), dpi=140)
        print(f"[grid] -> {OUT}/comparison.csv, grid_hole.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--run", default=None, help=f"comma-separated: {','.join(ARMS)}")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--grid-report", action="store_true",
                    help="summarise every finished arm under --out")
    ap.add_argument("--out", default=None,
                    help="results directory (default results_reg_alg2/)")
    a = ap.parse_args()
    if a.out:
        global OUT
        OUT = a.out if os.path.isabs(a.out) else os.path.join(HERE, a.out)
    if not (a.check or a.run or a.report or a.grid_report):
        a.check = a.report = True
        a.run = ",".join(ARMS)
    os.makedirs(OUT, exist_ok=True)
    rc = 0
    if a.check:
        rc = check()
    if a.run:
        for arm in a.run.split(","):
            run(arm)
    if a.report:
        report()
    if a.grid_report:
        grid_report()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
