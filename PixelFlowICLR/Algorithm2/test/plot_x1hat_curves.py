#!/usr/bin/env python
"""Loss curves for the 7.2 one-step test: x0_hat, x_tau and x1_hat vs tau.

Reads score_x1hat_<image>.csv and draws, per stage and on a log axis:

  ||x0_hat - x0||^2   the score error itself (this is what defines gamma^2)
  ||x_tau  - x1_gt||^2 how far the state is from the clean image to begin with
  ||x1_hat - x1_gt||^2 the implied clean image, i.e. what that score error costs

The exact-displacement curve is drawn too: it is the machine-precision floor and
shows the identity holding across the whole grid. A second figure splits the
model's x1_hat into the box hole and the observed region.
"""

import argparse
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
A2 = os.path.dirname(HERE)

import matplotlib                                              # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="junco")
    ap.add_argument("--dir", default=os.path.join(A2, "test/results/score_x1hat"))
    cli = ap.parse_args()

    path = os.path.join(cli.dir, f"score_x1hat_{cli.image}.csv")
    with open(path) as f:
        rows = [dict(r) for r in csv.DictReader(f)]
    stages = sorted({int(r["stage"]) for r in rows})

    def series(si, key):
        rs = [r for r in rows if int(r["stage"]) == si]
        return ([float(r["tau"]) for r in rs], [float(r[key]) for r in rs])

    # ── figure 1: the three losses ────────────────────────────────────────
    fig, axes = plt.subplots(1, len(stages), figsize=(4.2 * len(stages), 3.9),
                             sharey=True)
    axes = [axes] if len(stages) == 1 else list(axes)
    for ax, si in zip(axes, stages):
        for key, lab, style in [
                ("x_tau_mse", r"$\|x_\tau - x_1\|^2$  (state)", "o-"),
                ("x0err_model", r"$\|\hat x_0 - x_0\|^2$  (score)", "s-"),
                ("x1_model", r"$\|\hat x_1 - x_1\|^2$  (implied clean)", "^-"),
                ("x1_exact", r"$\|\hat x_1 - x_1\|^2$  exact $d_\tau$", "--")]:
            t, v = series(si, key)
            ax.semilogy(t, v, style, ms=4, lw=1.4, label=lab)
        ax.set_title(f"stage {si}")
        ax.set_xlabel(r"$\tau$")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("MSE (log)")
    axes[-1].legend(fontsize=7.5, loc="lower left")
    fig.suptitle(f"7.2 one-step losses — {cli.image}", fontsize=12)
    fig.tight_layout()
    out1 = os.path.join(cli.dir, f"curves_{cli.image}.png")
    fig.savefig(out1, dpi=130); plt.close(fig)

    # ── figure 2: x1_hat split by region ──────────────────────────────────
    fig, axes = plt.subplots(1, len(stages), figsize=(4.2 * len(stages), 3.9),
                             sharey=True)
    axes = [axes] if len(stages) == 1 else list(axes)
    for ax, si in zip(axes, stages):
        for key, lab, style in [
                ("x1_model_hole", r"$\hat x_1$ hole (unobservable)", "o-"),
                ("x1_model_obs", r"$\hat x_1$ observed", "s-"),
                ("x1_model", r"$\hat x_1$ full", "^--")]:
            t, v = series(si, key)
            ax.semilogy(t, v, style, ms=4, lw=1.4, label=lab)
        ax.set_title(f"stage {si}")
        ax.set_xlabel(r"$\tau$")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("MSE (log)")
    axes[-1].legend(fontsize=8, loc="lower left")
    fig.suptitle(f"7.2 implied clean image, by region — {cli.image}", fontsize=12)
    fig.tight_layout()
    out2 = os.path.join(cli.dir, f"curves_region_{cli.image}.png")
    fig.savefig(out2, dpi=130); plt.close(fig)

    print(f"[done] {out1}\n       {out2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
