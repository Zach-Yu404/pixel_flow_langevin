#!/usr/bin/env python
"""Algorithm 4 diagnostics — recorder, metrics and plots, in one place.

Used by ``main4.py --mode diagnose`` and ``--mode diversity``. The recorder is
passed to ``run_posterior_sampling_alg4(diag=...)``; it is READ-ONLY and
RNG-FREE, so a run with diagnostics on is bit-identical to one with them off.
That is not a claim, it is checked: ``main4.py --mode diagnose`` reruns the
sampler with ``diag=None`` and compares.

Everything the sampler exposes per frame is derived here rather than inside the
sampler, so utils.py keeps only three one-line hooks.

Two conventions worth stating once.

**frame id.** A "frame" is one (stage, step) of the outer loop -- the same index
the trajectory PNGs are numbered by. With 4 stages x 10 ODE steps there are 40
frames, 0..39, and stage 3 starts at frame 30.

**Cross-stage comparability.** ``mse_x1`` inside the sampler is against
``pyr[stage]``, the GT at THAT stage's resolution, so it cannot be compared
across a stage boundary. Everything here also carries ``mse_full`` / ``psnr_full``
-- x1 upsampled to full resolution with the same nearest U^(1) the sampler's
stage transition uses, scored against the full-resolution GT. That is the curve
to read across the whole trajectory.

**PSNR** uses MAX = 2, because demo_runner normalises images to [-1, 1]
(CONSTRAINTS: the MAX=1-era PSNR numbers are never to be quoted).
"""

import csv
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402

from ms_posterior_sampling_article_version_final_utils import (  # noqa: E402
    apply_H_tau,
)
from utils import mse_masked, power_iter_norm                  # noqa: E402
from onestep_visual import to_img                              # noqa: E402

DATA_RANGE = 2.0            # images live in [-1, 1]


# ───────────────────────────────── metrics ────────────────────────────────
def mse(a, b):
    return float(((a - b) ** 2).mean())


def psnr(a, b):
    m = mse(a, b)
    return float("inf") if m <= 0 else 10.0 * math.log10(DATA_RANGE ** 2 / m)


def upsample_to(x, size):
    """The sampler's own stage transition (l.18) is nearest interpolation, so
    scoring an intermediate x1 at full resolution uses the same operator."""
    return x if tuple(x.shape[-2:]) == tuple(size) else F.interpolate(
        x, size=tuple(size), mode="nearest")


def h_tau_eigenvalues(tau, s_k, e_k, eff_si):
    """H_tau = (1-tau) s_k G + tau e_k I.

    G is an orthogonal projection (G^2 = G, G^T = G; main.py --mode verify V1),
    so H_tau has exactly two eigenvalues -- one on range(G), one on ker(G):

        h_range = (1-tau) s_k + tau e_k
        h_ker   = tau e_k

    ker(G) is non-trivial at every stage (the former stage-3 G = I bypass was
    removed 2026-08-24). Returns (h_range, h_ker); eff_si is kept for
    signature compatibility.
    """
    h_range = (1.0 - tau) * s_k + tau * e_k
    return h_range, tau * e_k


class Alg4Diagnostics:
    """Read-only recorder. Consumes no randomness and mutates no sampler state.

    ``capture_frames`` selects the frames whose tensors are kept for the
    montage and for the block-by-block error decomposition images.
    """

    def __init__(self, gt_full, hole_full=None, capture_frames=(),
                 measure_precision=True):
        self.gt_full = gt_full
        self.hole_full = hole_full
        self.capture_frames = set(int(f) for f in capture_frames)
        self.measure_precision = measure_precision
        self.frames = []          # one row per (stage, step)
        self.inner = []           # one row per (stage, step, inner iteration)
        self.precision = []       # one row per frame: the Block-1 precision terms
        self.captured = {}        # frame -> dict of cpu tensors

    # ---- hook 1: once per frame, right after M_tau^den is built ----------
    def on_frame_setup(self, *, frame, stage, step, tau, s_k, e_k, sigma_tau,
                       gamma2, s2, eta, eff_si, A_fn, AT_fn, shape, device):
        h_range, h_ker = h_tau_eigenvalues(tau, s_k, e_k, eff_si)
        inv_s2 = 1.0 / sigma_tau ** 2
        inv_S = 1.0 / s2
        rec = dict(
            frame=frame, stage=stage, step=step, tau=tau,
            sigma_tau=sigma_tau, gamma2=gamma2, s2=s2,
            h_range=h_range, h_ker=h_ker,
            h_over_sigma=h_range / sigma_tau,
            h2_over_sigma2=h_range ** 2 * inv_s2,
            inv_s2_prior=inv_S,
            # C^-1 = H^T H / sigma^2 + S^-1, eigenvalue by eigenvalue (12)
            prec_prior_range=h_range ** 2 * inv_s2 + inv_S,
            prec_prior_ker=h_ker ** 2 * inv_s2 + inv_S,
        )
        if self.measure_precision:
            # ||A^T A|| / eta^2, matrix-free. power_iter_norm carries its own
            # CPU generator, so this does not touch the sampler's noise stream.
            lam = power_iter_norm(lambda x: AT_fn(A_fn(x)), shape, device,
                                  iters=20, seed=0)
            rec["prec_data_max"] = lam / eta ** 2
            rec["data_over_prior"] = rec["prec_data_max"] / rec["prec_prior_range"]
        # For box inpainting A_k = mask . U, so inside the hole the data term
        # contributes exactly zero and prec_prior_range is the ONLY precision
        # acting there. That is the number to read for the hole.
        self.precision.append(rec)

    # ---- hook 2: once per inner iteration --------------------------------
    def on_inner(self, *, frame, stage, step, inner, tau, sigma_tau,
                 x1_in, x1_hat, x1_out, x_tau, v, gt_k, hole_k,
                 s_k, e_k, eff_si):
        row = dict(frame=frame, stage=stage, step=step, inner=inner, tau=tau,
                   sigma_tau=sigma_tau,
                   mse_x1_in=mse(x1_in, gt_k),
                   mse_x1_hat=mse(x1_hat, gt_k),
                   mse_x1_out=mse(x1_out, gt_k))
        # Did Block 1 improve on the endpoint it was handed, or spoil it?
        row["block1_delta"] = row["mse_x1_out"] - row["mse_x1_hat"]
        # Did the endpoint improve on the state that produced it?
        row["endpoint_delta"] = row["mse_x1_hat"] - row["mse_x1_in"]
        if hole_k is not None:
            row["hole_x1_in"] = mse_masked(x1_in, gt_k, hole_k)
            row["hole_x1_hat"] = mse_masked(x1_hat, gt_k, hole_k)
            row["hole_x1_out"] = mse_masked(x1_out, gt_k, hole_k)
            row["obs_x1_out"] = mse_masked(x1_out, gt_k, 1.0 - hole_k)
        self.inner.append(row)
        # Only the FIRST inner iteration is kept, so the montage shows what the
        # frame was handed and what one clean-endpoint -> Block 1 -> Block 2
        # pass did with it. The frame's end state is captured separately by
        # on_frame_end, which is what the error column shows.
        if frame in self.capture_frames and inner == 0:
            self.captured.setdefault(frame, {}).update(
                x1_in=x1_in[0].detach().cpu(),
                x1_hat=x1_hat[0].detach().cpu(),
                x1_out=x1_out[0].detach().cpu(),
                x_tau=x_tau[0].detach().cpu())
        if frame in self.capture_frames:
            # ...and the LAST inner iteration's clean endpoint (overwritten each
            # s): the natural denoised output of the frame, needed to separate
            # "the estimate is bad" from "the draw carries injected variance".
            self.captured.setdefault(frame, {})["x1_hat_last"] = \
                x1_hat[0].detach().cpu()

    # ---- hook 3: once per frame, after the state leaves the coordinates --
    def on_frame_end(self, *, frame, row, x1, x0, x1_hat, gt_k, hole_k,
                     tau, sigma_tau, s_k, e_k, eff_si):
        full = self.gt_full.shape[-2:]
        x1_up = upsample_to(x1, full)
        out = dict(row)
        out["mse_full"] = mse(x1_up, self.gt_full)
        out["psnr_full"] = psnr(x1_up, self.gt_full)
        out["x0_sq_over_n"] = float((x0 ** 2).mean())
        if self.hole_full is not None:
            out["hole_full"] = mse_masked(x1_up, self.gt_full, self.hole_full)
            out["obs_full"] = mse_masked(x1_up, self.gt_full,
                                         1.0 - self.hole_full)
        self.frames.append(out)
        if frame in self.capture_frames:
            self.captured.setdefault(frame, {}).update(
                x1_end=x1[0].detach().cpu(), x1_end_up=x1_up[0].detach().cpu())

    # ---- output ----------------------------------------------------------
    def write_csv(self, out_dir):
        paths = {}
        for name, rows in (("trajectory_metrics", self.frames),
                           ("inner_decomposition", self.inner),
                           ("precision_terms", self.precision)):
            if not rows:
                continue
            fields = list(dict.fromkeys(k for r in rows for k in r))
            p = os.path.join(out_dir, f"{name}.csv")
            with open(p, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields, restval="")
                w.writeheader()
                w.writerows(rows)
            paths[name] = p
        return paths


# ─────────────────────── critical-frame detection (Sec. 1) ─────────────────
def find_critical_frames(frames, key="mse_full", jump=1.5):
    """Locate where the trajectory stops being well behaved.

    Rules, stated so the answer is checkable rather than eyeballed:

      stage_start      the first frame of each stage (the U^(1) transition)
      first_jump       first frame whose `key` is >= `jump` x the previous frame
      max_jump         frame with the largest ratio to the previous frame
      onset_sustained  earliest frame from which `key` never decreases again
                       and grows by at least 2x by the end -- i.e. the point
                       after which the run only gets worse
    """
    v = [r[key] for r in frames]
    n = len(v)
    out = []

    seen = set()
    for r in frames:
        if r["stage"] not in seen:
            seen.add(r["stage"])
            out.append(dict(kind="stage_start", stage=r["stage"],
                            frame=r["frame"], tau=r["tau"], value=r[key],
                            note=f"first frame of stage {r['stage']}"))

    ratios = [(v[i] / v[i - 1] if v[i - 1] > 0 else float("nan"))
              for i in range(1, n)]
    first = next((i + 1 for i, x in enumerate(ratios) if x >= jump), None)
    if first is not None:
        out.append(dict(kind="first_jump", stage=frames[first]["stage"],
                        frame=frames[first]["frame"], tau=frames[first]["tau"],
                        value=v[first],
                        note=f"first frame with {key} ratio >= {jump} "
                             f"(ratio {ratios[first - 1]:.2f})"))
    if ratios:
        mi = int(np.nanargmax(ratios)) + 1
        out.append(dict(kind="max_jump", stage=frames[mi]["stage"],
                        frame=frames[mi]["frame"], tau=frames[mi]["tau"],
                        value=v[mi],
                        note=f"largest single-frame ratio {ratios[mi - 1]:.2f}"))

    onset = None
    for i in range(n - 1):
        if all(v[j + 1] >= v[j] for j in range(i, n - 1)) and v[-1] >= 2 * v[i]:
            onset = i
            break
    if onset is not None:
        out.append(dict(kind="onset_sustained", stage=frames[onset]["stage"],
                        frame=frames[onset]["frame"], tau=frames[onset]["tau"],
                        value=v[onset],
                        note=f"{key} never decreases after this frame and grows "
                             f"{v[-1] / max(v[onset], 1e-12):.1f}x by the end"))
    return out


def montage_frames(frames, critical, base=(0, 10, 15, 20, 22, 24, 26, 28),
                   pad=3):
    """The frames the user asked for, plus +-pad around every detected turning
    point, plus stage-3 representatives and the final frame."""
    ids = set(base)
    last = frames[-1]["frame"]
    ids.add(last)
    for c in critical:
        for d in range(-pad, pad + 1):
            ids.add(c["frame"] + d)
    for r in frames:
        if r["stage"] == 3 and r["step"] in (0, 3, 6, 9):
            ids.add(r["frame"])
    return sorted(f for f in ids if 0 <= f <= last)


# ───────────────────────────────── plots ──────────────────────────────────
def _stage_lines(ax, frames):
    seen = set()
    for r in frames:
        if r["stage"] not in seen and r["stage"] > 0:
            seen.add(r["stage"])
            ax.axvline(r["frame"] - 0.5, color="k", lw=0.8, ls="--", alpha=0.6)
            ax.text(r["frame"] - 0.4, ax.get_ylim()[1], f" stage {r['stage']}",
                    fontsize=7, va="top", color="k", alpha=0.7)


def plot_loss_vs_frame(frames, critical, path):
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    fr = [r["frame"] for r in frames]

    ax = axes[0]
    ax.semilogy(fr, [r["mse_full"] for r in frames], "-o", ms=3,
                color="tab:purple", label=r"MSE$(U x_1,\ x_1^{GT})$ full-res")
    if "hole_full" in frames[0]:
        ax.semilogy(fr, [r["hole_full"] for r in frames], "-s", ms=3,
                    color="tab:red", label="hole (full-res)")
        ax.semilogy(fr, [r["obs_full"] for r in frames], "-^", ms=3,
                    color="tab:green", label="observed (full-res)")
    ax.set_ylabel("MSE"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("Algorithm 4 · box_inpainting · reconstruction vs frame "
                 "(cross-stage comparable: x1 upsampled to full resolution)")
    _stage_lines(ax, frames)
    for c in critical:
        if c["kind"] in ("first_jump", "onset_sustained"):
            ax.axvline(c["frame"], color="tab:orange", lw=1.4, alpha=0.9)

    ax = axes[1]
    ax.plot(fr, [r["psnr_full"] for r in frames], "-o", ms=3, color="tab:blue")
    ax.set_ylabel("PSNR (dB, MAX=2)"); ax.grid(alpha=0.3)
    _stage_lines(ax, frames)

    ax = axes[2]
    ax.plot(fr, [r["meas_resid"] for r in frames], "-o", ms=3,
            color="tab:orange", label=r"$\|A_kx_1-y\|/(\eta\sqrt{m})$")
    ax.axhline(1.0, color="k", ls=":", lw=1)
    ax.plot(fr, [r["x0_sq_over_n"] for r in frames], "-s", ms=3,
            color="tab:brown", label=r"$\|x_0\|^2/n$")
    ax.set_ylabel("diagnostics"); ax.set_xlabel("frame id")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    _stage_lines(ax, frames)

    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_h_sigma(precision, frames, critical, path):
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    fr = [r["frame"] for r in precision]

    ax = axes[0]
    ax.plot(fr, [r["h_range"] for r in precision], "-o", ms=3,
            label=r"$h_\tau$ on range$(G)$")
    ax.plot(fr, [r["sigma_tau"] for r in precision], "-s", ms=3,
            label=r"$\sigma_\tau$")
    ax.set_ylabel("schedule"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title(r"Schedule and Block-1 precision terms "
                 r"($C^{-1}=h_\tau^2/\sigma_\tau^2+S^{-1}$)")
    _stage_lines(ax, frames)

    ax = axes[1]
    ax.semilogy(fr, [r["h_over_sigma"] for r in precision], "-o", ms=3,
                label=r"$h_\tau/\sigma_\tau$")
    ax.semilogy(fr, [r["h2_over_sigma2"] for r in precision], "-s", ms=3,
                label=r"$h_\tau^2/\sigma_\tau^2$")
    ax.semilogy(fr, [r["inv_s2_prior"] for r in precision], "-", lw=2,
                color="k", label=r"$1/s^2$ (measured $S$)")
    ax.set_ylabel("precision"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    _stage_lines(ax, frames)

    ax = axes[2]
    if "prec_data_max" in precision[0]:
        ax.semilogy(fr, [r["prec_data_max"] for r in precision], "-o", ms=3,
                    color="tab:green", label=r"$\|A^\top A\|/\eta^2$ (data)")
    ax.semilogy(fr, [r["prec_prior_range"] for r in precision], "-s", ms=3,
                color="tab:red",
                label=r"$C^{-1}$ on range$(G)$ — the only precision in the hole")
    ax.set_ylabel("precision"); ax.set_xlabel("frame id")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    _stage_lines(ax, frames)

    for ax in axes:
        for c in critical:
            if c["kind"] in ("first_jump", "onset_sustained"):
                ax.axvline(c["frame"], color="tab:orange", lw=1.4, alpha=0.9)

    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_block_decomposition(inner, frames, critical, path):
    """Where the error appears: is x1_hat already bad, or does Block 1 spoil it?"""
    last = {}
    for r in inner:
        last[r["frame"]] = r          # the final inner iteration of each frame
    fr = sorted(last)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    ax = axes[0]
    for k, lab, c in (("mse_x1_in", r"$x_1$ into Block 1", "tab:blue"),
                      ("mse_x1_hat", r"$\hat{x}_1$ from (19)", "tab:orange"),
                      ("mse_x1_out", r"$x_1$ after Block 1", "tab:purple")):
        ax.semilogy(fr, [last[f][k] for f in fr], "-o", ms=3, color=c, label=lab)
    ax.set_ylabel("MSE vs GT at stage res"); ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("Block-by-block error decomposition (last inner iteration of "
                 "each frame). NOTE: stage resolution changes at each boundary.")
    _stage_lines(ax, frames)

    ax = axes[1]
    ax.plot(fr, [last[f]["endpoint_delta"] for f in fr], "-o", ms=3,
            color="tab:orange",
            label=r"$\mathrm{MSE}(\hat{x}_1)-\mathrm{MSE}(x_1^{in})$  (<0: (19) helps)")
    ax.plot(fr, [last[f]["block1_delta"] for f in fr], "-s", ms=3,
            color="tab:purple",
            label=r"$\mathrm{MSE}(x_1^{out})-\mathrm{MSE}(\hat{x}_1)$  (>0: Block 1 hurts)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Δ MSE"); ax.set_xlabel("frame id")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    _stage_lines(ax, frames)

    for ax in axes:
        for c in critical:
            if c["kind"] in ("first_jump", "onset_sustained"):
                ax.axvline(c["frame"], color="tab:orange", lw=1.4, alpha=0.9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_inner_feedback(inner, path, frames=None):
    """Within-frame drift: does the error grow across inner iterations?"""
    byf = {}
    for r in inner:
        byf.setdefault(r["frame"], []).append(r)
    picks = [f for f in sorted(byf) if len(byf[f]) > 1]
    if not picks:
        return
    sel = picks[:: max(1, len(picks) // 8)][:8]
    fig, ax = plt.subplots(figsize=(9, 5))
    for f in sel:
        rs = sorted(byf[f], key=lambda r: r["inner"])
        ax.semilogy([r["inner"] for r in rs], [r["mse_x1_out"] for r in rs],
                    "-o", ms=3, label=f"frame {f} (stage {rs[0]['stage']})")
    ax.set_xlabel("inner iteration $s$"); ax.set_ylabel(r"MSE$(x_1)$ vs GT")
    ax.set_title("Inner-loop feedback: error across the $S_{it}$ iterations of a frame")
    ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_montage(captured, frames, ids, gt_full, hole_full, path, device="cpu"):
    """One row per selected frame: GT | x1 into Block 1 | x1_hat | x1 after
    Block 1 | x_tau after Block 2 | error map."""
    ids = [f for f in ids if f in captured]
    if not ids:
        return
    byf = {r["frame"]: r for r in frames}
    cols = ["GT", r"$x_1$ in", r"$\hat{x}_1$ (19)", r"$x_1$ out",
            r"$x_\tau$ (Block 2)", "|err| at frame end"]
    fig, axes = plt.subplots(len(ids), len(cols),
                             figsize=(2.05 * len(cols), 2.15 * len(ids)),
                             squeeze=False)
    for ri, f in enumerate(ids):
        cap = captured[f]
        row = byf.get(f, {})
        gt_k = gt_full
        # A frame with no x1_in never ran: sigma_tau < sigma_min skipped its
        # whole block and the state was simply carried. Render the carried x1
        # instead of four blank panels, and say so.
        skipped = "x1_in" not in cap
        panels = [gt_k[0].cpu() if gt_k.dim() == 4 else gt_k,
                  cap.get("x1_in"), cap.get("x1_hat"),
                  cap.get("x1_out", cap.get("x1_end") if skipped else None),
                  cap.get("x_tau"), None]
        for ci in range(len(cols)):
            ax = axes[ri][ci]
            ax.set_xticks([]); ax.set_yticks([])
            if ci == 5:
                x1o = cap.get("x1_end", cap.get("x1_out"))
                if x1o is not None:
                    up = upsample_to(x1o.unsqueeze(0), gt_full.shape[-2:])
                    err = (up - gt_full.cpu()).abs().mean(1)[0].numpy()
                    ax.imshow(err, cmap="magma", vmin=0, vmax=1.0)
                if ri == 0:
                    ax.set_title(cols[ci], fontsize=9)
                continue
            t = panels[ci]
            if t is not None:
                ax.imshow(to_img(t.to(device)))
            elif skipped and ci in (1, 2, 4):
                ax.text(0.5, 0.5, "skipped\n" + r"$\sigma_\tau<\sigma_{min}$",
                        ha="center", va="center", fontsize=8, color="0.45",
                        transform=ax.transAxes)
            if ri == 0:
                ax.set_title(cols[ci], fontsize=9)
        axes[ri][0].set_ylabel(
            f"f{f}  st{row.get('stage','?')}{' (skip)' if skipped else ''}\n"
            fr"$\tau$={row.get('tau', float('nan')):.2f}"
            f"\nmse={row.get('mse_full', float('nan')):.3f}",
            fontsize=7, rotation=0, ha="right", va="center", labelpad=34)
    fig.suptitle("Algorithm 4 · box_inpainting · trajectory montage — columns 1-4 "
                 "are the FIRST inner iteration; the error map is the frame end",
                 fontsize=11)
    fig.tight_layout(rect=[0.02, 0, 1, 0.97])
    fig.savefig(path, dpi=140); plt.close(fig)


def plot_s_prior_vs_diversity(summary, path):
    """Reconstruction against spread, per S_prior arm.

    Read the right-hand panel with the range check beside it: a spread that is
    several times the GT std, on samples that leave [-1, 1], is divergence
    magnitude and not posterior width (draft Sec. 8.2 / 8.6).
    """
    lam = [r["lambda_eq"] for r in summary]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].semilogx(lam, [r["hole_mse_mean"] for r in summary], "-o",
                     color="tab:red", label="hole MSE")
    axes[0].set_xlabel(r"$\lambda_{eq}=1/s^2$"); axes[0].set_ylabel("hole MSE")
    axes[0].grid(alpha=0.3); axes[0].legend(fontsize=8)
    axes[0].set_title("reconstruction improves as precision grows")
    axes[1].semilogx(lam, [r["hole_spread"] for r in summary], "-o",
                     color="tab:blue", label="hole spread (std across seeds)")
    axes[1].semilogx(lam, [r["obs_spread"] for r in summary], "-s",
                     color="tab:green", label="observed spread")
    axes[1].axhline(0.478, color="k", ls=":", lw=1)
    axes[1].text(lam[0], 0.478, " GT hole std", fontsize=7, va="bottom")
    axes[1].set_xlabel(r"$\lambda_{eq}=1/s^2$")
    axes[1].set_ylabel("per-pixel std across seeds")
    axes[1].grid(alpha=0.3); axes[1].legend(fontsize=8)
    axes[1].set_title("...but is that posterior width, or divergence?")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_contraction_map(rows, path):
    """The one-step map of the inner loop, measured rather than inferred.

    x-axis: hole MSE planted before the pass. y-axis: hole MSE after one
    clean-endpoint -> Block 1 -> Block 2 pass. The diagonal is the fixed-point
    line: a frame whose curve sits below it contracts, above it diverges, and
    where the curve crosses the diagonal from below the crossing is a stable
    fixed point -- the value the inner loop settles at.
    """
    byf, byr = {}, {}
    for r in rows:
        # The replay arm is the accumulated error the sampler itself produced,
        # injected at this frame. It is drawn as a separate marker and never as
        # part of the white-corruption curve: the point of the probe is that the
        # two behave differently at the same magnitude.
        (byr if r.get("kind") == "replay" else byf).setdefault(
            r["frame"], []).append(r)
    frames = sorted(byf)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))

    ax = axes[0]
    lo = min(r["d_in"] for r in rows); hi = max(max(r["d_in"] for r in rows),
                                                max(r["d_out"] for r in rows))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1,
            label=r"fixed-point line $\delta'=\delta$")
    cmap = plt.get_cmap("viridis")
    for i, f in enumerate(frames):
        rs = sorted(byf[f], key=lambda r: r["d_in"])
        xs, ys = {}, {}
        for r in rs:
            xs.setdefault(r["delta_planted"], []).append(r["d_in"])
            ys.setdefault(r["delta_planted"], []).append(r["d_out"])
        dl = sorted(xs)
        ax.plot([np.mean(xs[d]) for d in dl], [np.mean(ys[d]) for d in dl],
                "-o", ms=3, color=cmap(i / max(len(frames) - 1, 1)),
                label=f"frame {f} (st {rs[0]['stage']}, "
                      fr"$\sigma_\tau$={rs[0]['sigma_tau']:.3f})")
    for i, f in enumerate(sorted(byr)):
        r = byr[f][0]
        col = cmap(frames.index(f) / max(len(frames) - 1, 1)) if f in frames else "k"
        ax.plot([r["d_in"]], [r["d_out"]], "*", ms=14, color=col,
                markeredgecolor="k", markeredgewidth=0.6,
                label="replay: error the sampler produced" if i == 0 else None)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"hole MSE planted, $\delta$")
    ax.set_ylabel(r"hole MSE after one pass, $\delta'$")
    ax.set_title("One-step map (lines = white corruption, stars = replay)")
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    ax = axes[1]
    for i, f in enumerate(frames):
        rs = sorted(byf[f], key=lambda r: r["d_in"])
        by_d = {}
        for r in rs:
            by_d.setdefault(r["delta_planted"], []).append(r)
        dl = sorted(by_d)
        ax.plot([np.mean([r["d_in"] for r in by_d[d]]) for d in dl],
                [np.mean([r["d_hat"] / max(r["d_in"], 1e-12) for r in by_d[d]])
                 for d in dl],
                "-o", ms=3, color=cmap(i / max(len(frames) - 1, 1)),
                label=f"frame {f}")
    for f in sorted(byr):
        r = byr[f][0]
        col = cmap(frames.index(f) / max(len(frames) - 1, 1)) if f in frames else "k"
        ax.plot([r["d_in"]], [r["d_hat"] / max(r["d_in"], 1e-12)], "*", ms=14,
                color=col, markeredgecolor="k", markeredgewidth=0.6)
    ax.axhline(1.0, color="k", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel(r"hole MSE planted, $\delta$")
    ax.set_ylabel(r"contraction of (19): MSE$(\hat{x}_1)$ / $\delta$")
    ax.set_title("Does (19) pull it back?  stars = replay")
    ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_final_output(captured, frames, gt_full, hole_full, path, device="cpu"):
    """The returned sample against the denoised endpoint at the last frame that
    actually ran. The draw carries the injected posterior variance 1/C^-1 on
    purpose -- a grainy hole is a SAMPLE, not a failed reconstruction -- and
    this figure is what separates the two."""
    ran = [f for f in sorted(captured) if "x1_hat_last" in captured[f]]
    if not ran:
        return
    f = ran[-1]
    cap = captured[f]
    draw = cap.get("x1_end", cap.get("x1_out"))
    hat = cap["x1_hat_last"]
    gt = gt_full[0].cpu() if gt_full.dim() == 4 else gt_full.cpu()
    def hole_mse(t):
        up = upsample_to(t.unsqueeze(0), gt_full.shape[-2:])
        return mse_masked(up, gt_full.cpu(), hole_full.cpu())
    fig, axes = plt.subplots(1, 5, figsize=(5 * 2.4, 3.0))
    items = [(gt, "GT"),
             (draw, f"returned draw\nhole={hole_mse(draw):.3f}"),
             (hat, f"$\\hat{{x}}_1$ (last iter)\nhole={hole_mse(hat):.3f}"),
             (None, "|err| draw"), (None, "|err| $\\hat{{x}}_1$")]
    for ci, (t, lab) in enumerate(items):
        ax = axes[ci]; ax.set_xticks([]); ax.set_yticks([])
        if ci < 3:
            ax.imshow(to_img(t.to(device)))
        else:
            src = draw if ci == 3 else hat
            up = upsample_to(src.unsqueeze(0), gt_full.shape[-2:])
            ax.imshow((up - gt_full.cpu()).abs().mean(1)[0].numpy(),
                      cmap="magma", vmin=0, vmax=1.0)
        ax.set_title(lab, fontsize=8)
    fig.suptitle(f"final executed frame f{f}: the sample vs the denoised "
                 "endpoint it was drawn around", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, dpi=150); plt.close(fig)
