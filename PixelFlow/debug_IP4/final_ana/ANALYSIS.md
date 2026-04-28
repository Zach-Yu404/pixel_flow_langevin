# Final Analysis — 5 candidate configs vs IP3 `ms_posterior_sampling_article_version_final` baseline

5 candidates selected from user's set of "relatively good" results:

| ID | Name | Source |
|----|------|--------|
| **A** | `baseline_IP3final`      | `ms_posterior_sampling_article_version_final.json` defaults (no IP4 extras) |
| **B** | `G_hx0.1_heps0.0001`     | R6 config #5 — heps=1e-4 + lam_prox=50 + termrep |
| **C** | `W1_heps_0.001_F1`       | R2 W1 paired with termrep (from N2_visual set) |
| **D** | `WINNER3`                | W1 + X4 + F1  (global heps=1e-3 + h_x stg3=0.7 + termrep) |
| **E** | `N2_plus_W1_hx0.5`       | N2 mask-aware-eps + heps=1e-3 + h_x stg3=0.5 + termrep |

Each run evaluated on **box** AND **random** inpainting with the same GT (2 samples).
Per-config `PSNR_all`, `PSNR_obs`, `PSNR_unobs` use the corrected `[0,1]`-convention
formula (see `W1_DIAGNOSIS.md §2` for the bug write-up). `HF` is the FFT
high-frequency energy on the unobserved region (noted unreliable — see §5).

Reconstructions: `box/*.png`, `random/*.png`.
Side-by-side grids: `box/comparison.png`, `random/comparison.png`.
Exact configs: `configs/*.json`.

---

## §1 — Differences vs baseline (knob-by-knob)

All candidates keep IP3's `warm_restart=true`, `g_bypass_stage3=true`,
`num_langevin=10`, `lambda_reg=50`, `rho_s=rho_e=1.0`, `cg_tol=1e-5`,
`cg_max_iter=50`. Only the **differences** are listed.

| Knob | A (baseline) | B | C | D (WINNER3) | E |
|------|--------------|---|---|-------------|---|
| `h_x` stage-3       | 0.1            | 0.1         | 0.1         | **0.7**     | **0.5**         |
| `h_epsilon`         | 0.01           | **0.0001**  | **0.001**   | **0.001**   | **0.001**       |
| `lambda_prox`       | 0.0            | **50.0**    | 0.0         | 0.0         | 0.0             |
| `terminal_replace`  | 0.0            | **1.0**     | **1.0**     | **1.0**     | **1.0**         |
| `mask_aware_eps`    | false          | false       | false       | false       | **true**        |

Interpretation:
- **B/C/D/E** all pull on the *eps-entropy* lever (`h_epsilon` ≤ 0.001) — the
  single lever that was shown in rounds 1-6 to actually create semantic
  content inside a removed region.
- **B** alone also restores the paper's proximal term on `x1` (`lambda_prox=50`).
- **D** adds the `h_x` stage-3 boost on top of C — meant to sharpen the
  generated texture.
- **E** replaces the `h_x` boost with N2 (mask-aware eps: in unobserved region,
  drop the Tweedie pull so eps stays Gaussian).
- **A** is the untouched baseline — no termrep, no eps-tempering, no paper prox.

---

## §2 — Metrics (corrected PSNR, [0,1] convention)

### BOX inpainting — GT HF = 0.675

| Config | PSNR_all | PSNR_obs | PSNR_unobs | HF | \|ΔHF\| | res |
|--------|----------|----------|------------|-----|---------|-----|
| A baseline           | **19.90** | 47.40 | **15.96** | 0.620 | 0.055 | 46   |
| B h_eps=1e-4+prox50  | 16.17     | ∞     | 12.22     | 0.674 | **0.001** | 0 |
| C h_eps=1e-3         | 17.13     | ∞     | 13.18     | 0.677 | 0.002 | 0 |
| D WINNER3            | 16.00     | ∞     | 12.05     | 0.750 | 0.075 (over) | 0 |
| E N2+W1+hx0.5        | 15.76     | ∞     | 11.81     | 0.725 | 0.050 (over) | 0 |

### RANDOM inpainting (p=0.8 retained) — GT HF = 0.577

| Config | PSNR_all | PSNR_obs | PSNR_unobs | HF | \|ΔHF\| | res |
|--------|----------|----------|------------|-----|---------|-----|
| A baseline           | **23.08** | 40.37 | **25.15** | 0.600 | 0.023 | 58 |
| B h_eps=1e-4+prox50  | 20.98     | ∞     | 23.02     | 0.609 | 0.031 | 0 |
| C h_eps=1e-3         | 20.98     | ∞     | 23.02     | 0.609 | 0.032 | 0 |
| D WINNER3            | 16.34     | ∞     | 18.38     | 0.682 | 0.105 (over) | 0 |
| E N2+W1+hx0.5        | 17.72     | ∞     | 19.76     | 0.655 | 0.077 (over) | 0 |

Bold = best in that column. `PSNR_obs=∞` when `terminal_replace_weight=1.0` zeroes residual.

---

## §3 — Visual summary (from comparison PNGs)

### BOX (see `box/comparison.png`)

- **A baseline**: **Smooth brown box, no bird body**. Classic failure mode.
  Looks cleaner than the others at the PSNR level because smooth fill is close
  to the GT low-frequency mean, but perceptually empty.
- **B (h_eps=1e-4)**: **Reconstructed bird** — visible head, body, feather pattern.
  Strong texture; this is the most detailed output. Different species/pose from GT
  (information in the box is gone so exact bird is unrecoverable).
- **C (h_eps=1e-3)**: **Reconstructed bird, slightly softer** than B. Less grainy,
  more contiguous body outline. Arguably the most "photo-like" of the four
  texture-generating candidates.
- **D (WINNER3)**: **Bird with sharper/noisier contours**. h_x stage-3 boost
  amplifies the prior score but also injects HF overshoot — HF=0.750 vs GT 0.675.
  Visually "busy".
- **E (N2+W1)**: Similar to D. The mask-aware eps term compounds with h_x=0.5 to
  produce comparable overshoot.

Verdict on BOX: **C is the best overall for box** (balanced PSNR + meaningful
content), with B a close second if more texture is desired. D and E oversharp.
A is only competitive on PSNR, not on perceptual semantics.

### RANDOM (see `random/comparison.png`)

- **A baseline**: **Clean, faithful reconstruction**. Bird looks natural, branch
  intact, no visible noise. PSNR_unobs = **25.15** is genuinely good.
- **B, C**: Nearly identical to each other. **Slight graininess** in the
  reconstructed regions (observed-sparse regions show texture instead of smooth
  fill). PSNR_unobs = 23.02 — about 2 dB lower than A.
- **D (WINNER3)**: **Significant degradation over the whole image**. Noisy
  texture everywhere, especially on the bird body and branch. HF overshoot of
  0.105 vs GT. PSNR_unobs drops to 18.38.
- **E (N2+W1)**: Also degraded but milder than D.

Verdict on RANDOM: **A baseline is the clear winner**. The eps-entropy recipes
designed for box inpainting actively *hurt* random inpainting because random
masks already provide dense per-pixel anchoring, and the preserved entropy
becomes visible noise rather than plausible texture.

---

## §4 — The key finding: best config depends on the mask type

| Mask type | Best config | Why |
|-----------|-------------|-----|
| **Box (large contiguous hole)** | **C** (`h_eps=0.001 + terminal_replace=1.0`) | Box removes the information needed for pixel-aligned reconstruction. Baseline fills with smooth mean. Low h_eps preserves eps-entropy → model generates plausible texture. Termrep snaps observed to y for free. |
| **Random (p=0.8 observations)** | **A** (IP3 defaults, no IP4 changes) | Random observations provide per-pixel guidance. Entropy preservation turns into noise. Baseline's strong damping is *correct* here. |

**Don't blindly port the box winner to random — the mask topology dictates the
optimal entropy-damping schedule.**

A single config that handles both would need `h_epsilon` adaptive to something
like the per-pixel observed-pixel density in a neighbourhood. Not tested in
Rounds 1-6.

---

## §5 — Metric reliability caveats

1. **HF energy metric is unreliable as a semantic-content indicator**. As shown
   in Rounds 5-6, configs like M3 (lambda reduction) and N2 (mask-aware) hit HF
   bullseye while producing **visually smooth-filled boxes identical to baseline**.
   HF is dominated by mask-boundary discontinuities, not box-interior content.
2. **PSNR_unobs biases toward smooth fill** because MSE penalises misaligned
   bird texture more than a content-free low-variance fill. On box, baseline's
   PSNR_unobs 15.96 "beats" C's 13.18, despite C actually generating a bird.
3. **Only visual inspection is trustworthy** for ill-posed inpainting.

---

## §6 — File manifest under `final_ana/`

```
final_ana/
├── ANALYSIS.md            (this file)
├── metrics.json           (machine-readable: all PSNR/HF/res per config/task)
├── configs/
│   ├── A_baseline_IP3final.json
│   ├── B_G_hx0.1_heps0.0001.json
│   ├── C_W1_heps_0.001_F1.json
│   ├── D_WINNER3.json
│   └── E_N2_plus_W1_hx0.5.json
├── box/
│   ├── GT.png
│   ├── A_baseline_IP3final.png
│   ├── B_G_hx0.1_heps0.0001.png
│   ├── C_W1_heps_0.001_F1.png
│   ├── D_WINNER3.png
│   ├── E_N2_plus_W1_hx0.5.png
│   └── comparison.png
└── random/
    ├── GT.png
    ├── A_baseline_IP3final.png
    ├── B_G_hx0.1_heps0.0001.png
    ├── C_W1_heps_0.001_F1.png
    ├── D_WINNER3.png
    ├── E_N2_plus_W1_hx0.5.png
    └── comparison.png
```

## §7 — One-line recommendation per mask type

- **Box**: `h_x=0.1`, `h_epsilon=0.001`, `lambda_prox=0`, `terminal_replace=1.0` (config **C**).
- **Random**: `h_x=0.1`, `h_epsilon=0.01`, `lambda_prox=0`, `terminal_replace=1.0` (config **A + termrep**) — or just IP3 baseline if termrep is not available.
- **Do NOT ship a single universal config** — auto-detect mask sparsity and
  switch between the two profiles.
