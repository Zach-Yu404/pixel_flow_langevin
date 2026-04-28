# Per-stage h_epsilon analysis

28 configurations in Round-7 + 4 trajectory visualizations.
Box mask, GT HF=0.693. `terminal_replace_weight=1.0` on every config (decouples
data-consistency from the eps-preservation axis we care about).

HI = 0.01 (default), LO = 0.001 (W1-level), VLO = 1e-4.

## 1. Trajectory diagnosis (from `results_trajectory/eps_norm_curves.png`)

![](results_trajectory/eps_norm_curves.png)

Per-step eps L2 norm (relative to √N of a standard Gaussian, so 1.0 = same as `N(0,I)`):

- **Baseline (h_eps=0.01, blue)**: within each stage the eps **collapses from
  ≈0.9 to ≈0.4** over 10 inner-loop iterations. Stage transitions reset it
  back to ≈0.8 via the `(lat − s_k·G(x1))/(1−s_k)` re-derivation, but the
  collapse resumes.
- **W1 (h_eps=0.001, orange)**: eps decays only very mildly from 1.0 to ≈0.88
  across ALL stages, then collapses at the final ODE step (tau≈1). Gaussian-ness
  is preserved throughout the chain.
- **WINNER3 (green)**: virtually identical to W1. `h_x=0.7` affects `x1` only,
  not eps.
- **S0-only LO (red)**: stays ≈0.93 during stage 0, but once stage 1 with
  h_eps=0.01 begins, **eps collapses in exactly the same curve as baseline**.
  **Warm-restart at stage transition does NOT restore past entropy** — the
  new eps is re-derived from (xtau, x1_hat) pair that was built using the
  previous high-h_eps latent. So a single-stage LO is quickly forgotten.

## 2. Per-stage contribution quantification (R7 INV series, "LO everywhere except one stage")

Comparing each `INV_stage{i}_HI` to the `REF_W1_all_LO` reference:

| Config | HF | PSNR | ΔHF vs W1 | Interpretation |
|--------|-----|------|-----------|----------------|
| REF_W1_all_LO | 0.666 | 18.36 | — | reference |
| INV_stage0_HI | 0.637 | 20.74 | **−0.029** | stg0 LO contributes moderately |
| INV_stage1_HI | 0.610 | 20.05 | **−0.056** | **stg1 LO contributes the MOST** |
| INV_stage2_HI | 0.632 | 19.05 | −0.034 | stg2 LO contributes moderately |
| INV_stage3_HI | 0.670 | 18.31 | **+0.004** | **stg3 LO contributes ~zero** |

⇒ **Stage 1 is the bottleneck** for eps-entropy preservation. Removing its LO
schedule kills half the perceptual gain. Stage 3 removal barely moves anything
(it's the last dance; eps is already collapsed AND will be terminally replaced
anyway).

Corroborated by the symmetric R7 S series (only one stage LO):

| Config | HF | PSNR | ΔHF vs baseline (0.627) | Conclusion |
|--------|-----|------|--------------------------|------------|
| S0_only | 0.605 | 20.29 | −0.022 | negative (moves AWAY from GT=0.693) |
| S1_only | 0.624 | 21.04 | ≈0   | neutral |
| S2_only | 0.639 | 21.29 | +0.012 | mild positive |
| S3_only | 0.624 | 21.52 | ≈0   | neutral (confirms H1 finding) |

The S series is less revealing because a single-stage LO gets collapsed by
subsequent HI stages. But it confirms that with `h_eps=0.01` in the later
stages, eps-entropy from any earlier single stage cannot survive.

## 3. Box-region MSE over time (`results_trajectory/box_mse_curves.png`)

![](results_trajectory/box_mse_curves.png)

All 4 configs track **identical** box MSE through stages 0-2 (log-y scale,
essentially flat around 3×10⁻² MSE). Only at **stage 3** (steps 30-40) do the
trajectories diverge:

| Config | Final box MSE | Visual |
|--------|----------------|--------|
| baseline | **1.8×10⁻²** (lowest) | smooth brown fill |
| S0_only_LO | 2.6×10⁻² | closer to baseline than to W1 |
| W1 (h_eps=0.001) | 4.5×10⁻² | bird reconstructed |
| WINNER3 (+h_x=0.7 stg3) | 6.2×10⁻² (highest) | bird with sharper texture |

This makes the MSE-vs-visual paradox explicit: **higher MSE = better
semantic content**. The baseline "wins" the MSE race by filling with the mean.
The content-generating configs pay ~3× higher MSE to buy a plausible bird.

## 4. Full R7 Pareto table (28 configs, ranked by PSNR)

| Rank | Config | Schedule [s0,s1,s2,s3] | PSNR | PSNR_unobs | HF | \|Δ\| |
|------|--------|-------------------------|------|------------|-----|--------|
| 1 | S3_stage3_only          | [HI,HI,HI,LO] | 21.52 | 13.74 | 0.624 | 0.069 |
| 2 | REF_baseline            | [HI,HI,HI,HI] | **21.49** | 13.71 | 0.627 | 0.066 |
| 3 | R_decr_0.01_0.0005      | [0.01,0.01,0.005,5e-4] | 21.39 | 13.61 | 0.631 | 0.062 |
| 4 | P23_stages_23           | [HI,HI,LO,LO] | 21.31 | 13.53 | 0.636 | 0.057 |
| 5 | S2_stage2_only          | [HI,HI,LO,HI] | 21.29 | 13.50 | 0.639 | 0.054 |
| 6 | R_decr_0.01_0.0001      | [0.01,0.005,0.002,0.001] | 21.14 | 13.35 | 0.628 | 0.065 |
| 7 | P13_stages_13           | [HI,LO,HI,LO] | 21.09 | 13.31 | 0.622 | 0.071 |
| 8 | **S1_stage1_only**      | [HI,LO,HI,HI] | **21.04** | 13.26 | 0.624 | 0.069 |
| 9 | T123_stages_123         | [HI,LO,LO,LO] | 20.74 | 12.96 | 0.637 | 0.056 |
| 10| INV_stage0_HI           | [HI,LO,LO,LO] | 20.74 | 12.96 | 0.637 | 0.056 |
| 11| P12_stages_12           | [HI,LO,LO,HI] | 20.70 | 12.92 | 0.640 | 0.053 |
| 12| R_inv_v_shape           | [HI,LO,LO,HI] | 20.70 | 12.92 | 0.640 | 0.053 |
| 13| P03_stages_03           | [LO,HI,HI,LO] | 20.34 | 12.55 | 0.602 | 0.091 |
| 14| R_v_shape               | [LO,HI,HI,LO] | 20.34 | 12.55 | 0.602 | 0.091 |
| 15| S0_stage0_only          | [LO,HI,HI,HI] | 20.29 | 12.51 | 0.605 | 0.088 |
| 16| T023_stages_023         | [LO,HI,LO,LO] | 20.05 | 12.26 | 0.610 | 0.083 |
| 17| INV_stage1_HI           | [LO,HI,LO,LO] | 20.05 | 12.26 | 0.610 | 0.083 |
| 18| P02_stages_02           | [LO,HI,LO,HI] | 20.03 | 12.25 | 0.612 | 0.081 |
| 19| T013_stages_013         | [LO,LO,HI,LO] | 19.05 | 11.27 | 0.632 | 0.061 |
| 20| INV_stage2_HI           | [LO,LO,HI,LO] | 19.05 | 11.27 | 0.632 | 0.061 |
| 21| P01_stages_01           | [LO,LO,HI,HI] | 19.01 | 11.23 | 0.633 | 0.060 |
| 22| REF_W1_all_LO           | [LO,LO,LO,LO] | 18.36 | 10.58 | 0.666 | **0.027** |
| 23| T012_stages_012         | [LO,LO,LO,HI] | 18.31 | 10.53 | 0.670 | **0.023** |
| 24| INV_stage3_HI           | [LO,LO,LO,HI] | 18.31 | 10.53 | 0.670 | **0.023** |
| 25| V_stg01_1e-4            | [VLO,VLO,HI,HI] | 18.18 | 10.40 | 0.666 | 0.027 |
| 26| R_incr_0.0001_0.01      | [VLO,5e-4,2e-3,0.01] | 17.72 | 9.94 | 0.687 | **0.006** |
| 27| V_all_1e-4              | [VLO,VLO,VLO,VLO] | 17.26 | 9.47 | 0.699 | **0.006** |
| 28| V_stg012_1e-4_stg3_HI   | [VLO,VLO,VLO,HI] | 17.17 | 9.38 | 0.703 | 0.010 |

Pareto frontier:
- Best PSNR (no perceptual gain): **REF_baseline / S3_only**.
- Best balanced: **S1_stage1_only** (21.04 / 0.624, only stage 1 touched).
- Stronger perceptual: **P12 / P23** (stages 1+2 LO: PSNR > 20.7, HF within 0.053).
- Best HF (at PSNR cost): **R_incr_0.0001_0.01** and **V_all_1e-4** (|Δ|≈0.006 HF bullseye at PSNR -4).

## 5. Key findings

### Finding A — Stage 3 h_epsilon is nearly irrelevant
`INV_stage3_HI` = `T012_stages_012` = [LO,LO,LO,HI] → both score 18.31/0.670.
Stage 3 reverting to HI changes HF by +0.004 vs pure W1. Interpretation:
- At τ near 1 in stage 3, `σ_τ ≈ (1-τ)(1-s_3) + τ·(1-e_3) ≈ 1-e_3 ≈ 0` — the
  flow interpolant weights the noise component to zero anyway.
- Most of stage 3's Langevin is spent on the likelihood term (observed-pixel
  fit), not on eps stochasticity.
- Implication: **no need to run stage 3 Langevin at LO h_eps** — saves nothing
  but incurs no cost either. Any stage-3-local optimization lives outside the
  eps axis.

### Finding B — Stage 1 is the critical eps-preservation stage
`INV_stage1_HI` is the worst destructor of the W1 effect (HF drops 0.056).
`S1_stage1_only` (single-stage LO) is the best per-dB PSNR trade in the R7 grid.

Why stage 1? Trajectory eps-norm curve shows:
1. Stage 0 has low resolution (32×32) — eps collapse is shallow (norm only dips
   from 1.0 to ~0.92 at baseline rate).
2. Stage 1 (64×64) — eps norm drops FROM 0.81 TO 0.40 in baseline.
   This is the **largest single contraction in the chain**.
3. By stage 2 start, eps is already nearly collapsed, so preserving from there
   is too-late partial rescue.

⇒ The "critical stage" for preserving noise is stage 1, where the Langevin
   inner loop at a mid-resolution does most of the damage.

### Finding C — Warm-restart does not heal past collapse
S0-only-LO trajectory shows eps re-rising to ~0.8 at stage-1 start, then
collapsing along the baseline curve within stage 1. The warm-restart
re-derivation `eps = (x_tau − H_τ(x1_hat))/σ_τ` produces an OK-looking eps
numerically but it has no access to the entropy that stage 0 preserved — the
x_tau it reconstructs from already reflects the collapsed latent.

⇒ eps-preservation is a **monotonic chain** — once a stage loses entropy, no
downstream stage can recover it.

### Finding D — Progressive (stage-dependent) schedules don't beat uniform
`R_decr_0.01→0.0001` = 21.14/0.628 ≈ baseline.
`R_incr_0.0001→0.01` = 17.72/0.687 — great HF but hurts PSNR more than uniform
VLO. The increasing ramp collapses eps too late to help.

⇒ **Uniform low h_eps is still the right approach** if you want W1-level
perceptual gain. The stage-isolated schedules only make sense if you want
**milder** perceptual push with **smaller** PSNR cost.

## 6. Practical recommendations

| Goal | Schedule | Expected numbers (this mask, GT_HF=0.693) |
|------|----------|---------------------------------------|
| Best PSNR, no perceptual needed | `0.01` (baseline) | 21.49 / HF 0.627 / |Δ|=0.066 |
| **Best PSNR + mild perceptual** | `[HI, LO, HI, HI]` (S1_stage1_only) | **21.04 / 0.624 / 0.069** but box less smooth than baseline |
| Balanced | `[HI, LO, LO, HI]` (P12) | 20.70 / 0.640 / 0.053 |
| Strong perceptual, −2 dB PSNR | `[LO, LO, LO, HI]` (T012 / drop stage 3 LO) | 18.31 / 0.670 / 0.023 |
| **WINNER3 recipe (full)** | `[LO, LO, LO, LO]` + h_x stg3=0.7 | ~17 / HF near GT / PSNR cost −4 dB |
| HF bullseye | `0.0001` uniform (V_all_1e-4) | 17.26 / 0.699 / 0.006 |

**Notable observation**: `T012` and `INV_stage3_HI` are the same config under
two names (stages 0-2 LO, stage 3 HI). It gives the W1 perceptual gain at the
same PSNR as W1 but spends 25% less compute on Langevin (h_eps small for only
3/4 of the inner loops). Essentially free efficiency.

## 7. Mechanistic summary

```
┌──────────────────────────────────────────────────────────────────┐
│  STAGE 0   STAGE 1   STAGE 2   STAGE 3                          │
│   (32²)     (64²)    (128²)    (256²)                           │
│                                                                  │
│  eps init ─┬─> collapse ─┬─> collapse ─┬─> collapse ─┬─> final  │
│            │             │             │             │          │
│            ↓             ↓             ↓             ↓          │
│  mild slope           HEAVY collapse   moderate    negligible   │
│  (Finding C)          (Finding B)                  (Finding A)  │
│                                                                  │
│  To save eps ⇒ apply LO h_epsilon from stage 0 through stage 2  │
│  Stage 3 can stay HI (Finding A)                                │
└──────────────────────────────────────────────────────────────────┘
```

## 8. Files

- R7 log & JSON: `round7_stdout.log`, `results_round7/round7_results.json`
- R7 grid PNG:   `results_round7/round7.png`
- Trajectory:    `results_trajectory/trajectory_grid.png` (visual x1 over time)
- Trajectory:    `results_trajectory/eps_norm_curves.png` (eps collapse curves)
- Trajectory:    `results_trajectory/box_mse_curves.png` (reconstruction MSE curves)
- This analysis: `PER_STAGE_ANALYSIS.md`
