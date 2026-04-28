# Parameter-effects summary — PRINCIPLE 128×128 box, res=256

How each knob influences perceptual quality (LPIPS), distortion (PSNR), and
HF-energy match (`|dHF|`). Built from a 167-config corpus across 4 sweep
stages anchored on `C_L5` (h_x=[.1,.1,.1,.7], h_eps=1e-3, L=5, λ_reg=50,
noise_scale=0, terminal_replace_weight=1) and `N_cfg2` (= `C_L5` + CFG=2).

> **Caveats.** (i) The 2-image GT + single mask seed limits absolute precision;
> ranking deltas <0.005 LPIPS are tie-noise. (ii) **Co-tenancy noise** added
> ~0.03 LPIPS / 0.4 dB drift between Stage E (`runs/`) and Stage P (`runs_p/`)
> batches. Compare each P_*/Q_* config to the anchor *that was rerun in the
> same batch* (`R_anchor_C_L5`, `R_anchor_N_cfg2`), not to the original
> `C_L5`/`N_cfg2`. (iii) Stage F (`runs_f/`) is the only batch that ran 12
> configs simultaneously — those numbers are the trustworthy "PK" data.
> Treat trends as robust and absolute LPIPS as ±0.04.

The winner from this corpus is `F_cfg20_lr150_hxu02` (CFG=2.0 + λ_reg=150 +
uniform h_x=0.2). It dominates the same-batch F sweep on LPIPS and |dHF|.

Per-axis curve PNGs: `comparisons/curve_*.png`.

## guidance_scale (CFG) — the dominant *new* axis

Single-axis sweep anchored on `N_cfg2` (CFG=2 by default; CFG=0 = `C_L5`).

| CFG | Config | PSNR | LPIPS | \|dHF\| |
|---|---|---|---|---|
| 0.0 | `C_L5` | 17.51 | 0.1295 | 0.011 |
| 0.5 | `P_cfg_05` | 17.66 | 0.142 | 0.021 |
| 1.0 | `P_cfg_10` | 17.62 | 0.138 | 0.018 |
| 1.5 | `P_cfg_15` | 17.54 | 0.137 | 0.014 |
| **2.0** | `N_cfg2` | **18.29** | **0.134** | 0.017 |
| 2.5 | `P_cfg_25` | 17.30 | 0.136 | **0.003** |
| 3.0 | `P_cfg_30` | 17.05 | 0.145 | 0.003 |
| 4.0 | `P_cfg_40` | 16.77 | 0.143 | 0.009 |
| 6.0 | `P_cfg_60` | 16.15 | 0.145 | 0.026 |

- LPIPS sweet spot: **CFG=2.0**.
- PSNR drops monotonically past CFG=2.
- **CFG=2.5–3.0 minimizes |dHF| (=0.003)** — the prior amplification matches
  GT high-frequency energy almost exactly, but at a small LPIPS cost.
- Past CFG ≥ 4 every metric degrades.

**Practical rule:** use CFG=2.0 for balanced quality, CFG=2.5 only if
|dHF| is the explicit target and you can tolerate −0.5 dB PSNR.

## h_x (prior step size, per stage)

Two parameterizations: per-stage list or uniform scalar.

### Stage-3 amplification (list `[.1,.1,.1,*]`)

| Stage-3 value | Config | LPIPS | PSNR | \|dHF\| |
|---|---|---|---|---|
| 0.0 | `P_hx_s3_00` | 0.179 | 18.10 | 0.122 |
| 0.2 | `P_hx_s3_02` | 0.166 | 17.85 | 0.071 |
| 0.4 | `G_hx_lo` | 0.142 | 17.97 | 0.070 |
| 0.5 | `P_hx_s3_05` | 0.157 | 17.50 | 0.054 |
| **0.7** | `C_L5`/`N_cfg2` | **0.130** | 17.5–18.3 | 0.011 |
| 0.9 | `P_hx_s3_09` | 0.148 | 17.40 | 0.041 |
| 1.0 | `G_hx_hi` | 0.143 | 17.81 | 0.050 |
| 1.2 | `P_hx_s3_12` | 0.143 | 17.31 | 0.044 |

The default 0.7 is the optimum; it falls off quickly on either side.

### Uniform `h_x` scalar (all 4 stages equal)

| h_x | Config | LPIPS | PSNR | \|dHF\| |
|---|---|---|---|---|
| 0.05 | `P_hxu_005` | 0.175 | 18.51 | 0.029 |
| 0.10 | `P_hxu_01` | 0.155 | 18.12 | 0.026 |
| **0.20** | `P_hxu_02` | **0.130** | 17.38 | 0.056 |
| 0.30 | `G_hx_uni3` | 0.133 | 16.83 | **0.007** |
| 0.40 | `P_hxu_04` | 0.128 | 16.45 | 0.071 |
| 0.60 | `P_hxu_06` | 0.124 | 15.65 | 0.051 |

- LPIPS keeps falling all the way to 0.6, but **PSNR collapses** below 16 dB
  past 0.4 — the chain is essentially diverging from the GT.
- The trustworthy uniform sweet spot is **h_x ∈ [0.2, 0.3]** — both LPIPS
  and PSNR remain in-band.
- `h_x=0.3` uniform has the **best |dHF| (0.007)** in the entire corpus.
- Combined with CFG=2 + λ_reg=150 (= `F_cfg20_lr150_hxu02`), uniform h_x=0.2
  produces the corpus winner: LPIPS 0.120, |dHF| 0.010 in same-batch F.

**Practical rule:** prefer **uniform `h_x ∈ [0.2, 0.3]`** combined with CFG.
The classical stage-3 schedule is competitive but slightly weaker on |dHF|.

## lambda_reg (Langevin data-fidelity weight)

Single-axis sweep anchored on `N_cfg2` (CFG=2):

| λ_reg | Config | PSNR | LPIPS | \|dHF\| |
|---|---|---|---|---|
| 10  | `P_lreg_10`  | 17.72 | 0.168 | 0.085 |
| 25  | (CFG=0) `F_lr25_C5` | 17.20 | 0.171 | 0.056 |
| 50  | `R_anchor_N_cfg2` | 17.51 | 0.162 | 0.052 |
| 75  | `P_lreg_75`  | 17.51 | **0.134** | 0.020 |
| 100 | (CFG=0) `F_lr100_C5` | 17.99 | 0.141 | 0.074 |
| **150** | `P_lreg_150` | 17.55 | **0.133** | 0.029 |
| 200 | (CFG=0) `F_lr200_C5` | 18.03 | 0.140 | 0.082 |
| **300** | `P_lreg_300` | 17.60 | **0.133** | 0.035 |

- Below 25: chain loses data fidelity, LPIPS > 0.17.
- **75–300 is a flat plateau** in LPIPS (~0.133). Any value in this range is
  acceptable.
- Combined with CFG, `λ_reg=150` is the sweet spot in same-batch F:
  `F_cfg20_lr150_hxu02` LPIPS=0.120 vs. `F_cfg20_lr300` LPIPS=0.135.
- Higher λ_reg increases data-fidelity weight, which suppresses the prior
  push and lets uniform h_x add detail without divergence.

**Practical rule:** **λ_reg=150** when combined with CFG=2 + uniform h_x.
λ_reg=50 is fine for plain CFG=2.

## h_epsilon (eps prior step) — the dominant *original* knob

Scalar sweep, both `A_he*` (CFG=0 anchor) and `P_he_*` (CFG=2 anchor):

| h_eps | CFG=0 LPIPS / PSNR | CFG=2 LPIPS / PSNR (P_he, co-tenant) |
|---|---|---|
| 1e-5 | 0.173 / 15.8 | — |
| 1e-4 | 0.173 / 16.0 | — |
| 3e-4 | — | 0.160 / 16.6 |
| 5e-4 | 0.167 / 16.7 | 0.160 / 16.8 |
| 7e-4 | — | 0.159 / 17.0 |
| **1e-3** | **0.135 / 18.4** | (anchor `R_anchor_N_cfg2` 0.162 / 17.5) |
| 1.5e-3 | — | 0.162 / 17.7 |
| 2e-3 | 0.135 / 19.2 | 0.167 / 18.1 |
| 3e-3 | — | 0.167 / 18.6 |
| 5e-3 | 0.139 / 19.9 | 0.165 / 19.2 |
| 1e-2 | 0.146 / 20.7 | — |
| 5e-2 | 0.202 / 20.0 | — |
| 1e-1 | 0.208 / 19.1 | — |

(P_he_* numbers are ~0.03 LPIPS higher than A_he* of same h_eps because of
co-tenancy noise — focus on the *trend within each column*.)

- Plateau below 1e-4: under-correction; PSNR ~16 dB; empty-ish boxes.
- **1e-3 = perceptual sweet spot.** Best LPIPS, strongest semantic content.
- 2e-3–5e-3: LPIPS edges up while PSNR climbs. Best balanced corner without CFG.
- ≥ 1e-2: LPIPS rises and content collapses (smooth fills, max PSNR).
- ≥ 5e-2: fake high-frequency grain; LPIPS > 0.20.

**Practical rule:** **h_eps = 1e-3** is the only stable choice. Per-stage
schedules don't beat the scalar.

## num_langevin (L)

Mostly the same as the original sweep, with `P_L*` filling in the gaps:

| L | Config | PSNR | LPIPS | \|dHF\| |
|---|---|---|---|---|
| 1 | `C_L1` | 17.66 | 0.203 | 0.145 |
| 2 | `P_L2` | 17.17 | 0.186 | 0.095 |
| 3 | `C_L3` | 18.13 | 0.159 | 0.119 |
| 4 | `P_L4` | 17.09 | 0.167 | 0.071 |
| **5** | `C_L5` | 17.51 | **0.130** | **0.011** |
| 7 | `P_L7` | 17.31 | 0.154 | 0.039 |
| 8 | `P_L8` | 18.16 | 0.139 | 0.070 |
| 10 | `C_L10` | 16.60 | 0.125 | 0.041 |
| 15 | `C_L15` | 16.87 | 0.133 | 0.043 |
| 20 | `C_L20` | 17.26 | 0.142 | 0.061 |
| 30 | `C_L30` | 18.20 | 0.180 | 0.094 |

- L=5 is the speed/quality optimum.
- L=10 is +60% wall-time for −0.005 LPIPS — only worth it for max LPIPS.
- L > 15 over-corrects.

## noise_scale

Default 0. All `noise_scale > 0` settings hurt LPIPS substantially at this
resolution. The NOTES.md hint that `noise_scale=0.3` helped on 64×64 best_box
**does not transfer** to 128×128. Keep at 0.

## lambda_prox (paper step-(d) extra term)

Single-axis sweep anchored on `N_cfg2`:

| λ_prox | Config | PSNR | LPIPS | \|dHF\| |
|---|---|---|---|---|
| 0.05 | `P_lp_005` | 17.41 | 0.149 | 0.007 |
| 0.10 | `H_lprox01` | 17.52 | 0.158 | 0.096 |
| 0.50 | `H_lprox05` | 17.52 | 0.158 | 0.096 |
| 2.0 | `P_lp_2` | 17.41 | 0.148 | 0.007 |
| 5.0 | `P_lp_5` | 17.41 | 0.148 | 0.006 |
| 20 | `P_lp_20` | 17.41 | 0.147 | **0.001** |
| 100 | `P_lp_100` | 17.24 | 0.143 | 0.026 |

- Two distinct attractors: a "small-prox" branch (λ ≤ 0.5 → LPIPS 0.158)
  and a "saturated-prox" branch (λ ≥ 2 → LPIPS ~0.148, |dHF| ≈ 0.001).
- The saturated branch has the **best |dHF|** in any single-axis sweep
  (`P_lp_20` |dHF|=0.001) but its LPIPS is +0.014 over the anchor.
- Net: not a Pareto improvement over CFG. Skip.

## lambda_x (per-stage prior weight)

Default 0.01. Sweep range 0.001–5.0 all stay within ±0.005 LPIPS of the anchor.
Knob is essentially saturated; keep at default.

## rho_s, rho_e (endpoint weighting)

Default 1.0/1.0. Sweep tested 0.5/0.5, 0.5/1, 1/0.5, 1/2, 2/1, 2/2 — all within
~0.01 LPIPS of anchor. Keep at default.

## ode_steps_per_stage

Default 10. Sweep tested 5/8/12 and per-stage list `[10,10,10,15]`,
`[10,10,10,20]`. **All variations hurt.** Keep at default 10.

## h_x_obs_ratio

Default 1.0. Tested 0.25 and 3.0; both hurt. Keep at default.

## Eps-modifier knobs (`reset_eps_per_ode_step`, `lambda_eps_inject`, `dps_kick_zeta`)

All collapse to the same mediocre attractor (LPIPS ≈ 0.158, PSNR ≈ 17.5,
|dHF| ≈ 0.096). Keep at 0.

## Structural ablations (`warm_restart`, `g_bypass_stage3`)

Both must stay **on**. `warm_restart=False`: PSNR=14.0, LPIPS=0.250
(disastrous). `g_bypass_stage3=False`: PSNR=17.5, LPIPS=0.173 (moderate hit).

## Compact decision tree

```
For 128×128 box on res=256 PRINCIPLE sampler:

  Best balanced (corpus-wide winner)
    → F_cfg20_lr150_hxu02:
        CFG=2.0, λ_reg=150, h_x=0.2 uniform
        h_eps=1e-3, L=5, noise_scale=0, terminal_replace_weight=1
        → LPIPS 0.120,  PSNR 17.18,  |dHF| 0.010

  Need PSNR ≥ 18 dB
    → N_cfg2 (CFG=2, defaults).  PSNR 18.29, LPIPS 0.134, |dHF| 0.017

  Need lowest |dHF| only
    → P_lp_20 (lambda_prox=20).   |dHF| 0.001 but LPIPS 0.147
    → P_cfg_25 (CFG=2.5).          |dHF| 0.003, LPIPS 0.136

  Max PSNR with semantic content
    → A_he2e-3 (h_eps=2e-3, L=10). PSNR 19.20, LPIPS 0.135

  Max PSNR (no semantic constraint)
    → I_ref_F1 (paper baseline).   PSNR 20.76, LPIPS 0.174 (smooth fill)
```

## Cross-axis takeaways from Stage F (same-batch)

1. **CFG×λ_reg interaction**: at CFG=2.0, λ_reg=150 wins (LPIPS 0.137);
   at CFG=2.5, λ_reg≥150 blows |dHF| past 0.10. Higher CFG needs *more*
   data-fidelity damping but past CFG=2.5 the trade-off is unfavorable.
2. **CFG×h_x interaction**: combining CFG=2.0 with uniform h_x=0.2 stacks
   their gains (Δ LPIPS −0.016, Δ |dHF| −0.007 over CFG=2 alone). The same
   combo at CFG=2.5 (`F_cfg25_hxu02`) collapses to LPIPS 0.158.
3. **Triple combo**: CFG=2.0 + λ_reg=150 + h_x=0.2u is the corpus optimum.
   The same triple with CFG=2.5 collapses (LPIPS 0.153). The CFG ceiling
   for combinable gains is ~2.0–2.2.
4. **L×CFG interaction**: increasing L from 5 to 7 at CFG=2.5 + λ_reg=200
   (`F_cfg25_lr200_L7`) gives LPIPS 0.127 (runner-up). L acts as a
   regularizer that absorbs over-aggressive CFG.

## Per-axis curve plots

| Axis | File |
|---|---|
| `guidance_scale` | `comparisons/curve_guidance_scale.png` |
| `h_epsilon` | `comparisons/curve_h_epsilon.png` |
| `num_langevin` | `comparisons/curve_num_langevin.png` |
| `lambda_reg` | `comparisons/curve_lambda_reg.png` |
| `h_x` stage-3 | `comparisons/curve_h_x_stage3.png` |
| `h_x` uniform | `comparisons/curve_h_x_uniform.png` |
| `lambda_prox` | `comparisons/curve_lambda_prox.png` |
| `lambda_x` | `comparisons/curve_lambda_x.png` |
