# debug5 — Comprehensive PRINCIPLE Posterior Sampling Experiments

## Overview

Tested 100+ configurations across 6 experiment rounds to find the best posterior
sampling settings for box inpainting on frozen PixelFlow prior. Focus on the new
pseudocode term `λ(sg(x̂₁) - x₁)` (lambda_prox) and fundamentally different
stage-3 strategies.

All experiments use 2 test images (class 10, goldfinch), box mask (80-160 px),
sigma_n=0.05, terminal_replace=1.0 (free lunch from IP4).

## Key Findings

### 1. lambda_prox (pseudocode term) — works for metrics, not for visual content

The proximal term `λ(sg(x̂₁) - x₁)` provides a clean, tunable PSNR-HF tradeoff:

| Recommended Config | h_x stg3 | lambda_prox stg3 | L stg3 | CG | PSNR | |dHF| | Time |
|--------------------|-----------|-------------------|--------|-----|------|-------|------|
| **Max PSNR** | 0.1 | 0 | 10 | 20 | **15.74** | 0.072 | 266s |
| **Fastest** | 0.7 | 20 (stg3) | 5 | 20 | 15.58 | 0.053 | **99s** |
| **Best balanced** | 0.5 | 20 (stg3) | 10 | 20 | **15.25** | **0.018** | 132s |
| **Best HF match** | 0.7 | 0 | 10 | 20 | 14.98 | **0.005** | 239s |
| **= Best HF, faster** | 0.5 | 50 (stg3) | 10 | 20 | 14.96 | 0.008 | **130s** |

**However**: Visual inspection shows ALL these configs produce smooth, featureless
box regions. The HF metric improvements come from edge effects and subtle frequency
shifts, NOT from generating meaningful semantic content (bird feathers, branches, etc.).

### 2. Warm restart dominates stage-3 behavior

Tested 15 fundamentally different stage-3 strategies (DPS, replacement, Tikhonov,
L=1 with high lambda_prox, blending). **ALL gave identical results** (PSNR within
0.05 dB, HF within 0.002) because warm restart at each ODE step overwrites all
modifications:

```
At each step: x1_k = model_prediction  ← overwrites DPS/replace corrections
              DPS/replace applied      ← but overwritten at next step
```

Without warm restart: unobserved pixels never update (stuck at stage-2 values),
and DPS gradient is exactly zero on unobserved pixels for masking operators
(A^T(y-Ax) = 0 where mask=0).

### 3. Noise injection breaks the smoothness cycle

**Root cause of smoothness**: Model sees smooth xtau in unobserved region →
predicts smooth velocity → produces smooth x1_model → smooth xtau again.
Self-reinforcing cycle.

**Solution**: Inject noise into xtau on unobserved pixels before velocity prediction:
```python
xtau_input = xtau + alpha * noise * (1 - mask)
mu = velocity_fn(xtau_input)  # model forced to denoise → generates texture
```

Fine-grid sweep results (GT HF ≈ 0.68, no Langevin at stage 3):

| alpha | PSNR | HF | |dHF| | Visual quality |
|-------|------|-----|-------|----------------|
| 0.00 | 14.21 | 0.568 | 0.108 | Smooth, featureless |
| 0.05 | 14.19 | 0.571 | 0.104 | Smooth |
| 0.07 | 14.09 | 0.585 | 0.090 | Slight texture |
| 0.09 | 13.92 | 0.609 | 0.067 | Noticeable texture |
| 0.10 | 13.81 | 0.624 | 0.051 | Grainy texture |
| 0.12 | 13.50 | 0.660 | 0.015 | More grain |
| **0.13** | **13.32** | **0.680** | **0.005** | **Near-GT HF, noisy** |
| 0.15 | 12.82 | 0.727 | 0.051 | Overshoot |
| 0.18 | 11.92 | 0.792 | 0.116 | Too noisy |

**Honest assessment**: The texture from noise injection is random grain/noise,
not structured semantic content. HF matches GT because noise has high-frequency
energy. Visually, the box goes from "flat smooth" to "noisy texture" but NOT
"meaningful content" (e.g., bird feathers).

### 4. Adaptive noise scaling fails; constant alpha works

| Scaling method | Result |
|----------------|--------|
| Constant alpha | **Works** — clean linear PSNR-HF tradeoff |
| sigma_tau scaled | Fails — too much at early steps, too little at late |
| tau scaled | Fails — same issue |
| Multi-stage (s23, all) | Same as stage-3 only |

### 5. Langevin smooths away injected noise

Adding Langevin after noise injection is equivalent to reducing alpha:

| Config | alpha | L | PSNR | HF |
|--------|-------|---|------|-----|
| a010 | 0.10 | 0 | 13.81 | 0.624 |
| a010_L3 | 0.10 | 3 | 13.98 | 0.599 |
| a008 | 0.08 | 0 | 14.02 | 0.595 |

L=3 at alpha=0.10 ≈ alpha=0.08 without L. No synergy — Langevin just undoes
injection. For aggressive alpha (0.2), L=10 recovers +2 dB PSNR.

### 6. Dead ends documented

| Approach | Configs tested | Result |
|----------|---------------|--------|
| lambda_prox on X4 (h_x=0.7) | B1-B6 (6 configs) | Monotonically worse — h_x and lambda_prox are redundant |
| Matched lambda (lp=lr) reduced | D1-D4 (4 configs) | D1-D3 collapse; D4 = C4 |
| lambda_prox > lambda_reg | E1-E5 (5 configs) | All collapse |
| h_x > 1.0 with any setting | F3,F5-F8,K1-K3 | All collapse (PSNR < 11) |
| L > 10 with lambda_prox | H2-H5 (4 configs) | Over-amplifies prior → collapse |
| DPS at stage 3 (any zeta) | D_dps_z1-z20 (hybrid v2) | Zero effect on unobserved (A^T zero on mask=0) |
| Tikhonov at stage 3 | tikh_l50-200 | Same as warm restart (model prediction dominates) |
| Skip warm restart + DPS | DPS_nowr (hybrid v2) | Unobserved pixels never update |

## Recommended Settings

### Profile 1: Max PSNR (data fidelity)
```python
terminal_replace_weight = 1.0
h_x = 0.1              # default, no amplification
lambda_prox = 0.0       # off
num_langevin = 10       # default
cg_max_iter = 20        # CG=20 ≈ CG=50
# PSNR ≈ 15.7, box region = smooth
```

### Profile 2: Fastest balanced
```python
terminal_replace_weight = 1.0
h_x = [0.1, 0.1, 0.1, 0.7]     # IP4 X4 at stage 3
lambda_prox = [0, 0, 0, 20]     # stage-3 only
num_langevin = [10, 10, 10, 5]  # fewer at stage 3
cg_max_iter = 20
# PSNR ≈ 15.6, |dHF| ≈ 0.05, time = 99s (2.7x faster than baseline)
```

### Profile 3: Best PSNR-HF tradeoff
```python
terminal_replace_weight = 1.0
h_x = [0.1, 0.1, 0.1, 0.5]     # moderate stage-3 boost
lambda_prox = [0, 0, 0, 20]     # stage-3 proximal anchor
num_langevin = 10
cg_max_iter = 20
# PSNR ≈ 15.3, |dHF| ≈ 0.02, box = smooth but HF near-GT
```

### Profile 4: Best HF match (perceptual metrics)
```python
terminal_replace_weight = 1.0
h_x = [0.1, 0.1, 0.1, 0.5]
lambda_prox = [0, 0, 0, 50]
num_langevin = 10
cg_max_iter = 20
# PSNR ≈ 15.0, |dHF| ≈ 0.008, equivalent to IP4 X4
```

### Profile 5: Noise injection (experimental texture)
```python
terminal_replace_weight = 1.0
inject_alpha = 0.10              # on unobserved xtau before velocity
inject_stages = [3]              # stage-3 only
use_langevin_s3 = False          # skip Langevin (it smooths away texture)
num_langevin_stages012 = 10
cg_max_iter = 20
# PSNR ≈ 13.5-13.8, |dHF| ≈ 0.01-0.05
# Box has grainy texture (NOT semantic content)
# alpha ∈ [0.08, 0.13] — linear PSNR-HF dial
```

## Structural Insights

1. **h_x and lambda_prox are redundant** at stage 3. Both amplify prior through
   the CG-preconditioned update. Use one or the other, not both at full strength.
   h_x=0.5 + lp=50 ≈ h_x=0.7 alone.

2. **CG=20 is sufficient** everywhere (vs CG=50). CG=5 works too with minor
   quality loss. 2-5x speedup.

3. **Langevin at stage 3 adds negligible value** (0.02 dB) but costs 2x time.
   The model prediction (warm restart) already captures the optimal x1 estimate.

4. **terminal_replace_weight=1.0** is always-on free lunch. Zero cost, res=0,
   +0.02 dB PSNR.

5. **The smoothness of unobserved regions is a model-level fixed point**, not an
   algorithm deficiency. The ODE flow converges to smooth predictions regardless
   of what inner loop we use. Breaking this requires either (a) noise injection
   before velocity prediction, or (b) DAPS-style full-image denoising with
   observation projection.

## File Manifest

```
debug5/
├── EXPERIMENT_SUMMARY.md              # this file
├── run_sweep.py                       # 52-config parameter sweep (Groups A-L)
├── run_round2.py                      # Round 2 with h_epsilon + lambda_prox
├── plot_interim.py                    # Interim analysis plotting
├── visualize_key_configs.py           # 12-config image comparison
├── test_hybrid_stage3.py              # 15-config hybrid v1 (warm restart issue)
├── test_hybrid_v2.py                  # 15-config hybrid v2 (no warm restart)
├── test_noise_inject.py               # 22-config noise injection test
├── test_noise_finegrid.py             # 16-config fine-grid alpha sweep
├── results/                           # Sweep results + images
│   ├── sweep_results.json
│   ├── sweep_all.png
│   ├── top10_detail.png
│   └── interim_analysis.png
├── results_vis/                       # Key config visualizations (12 configs)
│   ├── comparison_grid.png
│   ├── box_crops_grid.png
│   └── *.png                          # per-config images
├── results_hybrid/                    # Hybrid v1 (warm restart)
│   └── hybrid_grid.png
├── results_hybrid_v2/                 # Hybrid v2 (no warm restart)
│   └── hybrid_v2_grid.png
├── results_noise_inject/              # Noise injection (22 configs)
│   ├── noise_inject_grid.png
│   └── *.png
└── results_finegrid/                  # Fine-grid alpha sweep (16 configs)
    ├── finegrid.png
    └── *.png
```

## Comparison with Previous Experiments

| Experiment | Best PSNR | Best |dHF| | Key finding |
|------------|-----------|------------|-------------|
| IP3 methods sweep | 13.25 (PRINCIPLE) | 0.005 (X4) | DPS z=10 matches GT HF |
| IP4 round 1-3 | 19.88 (F1) | 0.001 (HXL2) | h_x stage-3 is dominant lever |
| IP4 round 4 | 14.76 | 0.006 | lambda_prox alone has minimal effect |
| **debug5 sweep** | **15.74** | **0.005** | lambda_prox + h_x are redundant |
| **debug5 hybrid** | 15.05 | 0.005 | Warm restart erases all stage-3 modifications |
| **debug5 noise inject** | 13.32-14.21 | **0.005** | Noise injection produces texture (grainy) |

## Path Forward

To achieve DAPS-comparable visual quality (structured inpainting content):

1. **DAPS-style approach**: Start from pure noise, denoise full image, project
   observations at each step. Requires modified ODE solver — not parameter tuning.

2. **Guided generation**: Use the frozen model's class-conditional generation
   with observation guidance, rather than posterior sampling on a fixed trajectory.

3. **Model fine-tuning**: Add inpainting-aware training objective to the PixelFlow
   model (breaks the "frozen prior" constraint).

Current framework's capability boundary: excellent data consistency (PSNR, SSIM)
on observed pixels + smooth/noisy fills on unobserved pixels. Structured content
generation requires architectural changes beyond parameter tuning.
