# TOP Configurations — Distilled from 220+ experiments

7 representative configs, ordered by increasing perceptual-aggressiveness.
Re-run fresh on BOTH **box** (GT HF=0.716) AND **random p=0.8** (GT HF=0.577)
with publication-convention PSNR (`10·log10(4/MSE)` for `[-1,1]` data,
mask-aware denominators). All configs enable `terminal_replace_weight=1.0`
so observed pixels are exact (`PSNR_obs=∞`, `res=0`) — differences are purely
in what the sampler generates for the **unobserved** region.

Reconstructions: `top_configs/box/*.png`, `top_configs/random/*.png`.
Side-by-side grids: `top_configs/{box,random}/comparison.png`.

---

## The 7 configurations

| ID | Name | `h_epsilon` schedule (stg 0/1/2/3) | Extra | Rationale |
|----|------|------------------------------------|-------|-----------|
| **0** | `ref_baseline_F1`        | `[0.01, 0.01, 0.01, 0.01]`          | + F1 | No eps-tempering; only free data-consistency |
| **1** | `S1_stage1_only_F1`      | `[0.01, 0.001, 0.01, 0.01]`         | + F1 | Minimal: touch the critical stage-1 only |
| **2** | `P12_stages12_F1`        | `[0.01, 0.001, 0.001, 0.01]`        | + F1 | Balanced: protect eps entropy through mid-chain |
| **3** | `T012_LO_LO_LO_HI_F1`    | `[0.001, 0.001, 0.001, 0.01]`       | + F1 | W1-equivalent perceptual, 25% cheaper (stg-3 HI) |
| **4** | `W1_global_heps_0.001_F1`| `[0.001, 0.001, 0.001, 0.001]`      | + F1 | Classic W1: global low h_eps |
| **5** | `WINNER3_W1+X4+F1`       | `[0.001, 0.001, 0.001, 0.001]`      | + h_x stg-3=0.7, + F1 | W1 + prior-score amplifier |
| **6** | `V_all_1e-4_F1`          | `[1e-4, 1e-4, 1e-4, 1e-4]`          | + F1 | Extreme — max eps preservation |

All other knobs identical to IP3 default (`lambda_reg=50`, `num_langevin=10`,
`cg_max_iter=50`, `warm_restart=true`, `g_bypass_stage3=true`).

---

## BOX inpainting (GT HF = 0.716)

| # | Config | PSNR_all | PSNR_unobs | HF | \|ΔHF\| | Visual verdict |
|---|--------|----------|------------|-----|---------|----------------|
| 0 | ref_baseline_F1        | **22.87** | **17.59** | 0.705 | 0.011 | smooth brown box — **NO bird** |
| 1 | S1_stage1_only_F1      | 22.57 | 17.29 | 0.702 | 0.014 | mostly smooth + tiny texture |
| 2 | P12_stages12_F1        | 22.28 | 17.00 | 0.707 | **0.008** | partial bird emerges |
| 3 | T012_LO_LO_LO_HI_F1    | 18.86 | 13.58 | 0.650 | 0.065 | **bird reconstructed** |
| 4 | W1_global_heps_0.001_F1| 18.89 | 13.61 | 0.648 | 0.068 | **bird reconstructed** |
| 5 | WINNER3_W1+X4+F1       | 18.19 | 12.91 | **0.699** | 0.016 | **sharp bird, high-contrast** |
| 6 | V_all_1e-4_F1          | 17.64 | 12.36 | 0.652 | 0.064 | bird + heavy grain |

**Takeaway**: this BOX mask is relatively small/lateral (GT_HF already 0.716),
so baseline's HF is close to GT. The perceptual "problem" is less severe —
baseline still fills smoothly (no bird body), but HF numerics don't punish it.

**Effective candidates for box**:
- Pure PSNR: **config 0** (baseline + F1) — acceptable when the box is small.
- **Best balanced**: **config 2 (P12)** — only −0.59 dB, starts showing content.
- Best semantic content: **config 5 (WINNER3)** — PSNR −4.7 dB but the bird
  has detailed head, body, feather structure.

---

## RANDOM inpainting p=0.8 (GT HF = 0.577)

| # | Config | PSNR_all | PSNR_unobs | HF | \|ΔHF\| | Visual verdict |
|---|--------|----------|------------|-----|---------|----------------|
| 0 | ref_baseline_F1        | **23.11** | **25.15** | 0.600 | 0.023 | clean faithful bird |
| 1 | S1_stage1_only_F1      | 22.76 | 24.80 | 0.602 | 0.024 | nearly identical to 0 |
| 2 | P12_stages12_F1        | 20.85 | 22.90 | 0.611 | 0.034 | mild graininess |
| 3 | T012_LO_LO_LO_HI_F1    | 20.87 | 22.91 | 0.610 | 0.033 | mild graininess |
| 4 | W1_global_heps_0.001_F1| 20.98 | 23.02 | 0.609 | 0.032 | mild graininess |
| 5 | WINNER3_W1+X4+F1       | **16.34** | **18.38** | 0.682 | 0.105 | **significant noise** — bird body degraded |
| 6 | V_all_1e-4_F1          | 20.55 | 22.59 | 0.612 | 0.035 | slightly grainy |

**Takeaway**: random p=0.8 provides dense per-pixel observation so the baseline
already has sharp reconstructions (PSNR_unobs=25.15). Preserving eps entropy
turns into **visible noise instead of plausible texture**. WINNER3's h_x=0.7
amplifies the prior score on an already-anchored latent, creating global noise
artifacts (PSNR −7 dB!).

**Effective candidates for random**:
- **Winner: config 0 (baseline + F1)** — don't touch h_eps.
- Config 1 (S1 only) is acceptable with <0.4 dB loss.
- Configs 3-5 cost 2-7 dB with no visual gain — **do not use** for random.

---

## Cross-mask recommendation table

| Priority | Box config | Random config |
|----------|------------|----------------|
| Max PSNR | **0 (baseline + F1)** | **0 (baseline + F1)** |
| Balanced (PSNR + perceptual) | **2 (P12_stages12)** | **0 (baseline + F1)** |
| Max semantic content | **5 (WINNER3)** | **0 (baseline + F1)** — random doesn't need it |
| HF bullseye | 2 (P12), accuracy 0.008 on this mask | n/a (baseline already close) |

**Key point**: the RIGHT config depends heavily on mask topology.
- **Box (contiguous hole, low info)** — eps-entropy-preservation configs (3, 4, 5)
  actually generate content. Choose aggressiveness per compute / perceptual needs.
- **Random (sparse, high info)** — baseline is already near-optimal. Any
  h_eps reduction below 0.01 adds noise without visual benefit.

---

## What changed vs `ms_posterior_sampling_article_version_final`

Default `ms_posterior_sampling_article_version_final.json` = config **0 without F1**
(no `terminal_replace_weight`). So:

- **Config 0** = baseline + `terminal_replace_weight=1.0` → adds **exact observed-pixel
  match (PSNR_obs=∞, res=0)** at zero cost. **Always enable this**.
- Configs 1-6 add `h_epsilon` scheduling (scalar or list) on top.
- Config 5 additionally adds `h_x` stage-3 list.

Code changes required to main file: support list-typed `h_x` / `h_epsilon` in the
config loader, and call `terminal_replacement(x1, y, mask_full, weight)` after the
final ODE step. These are already implemented in
`debug_IP4/ms_posterior_sampling_IP4.py` (drop-in replacement).

---

## File manifest under `final_ana/top_configs/`

```
top_configs/
├── configs/            — JSON for each of 7 configs (with rationale field)
│   ├── 0_ref_baseline_F1.json
│   ├── 1_S1_stage1_only_F1.json
│   ├── 2_P12_stages12_F1.json
│   ├── 3_T012_LO_LO_LO_HI_F1.json
│   ├── 4_W1_global_heps_0.001_F1.json
│   ├── 5_WINNER3_W1+X4+F1.json
│   └── 6_V_all_1e-4_F1.json
├── box/
│   ├── GT.png
│   ├── 0_ref_baseline_F1.png … 6_V_all_1e-4_F1.png
│   └── comparison.png          — 2×4 side-by-side grid
├── random/                     — same set for random inpainting
│   └── comparison.png
└── metrics.json                — machine-readable PSNR/HF/res per config/task
```
