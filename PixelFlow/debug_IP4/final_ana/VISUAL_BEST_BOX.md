# Visual-Best BOX Configurations — ignoring PSNR

15 aggressive configurations tested on box inpainting (GT HF=0.584).
Ranked purely by **visual box-content quality** (natural-looking bird,
clean texture, no over-saturation or fragmentation).

Images: `visual_best_box/images/*.png`
Side-by-side: `visual_best_box/comparison.png`
Full configs:  `visual_best_box/configs/*.json`
Numeric data:  `visual_best_box/metrics.json`

---

## 🏆 Top Tier — natural, clean bird reconstruction

| # | Config | Setting | PSNR | HF | Visual notes |
|---|--------|---------|------|----|--------------|
| 🥇 **1** | **A0_reference_WINNER3** | `h_eps=0.001 + h_x stg3=0.7 + F1` | 20.56 | 0.684 | Sharp, natural bird; good head/body detail |
| 🥈 **2** | **E1_W1+lam20** | `h_eps=0.001 + lambda_reg stg3=20 + F1` | **22.17** | 0.631 | Cleanest, smoothest (CG softening path). **Best PSNR of the perceptual configs** |
| 🥉 **3** | **C1_W1+hx_multistage** | `h_eps=0.001 + h_x=[0.1,0.2,0.4,0.7] + F1` | 20.06 | 0.703 | Progressive h_x ramp — coherent structure |
| 4 | **H1_W1+X4+noise0.3** | `WINNER3 + noise_scale=[0,0,0,0.3] + F1` | 20.31 | 0.692 | Natural grain/texture; feels photographic |

If you can only ship **one**: pick **E1_W1+lam20**. It gives a clear bird with
the highest PSNR among content-generating configs (22.17 dB, only −0.7 from
baseline).

If you want **maximum perceptual sharpness** regardless of PSNR: pick **A0
(WINNER3)**. Classic recipe, the one used throughout the debug_IP4 work.

## ⚠️ Middle Tier — over-processed (saturated / high-contrast)

These produce a bird but the colour palette becomes unnatural / over-sharp:

| # | Config | Issue |
|---|--------|-------|
| B1 | `h_x stg3=1.0` | Too saturated |
| B2 | `h_x stg3=1.5` | Heavy over-saturation |
| C2 | `h_x=[0.2,0.4,0.7,1.0]` | High-contrast everywhere |
| D1 | `h_eps=1e-4 + h_x=0.7` | Grain starts dominating |
| D2 | `h_eps=1e-4 + h_x=1.0` | Saturation + grain combine |
| F1 | `V + X4 + lam20` | Triple-lever overshoot, HF=0.798 (way over) |
| F2 | `W1 + X4 + lam20` | HF=0.792 overshoot |
| G1 | `WINNER3 + L=20 stg3` | Over-sharp, noisy edges |
| E2 | `W1 + lam10` | HF overshoots GT |

## ❌ Failed Tier — fragmentation or no effect

| # | Config | Issue |
|---|--------|-------|
| I1 | `WINNER3 + CFG=2.0` | **CFG dead** — identical to A0 (retest confirmed) |
| Z_ULTIMATE | all levers pushed max | HF=0.894 (way over), PSNR 13.61, **pattern fragmentation** |

---

## Recommendations by scenario

| What you want | Ship this |
|---------------|-----------|
| **Best pure PSNR (smooth fill)** | Don't use any of these — just baseline + F1 |
| **Best balanced (content + high PSNR)** | **E1 `h_eps=0.001 + lam_reg stg3=20 + F1`** |
| **Best visual bird (all-out perceptual)** | **A0 WINNER3 `h_eps=0.001 + h_x stg3=0.7 + F1`** |
| **Most natural-looking (no artifacts)** | **C1 `h_eps=0.001 + h_x=[0.1,0.2,0.4,0.7] + F1`** |
| **"Photographic" with grain** | **H1 WINNER3 + noise_scale stg3=0.3** |

**Avoid**: any combination of >=2 aggressive levers (F/Z series — they all overshoot).
The winning principle: **pull one lever firmly, not three gently**.

---

## File manifest

```
visual_best_box/
├── configs/            — JSON for all 15 configs (each with "desc" field)
├── images/             — individual PNGs (256x256)
│   ├── GT.png
│   ├── A0_reference_WINNER3.png
│   ├── B1_W1+X_hx1.0.png … Z_ULTIMATE.png
├── comparison.png      — 4x4 grid with red box overlay
└── metrics.json        — all PSNR/HF numbers
```
