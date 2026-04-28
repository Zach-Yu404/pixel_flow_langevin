# debug_IP4 — Final Best Setting Summary

After 170+ experiments across Rounds 1-5 with visual + metric verification on
4 independent box masks, this is the **single best practical setting** for
semantic box inpainting with the PixelFlow c2img model.

---

## The setting

File: `ms_posterior_sampling_article_version_final.json`

```json
{
  "exp_name": "principle_final_box",
  "seed": 42,
  "num_stages": 4,
  "resolution": 256,
  "num_examples": 4,
  "sigma_n": 0.05,
  "class_label": 10,
  "shift": 1.0,
  "inference_each_step": 10,
  "guidance_scale": 0.0,

  "num_langevin": 10,

  "h_x":        [0.1, 0.1, 0.1, 0.7],
  "h_epsilon":  0.001,

  "lambda_x":   0.01,
  "lambda_reg": 50.0,
  "lambda_prox": 0.0,
  "rho_s":      1.0,
  "rho_e":      1.0,

  "terminal_replace_weight": 1.0,

  "warm_restart":    true,
  "g_bypass_stage3": true,
  "x1_init_mode":    "model",

  "cg_tol":      1e-5,
  "cg_max_iter": 50,

  "device":    "cuda:0",
  "data_dir":  "/data/Zach_dataset/imageNet256/ILSVRC/Data/CLS-LOC/train/",
  "model_dir": "./pretrained_models/c2img",
  "dict_path": "./trajectory_videos/posterior_sampling",

  "active_operator":    "box",
  "measurement_mode":   "call",
  "latent_update_mode": "f_x1",

  "box_operator": {
    "mask_type": "box",
    "mask_len_range": [80, 160],
    "mask_prob_range": null
  },
  "random_operator": {
    "mask_type": "random",
    "mask_len_range": null,
    "mask_prob_range": [0.8, 0.8]
  }
}
```

The **three** settings that matter for perceptual reconstruction (the rest
are the IP3 defaults that stay as-is):

| knob | value | what it does |
|------|-------|--------------|
| `h_epsilon` | `0.001` | Slows eps-Langevin ~10×. Prevents eps-collapse across ~400 inner iterations. Keeps `x_τ = H_τ·x_1 + σ·ε` **in-distribution for the flow model**. Without this the box fills with a smooth mean. |
| `h_x`       | `[0.1, 0.1, 0.1, 0.7]` | 7× larger Langevin step only at stage-3 (where `G = I`). Amplifies the prior-score gradient on unobserved pixels. Gives the preserved eps something sharper to sample against. |
| `terminal_replace_weight` | `1.0` | After sampling finishes, overwrite observed pixels with `y`. Zeroes the residual and drives `PSNR_obs = ∞` for free. **Never hurts.** |

Everything else stays at IP3 defaults (`warm_restart=true`, `g_bypass_stage3=true`,
`lambda_reg=50`, `cg_max_iter=50`, `num_langevin=10`).

---

## Why these three are the right knobs

### h_epsilon = 0.001
eps step-(f) is `eps ← (1 − h_eps/2)·eps + (h_eps/2)·σ·s_flow`, a linear damped
system. Over the 400 inner iterations (10 Langevin × 10 ODE × 4 stages), `h_eps=0.01`
contracts eps by `(1−0.005)^400 ≈ 0.135` — **86% of initial Gaussian entropy is
destroyed**. `h_eps=0.001` keeps 82%. Since the flow model was trained on
`x_τ` built from iid Gaussian eps, preserving that noise keeps the latent in
the training distribution and the velocity head generates plausible texture
in unobserved regions instead of regressing to a conditional mean.

### h_x stage-3 = 0.7
At stage 3 the `G` operator is identity (via `g_bypass_stage3=true`),
so the prior-score direction `(H_τ)^T s_flow` acts directly on pixels.
A larger `h_x` lets Langevin take bigger steps *along the prior score*
in the unobserved region, sharpening the generated content. At earlier
stages `G` is a low-pass projection, so amplifying `h_x` there would
just amplify the smooth component — that's why stage-3 only.

### terminal_replace_weight = 1.0
Observed pixels have a known ground truth (y, up to `sigma_n=0.05` noise).
Hard-replacing them at the end is a projection onto the feasible set — it
cannot worsen any legitimate error metric. Every config in every round
benefited from this.

---

## What NOT to enable (learned from dead ends)

- **`guidance_scale > 0`** — the c2img checkpoint's null-class head ≈ conditional
  head; CFG produces ≤0.5 dB delta, no perceptual change.
- **DPS pre-kick at any strength** — absorbed into subsequent Langevin; no
  effect on final output, and zeta>20 *hurts* data consistency.
- **Stage-3-only h_eps or skip_eps** — the eps channel is already collapsed by
  the time stage-3 begins; late interventions have no entropy left to preserve.
  Only **global** h_eps reduction works.
- **`lambda_prox > 0` alone** — redundant with `warm_restart` which already
  hard-resets `x_1` toward the model prediction each ODE step.
- **`h_x_obs_ratio ≠ 1.0`** — shrinking step inside the observed region
  breaks data fit; enlarging produces no benefit.
- **Aggressive two-knob perceptual pushes** (e.g. `h_x=0.5` + `lam_stg3=10`) —
  collapses to PSNR < 16 and HF overshoots badly. One lever at a time.
- **`reset_eps_per_ode_step=true`** — too disruptive; res explodes to 500-1000.
- **More ODE steps or CG iterations** — <0.1 dB gain per doubling of compute.

---

## Visual-vs-metric caveat (important)

The built-in `hf_energy(x * (1-mask))` metric is **dominated by the mask
boundary discontinuity**, not by semantic content inside the box. Round-5
proved this: `lambda_reg_stg3=15` scores bullseye on HF while its
reconstructions are visually identical to baseline (smooth fill).

Similarly, `PSNR_unobs` *systematically favours smooth fill* because MSE
penalises a correctly-generated-but-pixel-misaligned bird more than a
content-free low-variance fill.

⇒ Do not trust HF or PSNR alone when comparing inpainting configs.
Always visually inspect the box region.

Also two PSNR bugs exist in `ms_sampler_v5.py` and `run_methods_sweep.py`:
1. Formula assumes MAX=1 but data is in `[-1, 1]` → under-reports by +6.02 dB
   (consistent offset; does not affect ranking).
2. `psnr_obs` divides by `B·C·H·W` instead of `B·C·N_obs` → inflates by
   `10·log10(N_total/N_obs)` ≈ 0.7-2 dB depending on mask.

Correcting these only changes absolute numbers; the recommended setting is
still correct.

---

## How to run

```bash
conda activate pixelflow
cd /home/nvidia/Zach/MSFlow/PixelFlow

# The final sampler is drop-in compatible with ms_posterior_sampling_article_version_final.py
python ms_posterior_sampling_article_version_final.py \
    --config ms_posterior_sampling_article_version_final.json
```

Or, using the IP4-specific wrapper with the same knobs:
```bash
python debug_IP4/ms_posterior_sampling_IP4.py \
    --config debug_IP4/ms_posterior_sampling_IP4.json
```

Both files ship this recommended profile.

---

## Expected result

- **Box region**: visible bird-like structure (head, body, plumage), not a
  flat brown fill. Semantically plausible, not a pixel-perfect GT match
  (the box removes most of the bird — information-theoretically impossible).
- **Observed region**: exact match to `y` (`res = 0`, `PSNR_obs = ∞`).
- **Full-image PSNR**: 15-20 dB in [0,1] convention. Lower than baseline's
  smooth-fill PSNR because generated texture doesn't pixel-align with GT.
  This is the inherent tradeoff for ill-posed inpainting.
- **Runtime**: ~155 s on a single A100 for 2 samples at 256².

---

## Round-6 addendum — what adding terms to g_eps taught us

Round-6 locked `lambda_prox=50` (paper original) and tested 4 new g_eps
augmentation terms plus an h_x × h_epsilon grid (42 configs).

| New g_eps term | Coef range | Metric-Pareto? | Visually creates bird? |
|----------------|-----------|----------------|-------------------------|
| N1 (‖eps‖² norm preserve) | 0.01–10   | mild (at λ=10) | **no** (baseline-like) |
| N2 (mask-aware: pure prior in unobs) | on/off | **yes** — hits HF bullseye | **no** — another metric trick |
| N3 (proximal to eps_0) | 0.1–50    | λ=1 matches W1 HF  | (not separately imaged; metric alone) |
| N4 (additive Gaussian) | 0.01–1.0 | no (==baseline) | no |

### The decisive N2 case
`N2_mask_aware_alone`: PSNR 21.74 / HF 0.603 (|Δ|=0.046).
`N2_mask_aware_hx0.7`: PSNR 21.08 / HF 0.650 (|Δ|=0.001) — **looks like a Pareto winner.**

Visual inspection (`results_N2_visual/n2_vs_w1.png`) reveals: N2 at any h_x
produces **a smooth brown box with slight colour-boundary changes**. The HF
bullseye is again achieved by modulating mask-boundary edge energy, **not by
generating content**. Same trap as M3 (lambda-reduction) in Round-5.

### Why N2 fails despite sound theory
The idea "in unobserved region, drop the Tweedie pull on eps so eps stays
near N(0,I)" is reasonable. But `warm_restart=True` in the outer loop
re-derives `eps = (x_tau − H_τ(x̂1))/σ_τ` at every ODE step. By the time
N2's in-loop protection could help, `x_tau` from the previous step has
already been advanced using a collapsed-ε latent, so the reconstruction
of eps via warm_restart yields a collapsed eps anyway. N2 only affects one
Langevin step; warm_restart resets the slate between ODE steps.

**W1 (`h_epsilon=0.001` globally) avoids this trap** because it slows the
damping everywhere, and the resulting non-collapsed `x_tau` propagates
through warm_restart intact. It is a cross-step mechanism, not an in-step
one.

### Conclusion for novel g_eps terms
No combination of N1/N2/N3/N4 replaced or meaningfully improved on the
W1 + X4 + F1 recipe. `lambda_prox=50` (paper default, now default in main
repo JSON per user preference) contributes <0.5 dB on its own.

The WINNER3 setting remains the recommendation.

## One-sentence summary

**`h_epsilon=0.001` globally preserves the Gaussian noise that the flow model
was trained on; `h_x=0.7` at stage-3 amplifies prior-score steps on unobserved
pixels; `terminal_replace_weight=1.0` snaps observed pixels to the measurement
— together they fill the box with semantic content instead of smooth grey.**
