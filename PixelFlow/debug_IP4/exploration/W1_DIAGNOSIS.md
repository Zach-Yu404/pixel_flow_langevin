# W1 Diagnosis — Why low h_epsilon produces meaningful box content

Evidence-based analysis, ordered by user's requested output.

## 1. Code-level diagnosis

### How eps behaves now (`debug_IP4/ms_sampler_v5.py`)

eps is updated / reset in **five distinct places** per sample:

| # | Location | What happens to eps_k |
|---|----------|-----------------------|
| 1 | stage-0 init (L107)                  | `eps_k ~ N(0, I)` |
| 2 | stage transition si>0 (L128)         | re-derived from upsampled latent: `eps = (lat − s_k·G(x1))/(1−s_k)` |
| 3 | every ODE step, warm_restart (L175)  | re-derived from *consistency*: `eps = (x_tau − H_τ(x1_hat))/σ_τ`. **x1_k is simultaneously hard-overwritten to x1_hat** |
| 4 | optional per-ODE reset (L180)         | `eps_k ~ N(0, I)` fresh (new J-axis knob) |
| 5 | inside `principle_langevin_v5` step(f) | `eps += (h_eps/2)(σ·s_flow − eps) + √h_eps · ξ` (Langevin drift toward Tweedie fixed point + noise) |

**The key insight from (3) is non-obvious**: when `warm_restart=True`, the Langevin
refinement of x1 is **not** discarded — it is *absorbed into eps* via the
consistency equation, because x1_k gets hard-overwritten and eps re-derives
from whatever `x_tau` was built from the previous-step's refined x1.
So the entire inter-step memory of posterior refinement flows through eps.

Now look at the eps update (5). Writing `g_eps = σ · s_flow − eps`, the dynamics are:
```
eps_{t+1} = (1 − h_eps/2)·eps_t + (h_eps/2)·σ·s_flow + √h_eps · ξ_noise
```
This is a **linear damped system** with fixed point `eps* = σ · s_flow`. As x1
converges to x̂1 during Langevin refinement, `H_τ(x̂1 − x1) → 0`, so
`s_flow ∝ (H_τ(x̂1) − x_tau)/σ²` stays small, driving `eps* → 0`. Thus:

- **Default h_eps=0.01** over 400 inner iterations: contraction
  `(1 − 0.005)^400 ≈ 0.135` — eps keeps only **14%** of initial entropy, and the
  Tweedie-driven fixed point is nearly zero → **eps collapses** to a near-deterministic quantity.
- **W1 h_eps=0.001**: `(1 − 0.0005)^400 ≈ 0.818` — eps retains **82%** of its
  initial stochasticity, and the forward path's `x_tau = H_τ(x1) + σ·eps`
  stays close to the flow model's training distribution.

### Does the "over-shrunk eps ⇒ OOD latent ⇒ smooth fill" hypothesis hold?

**Yes, but only as one of two coupled effects**. The second effect is the
one I did not initially appreciate:

- **Effect A (entropy collapse)**: eps → 0 makes `x_tau ≈ H_τ(x1)` which is
  noise-free. The flow model, trained on noisy `x_tau`, sees OOD input and
  regresses to a smooth conditional-mean → the box fills with a color constant.
- **Effect B (loss of refinement memory)**: because warm_restart routes the
  Langevin's x1 refinement through eps, damping eps simultaneously erases
  the accumulated posterior correction. Pure "model prediction without
  posterior correction" looks smooth in the unobserved region because the
  model's one-step prediction for that region has no data signal to lock onto.

W1's `h_eps=0.001` preserves both — entropy AND refinement memory.
That is why it is a single-knob fix with outsized effect.

## 2. PSNR code audit (**bugs found**)

Code: `ms_sampler_v5.py` L224 & `debug_IP3/run_methods_sweep.py` L222.

```python
psnr_all = -10 * torch.log10(((xf - gt) ** 2).mean()).item()
psnr_obs = -10 * torch.log10(((mask_full * xf - mask_full * gt) ** 2).mean()).item()
```

Input range: `[-1, 1]` (transforms.Normalize([0.5]*3, [0.5]*3) on ToTensor output).

### Bug A — `psnr_all` under-reports by exactly +6.02 dB
Signal range is `[-1, 1]` (dynamic range 2), but the formula assumes MAX=1.
Correct formula either adds `+20·log10(2) = +6.02 dB` or denormalizes to `[0,1]` first.

Verified numerically in `/tmp/psnr_audit2.py`:
| | current code | corrected | Δ |
|-|-|-|-|
| psnr_all | 20.24 dB | 26.26 dB | +6.02 |

**Impact**: consistent offset → does NOT change ranking within or across rounds.
Our published "X4 = 19.08 dB" is really 25.10 dB in [0,1] convention.

### Bug B — `psnr_obs` has a mask-area error
Numerator: `sum(mask·diff²)` (mask is {0,1}).  
Denominator (current): `B · C · H · W` (total pixels).  
Denominator (correct): `B · C · N_obs` (observed pixels only).

This **inflates** `psnr_obs` by `10·log10(N_total/N_obs)` — depends on mask.
For a 256×256 image with 15% unobserved, inflation ≈ **+0.72 dB**.
For larger boxes (30-40% unobserved), inflation grows to ~**2 dB**.

Verified in the same audit script.

**Impact**: within-round ranking preserved (same mask), cross-round PSNR_obs
numbers mildly distorted. `psnr_obs = ∞` correctly reports when observed
pixels exactly match (F1 case).

### Corrected PSNR table for the winner head-to-head (this mask: GT HF=0.663)

| config | PSNR_all ([0,1]) | PSNR_obs (fixed) | **PSNR_unobs** (fixed) | HF | res |
|--------|------------------|-------------------|------------------------|-----|-----|
| baseline            | 19.39 | 44.47 | 14.35 | 0.591 | 38.6 |
| W1_heps_global      | 16.09 | 44.45 | 11.05 | 0.637 | 38.8 |
| WINNER1_W1+F1       | 16.09 | ∞     | 11.05 | 0.637 | 0.0 |
| WINNER2_W1+X4       | 15.11 | 48.58 | 10.07 | 0.694 | 15.0 |
| **WINNER3_W1+X4+F1**| 15.11 | ∞     | 10.07 | **0.694** | **0.0** |
| W1 + h_eps=0.0001 + F1 | 15.09 | ∞ | 10.05 | 0.653 | 0.0 |
| X4 + F1             | 19.17 | ∞     | 14.13 | 0.613 | 0.0 |

`PSNR_unobs` is the right metric for inpainting — it isolates the box region.
Observe: **PSNR_unobs DROPS from 14.35 → 10.07 between baseline and WINNER3.**
Yet visually WINNER3 is dramatically better. This is the ill-posed-inpainting trap:
*MSE penalises a correct-but-misaligned bird more than it penalises a smooth fill.*

## 3. Image evidence (box-region ranking)

Files: `debug_IP4/results_winner_visual/winner_comparison.png` + individual PNGs.

Red rectangle = box. GT = full bird on branch (HF=0.663).

| Rank | Config | What's inside the box |
|------|--------|-----------------------|
| 1 | **WINNER3 = W1 + X4 + F1** | Full bird: head w/ eye+beak, round orange-brown body, clearly on branch. Visible white seam where generated bird boundary differs from GT (F1 hard-replace). |
| 2 | WINNER2 = W1 + X4         | Same bird reconstruction; observed pixels slightly off (no F1). |
| 3 | WINNER1 = W1 + F1         | Good bird, body slightly less textured than WINNER3. |
| 4 | W1 alone                  | Full bird body, slightly softer features. |
| 5 | W1 h_eps=0.0001 + F1       | Bird visible, extra granular texture — slightly over-sharpened. |
| 6 | **X4 + F1**                | **Smooth brown box — NO bird** (h_x alone does not trigger semantic content). |
| 7 | baseline                  | Smooth brown box, no bird. |

**The contrast is decisive**: X4 alone (h_x boost) does NOT produce a bird, but
X4 *combined with* W1 (global low h_eps) produces a crisper bird than W1 alone.
This is empirical confirmation that **eps entropy is the necessary condition**
and h_x is a multiplier on the prior-score contribution *once entropy is
available*.

Also confirmed from the earlier IP4 visualizer on a different mask:
- **Stage-3-only h_eps (H1) = baseline** (smooth brown)
- **Stage-3-only skip_eps (K2) = baseline** (smooth brown)
- Global h_eps (W1) and global skip_eps (K1) both produce bird structure

⇒ *Low h_eps must apply from the earliest stage*; by the time stage 3 starts,
stages 0-2 with default h_eps have already driven eps close to its damped
fixed point, leaving stage 3 no entropy to burn into texture.

## 4. Best method recommendation

### **WINNER3 = global h_epsilon=0.001 + stage-3 h_x=0.7 + terminal_replace=1.0**

```json
{
  "h_epsilon": 0.001,                       // preserve eps entropy globally
  "h_x": [0.1, 0.1, 0.1, 0.7],              // amplify prior-score lever at stage 3
  "terminal_replace_weight": 1.0,           // snap observed pixels to y (free)
  "lambda_reg": 50.0,
  "lambda_prox": 0.0,
  "num_langevin": 10
}
```

Prioritize this profile when the goal is **semantic box reconstruction**.
PSNR_all drops ~4 dB vs baseline, but the box actually contains a bird.

### Alternatives

- **If PSNR is not negotiable**: X4 + F1. No bird, but box is a smooth fill
  that matches GT's mean color and preserves observed pixels exactly.
- **If boundary seams in WINNER3 are bothersome**: drop F1, use WINNER2 alone.
  Observed pixels then have a small residual (~15 instead of 0) but the
  generated bird blends naturally into its surroundings without white edges.

## 5. Why this works in PRINCIPLE

Starting from the flow interpolant:
```
  x_τ^k = H_τ^k · x_1^k + σ_τ · ε^k
```

The flow model `μ_θ(x_τ, τ)` was trained on (x1, x0) pairs with `x0 ~ N(0, I)`
— i.e. **ε^k fed to the model at inference must look like an independent
Gaussian** for the velocity head to produce in-distribution predictions.

The PRINCIPLE inner loop does **two simultaneous jobs on ε**:
1. **Tweedie correction** — drives ε toward `σ·s_flow`, the MAP-consistent
   posterior value for ε given current x1. This is a variance-reduction mode.
2. **Training-distribution anchoring** — ε must remain Gaussian-like to keep
   x_τ in-distribution for the flow head.

These are in **direct tension**. `h_eps=0.01` × 400 iters collapses job 1 almost
completely → job 2 is destroyed → subsequent velocity predictions are OOD →
the model conservatively fills unobserved regions with a low-variance
conditional mean = smooth grey/brown. This is exactly the IP3 "smooth box" failure.

`h_eps=0.001` damps job 1 so job 2 is preserved. The Langevin now acts
**mostly on x_1** (via preconditioned CG update), while ε stays ≈ Gaussian.
x_τ remains in-distribution → the flow model freely generates texture.
x_1 still receives full posterior refinement because its update is gated by
`h_x` not `h_eps`.

In short: **the correct tempering is on ε, not x_1**. Our extra lever
(`lambda_prox`) found in Round-4 plays the same role as a counter-force against
Langevin over-refinement, but operates on x_1. These two mechanisms are not
redundant but most of the perceptual gain comes from the ε side.

## 6. Concrete code changes (already made to debug_IP4 files)

File: `debug_IP4/ms_sampler_v5.py` — no changes required to enable the winner;
existing knobs already cover `h_epsilon` scalar, per-stage list, plus the new
`skip_eps_update` / `reset_eps_per_ode_step` axes.

File: `debug_IP4/ms_posterior_sampling_IP4.json` — change to ship the WINNER3 profile:

```json
{
  "h_x":        [0.1, 0.1, 0.1, 0.7],
  "h_epsilon":  0.001,
  "lambda_reg": 50.0,
  "lambda_prox": 0.0,
  "terminal_replace_weight": 1.0,
  "num_langevin": 10
}
```

File: `ms_posterior_sampling_article_version_final.py` — adopt WINNER3 requires
also plumbing `h_epsilon` as a scalar (already done) and making sure warm_restart
is enabled (already True by default).

### PSNR bugs — fix in place

In `ms_sampler_v5.py` L224-225, replace with:

```python
xf01 = (xf + 1) / 2
gt01 = (gt + 1) / 2
psnr_all = (-10 * torch.log10(((xf01 - gt01) ** 2).mean())).item()

# observed pixels only: divide by N_obs, not N_total
m = mask_full
diff2 = ((xf01 - gt01) ** 2) * m
n_obs = m.sum().item() * xf.shape[1]  # pixels × channels
psnr_obs = (-10 * torch.log10(diff2.sum() / (xf.shape[0] * n_obs + 1e-12))).item()

# inpainting-relevant: PSNR on the unobserved region
m_unobs = 1 - m
diff2u = ((xf01 - gt01) ** 2) * m_unobs
n_unobs = m_unobs.sum().item() * xf.shape[1]
psnr_unobs = (-10 * torch.log10(diff2u.sum() / (xf.shape[0] * n_unobs + 1e-12))).item()
```

## 7a. Addendum — Round-5 closed the HF-metric gap

Round-5 (no terminal_replace, pure sampling axes) produced a Pareto table
where **M3 (lambda_reg stage-3 = 15)** tied for #1 HF proximity to GT at
|Δ|=0.001 with PSNR=13.13. W1 (h_eps=0.001 global) scored |Δ|=0.007 but
PSNR=10.54. **By metrics alone, M3 dominates W1.**

Running a focused side-by-side visualizer (`visualize_M3_vs_W1.py`) revealed:

- **M3 is visually identical to baseline** — smooth brown fill in the box,
  NO bird structure. Yet HF=0.625 matches GT 0.627.
- **W1 produces a reconstructed bird** with feathers, body, head.
  HF=0.620 is further from GT numerically, but visually correct.

**Why HF metric fails here**: `hf_energy(x * (1 − mask))` creates a hard
discontinuity at the mask boundary (value jumps from 0 inside mask to
finite outside). This jump dominates the FFT magnitudes regardless of
what is inside the box. Any config that slightly shifts the observed
region's intensity profile will change the boundary-jump amplitude and
therefore the HF sum — *without* affecting what's in the box.

⇒ **The HF metric cannot distinguish "smooth-filled box with crisp boundary"
from "bird-filled box with natural boundary"**. It is a **mask-boundary
energy measure**, not a semantic-texture measure.

**Revised trust ranking of our metrics**:
1. **Visual inspection (box interior)** — trustworthy
2. **PSNR_unobs** — trustworthy but biases toward smooth fill (MSE penalises
   misaligned texture more than low-variance mean fill)
3. **HF energy (current formula)** — **not trustworthy** for semantic content
   judgement; contaminated by mask-boundary artefacts
4. **PSNR_all** — partial mix; diluted by PSNR_obs

**A better perceptual metric would be**: FFT on a CENTRE crop that avoids
the mask boundary, or LPIPS between the box region and a neighbourhood patch,
or a learned no-reference quality score on the box interior.

### Correct final visual ranking (confirmed by side-by-side imagery)

| Rank | Config | Box content |
|------|--------|-------------|
| 🥇 1 | **WINNER3 = h_eps=0.001 + h_x stg3=0.7 + F1** | Sharp bird with head, body, orange chest, clear boundary (white seam from F1) |
| 🥈 2 | **WINNER1 = h_eps=0.001 + F1** | Clear bird, softer features, less seam visible |
| 🥉 3 | **M3 + W1 + F1** (lam_stg3=15 + h_eps=0.001 + F1) | Bird reconstructed; M3 adds mild edge sharpening |
| ❌ | M3 alone, BT*, Z*, all pure-lambda_reg configs | Visually = baseline. Metrics pretend otherwise. |

**M3 with W1 is fine because W1 dominates** — the W1 mechanism (eps entropy
preservation) is what creates content; lambda_reg reduction without W1 is a
metric-gaming null result.

## 7b. Three most valuable follow-ups

1. **Boundary blending**. WINNER3 shows clear white seams where the generated
   bird's silhouette disagrees with the true bird at observed pixels. The
   fix: replace hard `terminal_replace_weight=1.0` with a feathered alpha
   blend in a 4-8 pixel strip around the mask boundary, OR apply F1 only
   on pixels where `|xf - y| > sigma_n` (uncertainty-gated replace).

2. **Per-stage h_epsilon schedule with annealing from `0.01` down to `0.001`
   over stages**. The observation "stage-3-only doesn't work" proves the early
   stages matter, but maybe stage 0's full default `h_eps=0.01` helps build
   a coherent x_1 first. Try `[0.01, 0.005, 0.002, 0.001]`.

3. **Multi-sample posterior**. Run the same config with different seeds and
   blend / take median over samples. WINNER3's bird is plausible but not
   unique — a posterior average would recover a less-sharp but more
   GT-aligned result, recouping some of the lost PSNR_unobs.

Configs to **stop testing** (clearly dead from R1-R5 evidence):
- CFG (dead for this c2img model)
- DPS pre-kick in a Langevin pipeline (absorbed)
- Stage-3-only h_eps / skip_eps (no effect under warm_restart)
- `lambda_prox` alone (redundant with warm_restart)
- Aggressive `h_x_obs_ratio`
