# PSNR audit — `larger_box` (and `exploration/`) sweeps

**Verdict:** PSNR / PSNR_unobs computations are **correct** for the 128×128 box at
resolution=256 setting. The numbers in `larger_box/results/*.json` and
`exploration/runs/*.json` can be trusted without correction.

---

## Where PSNR is computed

Two layers:

1. **`ms_sampler_v5.run_ip4`** (`debug_IP4/ms_sampler_v5.py:245-262`) — internal
   publication-formula PSNR on `[-1, 1]` data:
   ```python
   psnr_all = 10 * log10(4.0 / mse_all.clamp(min=1e-12))
   psnr_obs   = 10 * log10(4.0 / mse_obs)     # obs uses mask=1
   psnr_unobs = 10 * log10(4.0 / mse_unobs)   # unobs uses (1-mask)
   # mse_obs   = (mask_full * (xf-gt)^2).sum() / (mask_full.sum() * C)
   # mse_unobs = ((1-mask_full)*(xf-gt)^2).sum() / ((1-mask_full).sum() * C)
   ```
   Pixels in `[-1, 1]`, so `MAX = 2 → MAX² = 4`. Mask-aware denominators are
   correct: `mask.sum()` = number of mask-1 spatial positions, multiplied by
   `C = 3` to get total scalars. Numerator broadcasts correctly because mask
   has shape `[B, 1, H, W]` and `diff²` has `[B, 3, H, W]`.

2. **`larger_box/run_chunk.py`** and **`exploration/scripts/run_chunk_e.py`**
   (the JSON we actually rank against) — recompute via `piq` on `[0, 1]`:
   ```python
   recon_01 = ((xf + 1) / 2).clamp(0, 1)         # convert from [-1,1]
   gt_01    = ((gt + 1) / 2).clamp(0, 1)
   psnr_all = piq.psnr(r1, g1, data_range=1.0)   # = -10*log10(MSE_[0,1])
   # psnr_unobs:
   m = (1 - mask)
   diff2 = (r1 - g1).pow(2)
   n_unobs = m.sum() * r1.shape[1]               # spatial × C channels
   mse_u   = (m * diff2).sum() / n_unobs
   psnr_u  = -10 * log10(mse_u)
   ```
   Equivalent algebraically:
   `MSE_[0,1] = MSE_[-1,1] / 4`, so
   `-10·log10(MSE_[0,1]) = -10·log10(MSE_[-1,1]/4) = 10·log10(4/MSE_[-1,1])`.
   Same number as the publication formula.

## Sanity check (geometric)

For `terminal_replace_weight=1.0` (every config in the larger_box sweep), observed
pixels match exactly so MSE=0 there. With 128×128 box on a 256×256 image the box
fraction is `128² / 256² = 0.25`. So:
- `MSE_all = 0.25 · MSE_unobs`
- `PSNR_all − PSNR_unobs = −10·log10(0.25) = +6.02 dB`

Empirical: `C_L10` reports `PSNR=16.60`, `PSNR_unobs=10.57` → diff 6.03 dB. ✅
This identity holds across all 32 configs (within 0.02 dB), confirming the mask
denominators are using the right number of valid positions.

## Image range / clipping / dtype

- All sampler outputs and GTs are `[-1, 1]` `float32`. ✅
- `(xf+1)/2` then `.clamp(0,1)` is correct. ✅
- No accidental resize/interp before the metric — both tensors stay 256×256. ✅

## Mask convention

- `operator.get_mask(x)` returns `1` for **observed** pixels and `0` for
  **unobserved** (the box). ✅
- `mask` shape is `[1, 1, 256, 256]` then expanded to batch `B`. Single mask
  shared across the 2 GT samples (deterministic seed `7919`).
- `psnr_unobs` uses `(1 - mask)` — restricts to the box interior.
- `box_fraction = (1-mask).sum() / mask.numel() = 0.25` confirmed by inspection.

## Per-image vs batch averaging

`run_chunk.py` and `run_chunk_e.py` compute **per-image** PSNR / SSIM / LPIPS,
then average. Since both GT samples have the same mask shape and the same
denominator, simple-mean across the 2 images gives the right per-batch
mean (no weighted-mean bug). ✅

## What is NOT trustworthy

- **Statistical robustness**: 2 GT images is too small a sample to declare
  fine-grained winners. Treat differences `< 0.005 LPIPS` and `< 0.3 dB PSNR`
  as ties.
- **Single mask seed**: only `seed=7919` is used. A different random box
  position could shift winners by more than the metric noise.
- **HF / |dHF|**: this is a custom statistic from `ms_sampler_v5.hf_energy`
  (FFT-based, ratio of HF magnitude to total). Useful as a directional signal
  for "does the recon have GT-level texture" but **not** a perceptual metric.

## Recommendations going forward

1. The current PSNR pipeline is correct — **no code change needed**.
2. For final ranking decisions, weight visual judgment heavily and require
   `> 0.5 dB` PSNR delta or `> 0.005` LPIPS delta to break ties.
3. If you later run a larger eval (>=20 GTs, multiple mask seeds), keep the
   same `(mask*diff²).sum() / (mask.sum()*C)` pattern — it generalizes cleanly.
