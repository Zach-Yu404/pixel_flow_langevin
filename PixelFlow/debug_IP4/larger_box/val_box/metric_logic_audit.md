# Metric & Pipeline Logic Audit

This file documents how each metric is computed, normalization conventions,
GT/mask alignment, label sourcing, and known caveats — written **before**
running the 15-experiment validation so issues are surfaced upfront.

---

## 1. Image range & normalization

| Stage | Range | Notes |
|---|---|---|
| GT loaded from disk | `[0, 1]` (PIL → ToTensor) → normalized to `[-1, 1]` (`Normalize([0.5]*3, [0.5]*3)`) | Matches the c2img model's training input range (model expects `[-1, 1]`). |
| Inference output `xf` | `[-1, 1]` | Returned by `run_ip4`. |
| For PSNR / SSIM / LPIPS / FID | converted via `(t + 1) / 2` then `clamp(0, 1)` | Standard piq input convention. |
| For Inception (FID) | `[0, 1]`, `normalize_input=True` (piq normalizes internally with ImageNet stats and resizes to 299×299) | piq handles resizing + ImageNet norm. |

GT and recon use the **same** transform pipeline (CCA crop → 256, [-1,1]). No
downstream resize before metrics.

---

## 2. Per-image label sourcing

```
LOC_val_solution.csv  -> (image_id) -> synset string  (e.g. n01751748)
LOC_synset_mapping.txt[line N, 1-based] -> synset on line N gets class index N-1
```

Spot-check: `n01751748` is on line 66 of `LOC_synset_mapping.txt` → class index **65**, which matches the previously-observed "sea snake" assignment for `ILSVRC2012_val_00000001` in earlier runs. ✅

The `run_ip4` signature accepts `class_label` as `list[int]` of length `B`
(verified at `ms_sampler_v5.py:97-99`):

```
if isinstance(class_label, (list, tuple)) or torch.is_tensor(class_label):
    pe_labels = torch.as_tensor(class_label, dtype=torch.int32, device=device)
    assert pe_labels.shape[0] == B, ...
```

Per-image labels are passed at every batch invocation: **no fixed class=10
fallback used anywhere** in this validation pipeline.

---

## 3. Mask consistency across configs

The `Inpainting` operator caches its mask in `base_mask` on first call (see
`inpaintingStart.py:288-291`). `_get_mask_and_input` re-uses the cache. Once
populated it is never re-drawn.

We exploit this by **forcing** `operator.base_mask = pre_computed_masks` on
each fresh `Inpainting` instance, where `pre_computed_masks` is a `[B, 1, H, W]`
tensor (one mask per image). All downstream operator paths (`__call__`,
`adjoint`, `precondition`, `get_mask`) just multiply / scale the mask
element-wise, so per-image masks broadcast correctly with `[B, 3, H, W]`
images.

Mask seeds:
- group `g` ∈ {0..4} → mask seed `1000 + g`
- 100 distinct masks pre-generated in `prepare_groups.py` and saved to
  `groups/group_{g}.pt` under key `masks` (shape `[100, 1, 256, 256]`)
- Same `groups/group_{g}.pt` is loaded by all 3 configs in group `g`, so
  **the i-th image gets the same mask under every config**

Hole count per mask is asserted to equal exactly 128² = 16384 in
`prepare_groups.py`.

---

## 4. Measurement `y`

```python
y = operator(gt).detach()    # __call__ returns mask * x_use, NO additive noise
```

This matches the existing 36-config sweep convention. The `sigma_n = 0.05`
parameter is passed to `run_ip4` for use inside the prox / Tweedie damping,
but `y` itself is **noiseless**. (Synthetic-experiment convention; same
across all 3 configs so the comparison stays fair.)

---

## 5. PSNR

`piq.psnr(recon01, gt01, data_range=1.0)`. Computed **per image** (B=1
slice), then averaged across the 100 images. **No mask weighting** in the
"global" PSNR — full-image PSNR.

A second metric `psnr_unobs` is computed only over the unobserved region
(weighted by `1 - mask`), useful as a debugging signal but not a primary
metric in this validation run.

---

## 6. SSIM

`piq.ssim(recon01, gt01, data_range=1.0)` — default `kernel_size=11`,
`sigma=1.5`, RGB averaged. Per image, then averaged across 100.

---

## 7. LPIPS

`piq.LPIPS(replace_pooling=True)` — VGG-based, default backbone weights.
`replace_pooling=True` is the piq-recommended setting (max-pool → avg-pool)
and is what the 36-config sweep used. Per image, then averaged.

---

## 8. FID

Two reported FID numbers per (group, config):

1. **Per-group FID** (N=100): real = 100 GT features, fake = 100 recon
   features. Reported in `metrics_all_runs.csv`.
2. **Pooled FID** (N=500): real = 500 GT features (5 groups concatenated),
   fake = 500 recon features. Reported in `metrics_by_config.csv`.

Feature extractor: `piq.feature_extractors.InceptionV3(output_blocks=[3],
normalize_input=True, use_fid_inception=True)` → 2048-d features from
mixed_7c. Inputs in `[0, 1]` (auto-resized to 299×299 by piq).

### ⚠️ FID at N=100 is statistically very noisy
Standard FID benchmarks use N≥10000 (commonly 50K). At N=100, FID variance
can be larger than the signal between configs. Reported per-group FID is
included for completeness and for **relative** within-group ranking, but
the **pooled N=500 FID** is the more reliable per-config number — and even
that is below standard FID conventions.

The 5 per-group FIDs do let us estimate variance (`fid_std_pergroup` in
`metrics_by_config.csv`), which gives a sense of how much signal vs. noise
is present.

---

## 9. Configs used (locked)

All three configs are sourced from `final_configs/full_validation/complete_configs/`:

| validation tag | source JSON | h_eps | num_langevin | sigma_ref_sq | final_denoise | terminal_replace_weight |
|---|---|---|---|---|---|---|
| `LPIPS_king_srs1e-4__fd1_tr1`          | `LPIPS_king_srs1e-4.json`          | 0.001 | 10 | 1e-4 | True | 1.0 |
| `balanced_perceptual_srs1e-4__fd1_tr1` | `balanced_perceptual_srs1e-4.json` | 0.01  | 10 | 1e-4 | True | 1.0 |
| `pareto_dual_king_srs1e-4__fd1_tr1`    | `pareto_dual_king_srs1e-4.json`    | 0.01  | 15 | 1e-4 | True | 1.0 |

`final_denoise` and `terminal_replace_weight` are explicitly overridden in
`run_one.py:load_kw` to ensure the `__fd1_tr1` axis is used regardless of
what the source JSON has set (defensive). All other 29 kwargs come from the
complete-spec JSON.

---

## 10. Reproducibility notes

- **Image sampling seed:** `np.random.RandomState(seed=20260428)` over the
  sorted list of 50000 val filenames, picks 500 disjoint indices.
- **Mask seed:** `np.random.seed(1000+g) + torch.manual_seed(1000+g) + cuda
  seed` per group.
- **Per-batch run_ip4 seed:** `1_000_000 + group_id * 1000 + batch_start`.
  This means the same image (same `(group, batch_idx)`) gets identical
  noise/init across the 3 configs.

---

## 11. Things that did **not** look right and were rejected as likely-issues

None of the following were observed; listed for transparency:

- ❌ Fixed `class_label=10` for all 100 images (rejected: per-image labels
  used everywhere)
- ❌ One mask shared across 100 images (rejected: per-image masks via
  `operator.base_mask = [100, 1, H, W]` slicing per batch)
- ❌ Different masks across the 3 configs (rejected: same `group_{g}.pt`
  loaded by all 3)
- ❌ piq.LPIPS without `replace_pooling=True` (rejected: kept consistent
  with the 36-config sweep)
- ❌ `data_range=255` mismatch (rejected: all metrics use `data_range=1.0`
  with images in `[0, 1]`)

---

## 12. Known limitations / caveats (no fix attempted, documented only)

1. **N=100 FID noise** (above). Mitigated by reporting pooled-N=500 in
   addition.
2. **Noiseless y** (above). Acceptable: matches the sweep convention; same
   for all 3 configs.
3. **Inception weights** are the standard FID checkpoint
   (`pt_inception-2015-12-05-6726825d.pth`, mseitzer/pytorch-fid). This is
   the established FID benchmark; comparable to most published numbers.
4. **piq.SSIM** is RGB-channel-averaged, not luminance-Y SSIM. Some papers
   report Y-SSIM; ours is RGB. Internally consistent across configs.
