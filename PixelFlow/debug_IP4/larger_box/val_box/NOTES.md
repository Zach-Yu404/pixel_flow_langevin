# val_box — 5×100 ImageNet val validation of 3 box-inpainting configs (memory)

128×128 box inpainting on **per-image-class-conditioned** ImageNet val. 5 disjoint
groups × 100 images × 3 configs = **15 cells, 1500 inferences**. Built on top of
the 36-cell `final_configs/full_validation` sweep — this folder takes the 3
srs=1e-4 winners (one per "level": LPIPS_king / balanced_perceptual /
pareto_dual_king), pins `final_denoise=True / terminal_replace_weight=1.0`, and
re-validates on a real val sample with FID + class-faithful guidance.

All 1500 inferences finished. Results stable across groups (PSNR std ≤0.33).

## TL;DR — what to use for 128×128 box (N=100 val per group, 5 groups)

| Need | Pick | PSNR | LPIPS | FID(pooled N=500) |
|------|------|---:|---:|---:|
| Best PSNR / SSIM | `pareto_dual_king_srs1e-4__fd1_tr1` | **20.00** | 0.1603 | 66.2 |
| Best LPIPS / FID | `LPIPS_king_srs1e-4__fd1_tr1` | 17.68 | **0.1556** | **52.4** |
| (Pareto-dominated, do not pick) | `balanced_perceptual_srs1e-4__fd1_tr1` | 19.94 | 0.1627 | 68.0 |

Rank-sum across 4 metrics → pareto_dual_king wins overall (score=3 vs 4 vs 5).
But if perceptual quality (LPIPS/FID) is the priority, **LPIPS_king is decisive**:
its pooled FID is **14 lower** than the runner-up — a real signal, much larger
than the per-group FID std (~3-7) so it's not N=100 noise.

## Parameter impact (cumulative across all box experiments)

Order by leverage on the 4 metrics:

### 🟢 Top levers (always set these)

1. **`terminal_replace_weight`: 0 → 1** — biggest single lever (36-cell sweep,
   18 paired cells averaged): ΔPSNR=+0.13, ΔLPIPS=−0.074, ΔSSIM=+0.094.
   All three improve, zero cost. **Default = 1.0**.

2. **`h_epsilon`: 0.001 vs 0.01** — splits PSNR vs LPIPS/FID:
   - `h_eps=0.001` → LPIPS 0.156, FID 52 (perceptual king)
   - `h_eps=0.01`  → PSNR 20.0  (PSNR king)
   This is the route-defining axis when picking 1 of 3.

### 🟡 Mid levers

3. **`num_langevin` (L)**: 5/10/15. PSNR monotonically up (~+0.2-0.5 dB / +5).
   L=10 sweet spot for cost/quality. L=15 is +0.05 dB marginal at +50% wall time.

4. **`sigma_ref_sq`**: 1e-2 → 1e-3 → 1e-4 universally improves both metrics
   (small Δ). 1e-4 is the safe default. (1e-5 not yet tested at 128 box.)

### 🔴 Weak / can ignore

5. **`final_denoise`** — near no-op (mean ΔPSNR=+0.010, ΔLPIPS=−0.0003 across
   18 paired cells). Defensive `True` keeps it on, but `False` is free perf.

6. **`h_x` = 0.1** — sweet spot. Other values worse or no change in 36-cell
   sweep. Don't touch for box.

7. **`noise_scale = 1.0`** — kept on for sample diversity. `noise_scale=0`
   loses perceptual variability per earlier `noise_uni3` exploration.

## What was checked before launching (all GREEN)

| Item | Verification |
|------|--------------|
| Class label per image | `LOC_val_solution.csv` × `LOC_synset_mapping.txt` (1-based line → 0-based class). Spot-check: `n01751748` line 66 → class 65 (sea snake). |
| `run_ip4` accepts `class_label=list[B]` | `ms_sampler_v5.py:97-99` explicit assert. |
| Mask fairness across configs | `Inpainting._init_mask` caches `base_mask` once; we force `operator.base_mask = pre_built [B,1,256,256]`. Same `groups/group_{g}.pt` is loaded by all 3 configs in group g — i-th image gets identical mask under every config. Hole count assert = 128² = 16384 per mask. |
| Metric inputs | All metrics use `data_range=1.0` with images in `[0,1]` (`(x+1)/2` then clamp). FID via `piq InceptionV3(output_blocks=[3], normalize_input=True)` — auto-resize to 299×299. |
| Smoke test | N=8, B=8, LPIPS_king: PSNR=17.54, LPIPS=0.1645, t=779s ✅ before full launch. |

## Bug found and fixed during launch

`GROUPS=(0 1 2 3 4)` was silently rejected by bash — **`$GROUPS` is a built-in
readonly array** (user's Unix groups: 1001=nvidia, 986=docker). First launch
dispatched only 6 jobs with `--group 1001` and `--group 986`. Renamed to
`VAL_GROUPS` → 15 jobs dispatched correctly.

## File layout

```
val_box/
├── NOTES.md                      ← this file
├── prepare_groups.py             ← builds 5×100 disjoint groups + per-img masks
├── run_one.py                    ← one (group, config) runner; drives B=8 batches
├── aggregate.py                  ← FID + emits CSVs + validation_summary.md
├── launch_val.sh                 ← 15 jobs / 8 GPUs dispatcher
├── metric_logic_audit.md         ← detailed metric / normalization / mask audit
├── validation_summary.md         ← N=500 pooled summary + best overall
├── metrics_all_runs.csv          ← 15 rows
├── metrics_by_config.csv         ← 3 rows (mean ± std + pooled FID)
├── metrics_by_group.csv          ← 5 rows × 3 configs wide
├── configs/                      ← 9 complete configs (3 levels × 3 srs), all fd=True/tr=1.0
│   ├── LPIPS_king_srs1e-{2,3,4}__fd1_tr1.json
│   ├── balanced_perceptual_srs1e-{2,3,4}__fd1_tr1.json
│   └── pareto_dual_king_srs1e-{2,3,4}__fd1_tr1.json   ← 3 srs=1e-4 are validated, 6 srs=1e-2/1e-3 are spec only
├── groups/group_{0..4}.pt        ← {gts, masks, labels, fnames, mask_seed} per group
└── runs/group{g}_{cfg}/{data.pt, meta.json}   ← 15 result dirs
```

## Reproducibility seeds

- Image sampling: `np.random.RandomState(seed=20260428).choice(50000, 500)` then split
- Per-group mask seed: `1000 + g` (numpy + torch + cuda seeded)
- Per-batch run_ip4 seed: `1_000_000 + group*1000 + batch_start`
  (so the same image gets identical noise/init across the 3 configs)

## Caveats (documented, no fix)

- **FID at N=100 per group is noisy** by FID standards (≥10K usual). Pooled
  N=500 per config is the more reliable number, and even that is below
  convention. The relative gap (LPIPS_king 52.4 vs 66.2/68.0) is 4-5× the
  per-group std (~3-7), so the FID ranking is robust signal not noise.
- `y = mask * gt` is **noiseless** (matches the existing sweep convention);
  `sigma_n=0.05` only enters the algorithm prox. Same convention for all 3
  configs, fair within this study.
- `piq.SSIM` is RGB-channel-averaged, not luminance-Y SSIM.

## Pointers

- 36-cell sweep memory: see `final_configs/full_validation/runs/` and the
  parent `larger_box/NOTES.md` for the original h_eps × L sweep.
- 9 base configs (pre-fd/tr crossing): `final_configs/full_validation/complete_configs/`.
- All val_box runtime kw can be reproduced via `from run_one import load_kw, CONFIG_MAP`.

## What this *doesn't* tell you

- Random inpainting performance — see sibling `debug_IP4/random_inpainting/` (OAT
  in flight, separate task).
- Mask shapes other than 128×128 box — only this size validated here.
- N≥1000 FID — all reported FIDs are at most N=500 pooled.
