# Validation Summary — 5 groups × 3 configs (N=100 each, total 1500 inferences)

## Setup
- **Val source:** /data/Zach_dataset/imageNet256/ILSVRC/Data/CLS-LOC/val (50000 images)
- **Sampling:** 500 disjoint indices via `np.random.RandomState(seed=20260428).choice`, split into 5 disjoint groups of 100
- **Class labels:** per-image, from LOC_val_solution.csv (synset) + LOC_synset_mapping.txt (line→0-based idx)
- **Mask:** 128×128 box at random position, **per-image** (operator.base_mask forced to [B,1,H,W]); mask seed `1000+g` for group g; identical 100 masks across all 3 configs
- **Measurement:** y = mask * gt (noiseless, matches existing sweep convention; sigma_n=0.05 only enters the algorithm prox)
- **Configs (all srs=1e-4, fd=1, tr=1):**
  - `pareto_dual_king_srs1e-4__fd1_tr1` (kw from `complete_configs/pareto_dual_king_srs1e-4.json`)
  - `LPIPS_king_srs1e-4__fd1_tr1` (kw from `complete_configs/LPIPS_king_srs1e-4.json`)
  - `balanced_perceptual_srs1e-4__fd1_tr1` (kw from `complete_configs/balanced_perceptual_srs1e-4.json`)

## Per-config summary (mean ± std across 5 groups, N=100 each)

| config | PSNR | SSIM | LPIPS | FID per-group (N=100) | FID pooled (N=500) |
|---|---|---|---|---|---|
| `pareto_dual_king_srs1e-4__fd1_tr1` | 19.998 ± 0.329 | 0.7929 ± 0.0010 | 0.1603 ± 0.0019 | 100.67 ± 6.71 | 66.15 |
| `LPIPS_king_srs1e-4__fd1_tr1` | 17.684 ± 0.230 | 0.7748 ± 0.0016 | 0.1556 ± 0.0017 | 81.16 ± 2.86 | 52.37 |
| `balanced_perceptual_srs1e-4__fd1_tr1` | 19.945 ± 0.324 | 0.7946 ± 0.0017 | 0.1627 ± 0.0026 | 103.24 ± 3.24 | 68.03 |

## All 15 runs

| group | config | PSNR | SSIM | LPIPS | FID(N=100) |
|---|---|---|---|---|---|
| 0 | `pareto_dual_king_srs1e-4__fd1_tr1` | 20.036 | 0.7934 | 0.1588 | 100.40 |
| 0 | `LPIPS_king_srs1e-4__fd1_tr1` | 17.735 | 0.7736 | 0.1564 | 85.40 |
| 0 | `balanced_perceptual_srs1e-4__fd1_tr1` | 20.260 | 0.7957 | 0.1602 | 103.36 |
| 1 | `pareto_dual_king_srs1e-4__fd1_tr1` | 19.566 | 0.7944 | 0.1590 | 91.80 |
| 1 | `LPIPS_king_srs1e-4__fd1_tr1` | 17.371 | 0.7772 | 0.1573 | 81.14 |
| 1 | `balanced_perceptual_srs1e-4__fd1_tr1` | 19.575 | 0.7972 | 0.1610 | 100.45 |
| 2 | `pareto_dual_king_srs1e-4__fd1_tr1` | 20.080 | 0.7929 | 0.1585 | 100.77 |
| 2 | `LPIPS_king_srs1e-4__fd1_tr1` | 17.814 | 0.7763 | 0.1528 | 81.08 |
| 2 | `balanced_perceptual_srs1e-4__fd1_tr1` | 20.045 | 0.7946 | 0.1607 | 100.81 |
| 3 | `pareto_dual_king_srs1e-4__fd1_tr1` | 20.541 | 0.7923 | 0.1625 | 112.46 |
| 3 | `LPIPS_king_srs1e-4__fd1_tr1` | 18.013 | 0.7735 | 0.1548 | 76.41 |
| 3 | `balanced_perceptual_srs1e-4__fd1_tr1` | 20.296 | 0.7936 | 0.1650 | 109.39 |
| 4 | `pareto_dual_king_srs1e-4__fd1_tr1` | 19.769 | 0.7916 | 0.1629 | 97.93 |
| 4 | `LPIPS_king_srs1e-4__fd1_tr1` | 17.486 | 0.7736 | 0.1569 | 81.79 |
| 4 | `balanced_perceptual_srs1e-4__fd1_tr1` | 19.548 | 0.7922 | 0.1667 | 102.20 |

## Best overall config

**`pareto_dual_king_srs1e-4__fd1_tr1`** (rank-sum across PSNR↑, SSIM↑, LPIPS↓, pooled-FID↓; lower is better).

Rank scores:
- `pareto_dual_king_srs1e-4__fd1_tr1`: 3
- `LPIPS_king_srs1e-4__fd1_tr1`: 4
- `balanced_perceptual_srs1e-4__fd1_tr1`: 5

See also `metric_logic_audit.md` for noteworthy considerations (FID with N=100 noise, etc.).
