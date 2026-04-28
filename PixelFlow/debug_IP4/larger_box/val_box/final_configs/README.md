# final_configs — 9 tr=0 configs (3 srs × 3 levels) + 1 tr=1 ultimate

All configs share `lambda_reg=150, lambda_prox=150, noise_scale=1.0,
guidance_scale=2.0, h_x=0.1` and run on 128×128 box, 2-image baseline_05 GT,
seed=20000120, mask seed=7919.

## The 3×3 tr=0 grid

Picked by Pareto analysis of 12 sweep2 tr=0 configs (srs=1e-2 default) plus
24 folder_woNoiseLast configs (srs=1e-3, 1e-4).

|        | LPIPS_king | balanced_perceptual | pareto_dual_king |
|--------|------------|---------------------|-------------------|
| **srs=1e-2** | (h_eps=0.001, L=5) PSNR=17.70 LPIPS=0.2392 | (h_eps=0.001, L=10) PSNR=16.98 LPIPS=0.2418 | (h_eps=0.01, L=5) PSNR=19.18 LPIPS=0.2552 |
| **srs=1e-3** | (h_eps=0.001, L=10) PSNR=16.14 LPIPS=0.2310 | (h_eps=0.01, L=10) PSNR=18.82 LPIPS=**0.2366** | (h_eps=0.01, L=15) PSNR=**19.89** LPIPS=0.2622 |
| **srs=1e-4** | (h_eps=0.001, L=10) PSNR=16.13 LPIPS=**0.2268** | (h_eps=0.01, L=10) PSNR=18.83 LPIPS=0.2348 | (h_eps=0.01, L=15) PSNR=**19.93** LPIPS=**0.2593** |

### Key cross-srs trends visible in the grid

- **srs↓ 在 h_eps=0.01 cells 显著改善 PSNR**：双冠 line 18.40 → 19.89 → 19.93。
- **srs↓ 在 h_eps=0.001 cells 微调改善 LPIPS**：LPIPS_king line 0.2392 → 0.2310 → 0.2268，但 PSNR 略跌。
- **`balanced_perceptual` triplet (0.01, 0.1, L=10) + srs≤1e-3 给出 \|dHF\|=0.004**：全 sweep 最干净的高频匹配。
- **srs=1e-3 与 1e-4 几乎饱和**：两列的差异在所有 cell 都 < 0.05 PSNR、< 0.005 LPIPS。

## How to pick (tr=0 limit)

1. **Pure perceptual**：`LPIPS_king_srs1e-4`（LPIPS 0.2268 但 PSNR 16.13 偏低）。
2. **Balanced 默认推荐**：`balanced_perceptual_srs1e-4`（PSNR=18.83, LPIPS=0.2348，且 \|dHF\|=0.004 最干净）。
3. **Max PSNR**：`pareto_dual_king_srs1e-4`（PSNR=19.93, LPIPS=0.2593）。

`srs=1e-3` 是 `srs=1e-4` 的可替代品，差异在噪声内。`srs=1e-2`（默认）只在 baseline 对照时使用。

## Bonus: `ultimate.json` (tr=1)

如果允许打开 `terminal_replace_weight=1.0`，最强配置是
`balanced_perceptual` triplet + tr=1：

| | PSNR | LPIPS | SSIM | \|dHF\| |
|--|------|-------|------|---------|
| `ultimate` (tr=1, srs=1e-4) | **19.30** | **0.1480** | **0.806** | 0.063 |

全维度反超 sweep2 的之前 PSNR/LPIPS 双冠 (19.28 / 0.1566)。validation 已确认。

## `final_denoise` 选项

`run_ip4` 现在接受 `final_denoise=True/False`（在 sampling 末加一次
WLS-CG 收尾）。**4-config validation (2×2 fd × tr) 显示 fd 单独是
no-op (Δ < 0.01 全维度)**，因 sampling 末段 sigma_τ 已极小，x1_k
已经在每个 Langevin step 内被 CG 投影到 WLS 解附近。API 保留供后续算法改动使用。

## Usage

```python
import json
cfg = json.load(open("final_configs/balanced_perceptual_srs1e-4.json"))
xf, *_ = run_ip4(model, config, gt, y, operator, sigma_n, device,
                 class_label=10, seed=20000120, **cfg["kw"])
```

## File layout

```
final_configs/
├── README.md                              (this file)
├── LPIPS_king_srs{1e-2,1e-3,1e-4}.json    (3 files)
├── balanced_perceptual_srs{...}.json      (3 files)
├── pareto_dual_king_srs{...}.json         (3 files)
├── ultimate.json                          (tr=1 winner)
└── validation/                            (final_denoise × tr 4-cell验证)
```
