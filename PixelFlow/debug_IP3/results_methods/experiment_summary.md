# debug_IP3 实验总结

## 背景
在 frozen PixelFlow prior 上做 multi-scale inverse problem (inpainting) 的 posterior sampling。
模型本身在 ImageNet 上取得 SOTA FID/SSIM，问题在于 posterior sampling 算法没有充分利用模型的高频生成能力。

## 实验体系

### v2 ablation (debug_IP3/run_ablation.py)
测试 soft sigma damping, tau-dependent rho, score clipping 的独立和组合效果。
- **结论**: soft sigma damping (σ_ref²=0.01) 是唯一有效改进 (+0.76 dB)，已合入 final_utils

### v3 ablation (debug_IP3/run_v3_ablation.py)  
测试三类结构性改动：null-space boost, reduced λ_reg, Tikhonov one-shot
- **null-space boost**: 全部失败，因为 A_k/AT_k 在低分辨率 stages 涉及 non-orthogonal resize，null-space projection 泄漏到 observed pixels
- **reduced λ_reg**: λ=10/20/30 全部比 λ=50 差，CG preconditioner 的 balance 已经 near-optimal
- **Tikhonov one-shot**: PSNR 低 (8-9 dB) 但 box region 生成了 reasonable texture（关键发现）
- **结论**: baseline L10_CG50 在 PSNR 维度最优；Tikhonov 揭示了模型 CAN 生成高频但 Langevin 磨掉了

### v4 ablation (debug_IP3/run_v4_ablation.py)
测试 frequency-aware 改进：x1_hat HF boost, score HF boost, annealed noise
- **x1_hat sharpening**: boost≥1.0 diverge（cascade errors across stages）; boost=0.5 只 +HF 不 +PSNR
- **score HF boost**: 完全无效，CG preconditioner 等价压制所有频率
- **annealed noise**: noise_init=0.3 annealing to 0 给出最佳 +0.08 dB
- **结论**: inference-time frequency 改动空间极小；smooth 是 CG+Langevin 的 structural property

### CG/Langevin sweep (debug_IP3/run_cg_sweep.py)
系统测试 CG iterations (1-50) × Langevin steps (1-10) 的完整网格
- CG=20 vs CG=50: 只差 0.03 dB → CG=20 足够（2x speedup）
- Langevin 步数是主导因子：L10→L5 丢 0.6 dB, L10→L1 丢 3.3 dB
- 但 L1-L3 在 box region 生成了 reasonable content（和 Tikhonov 一致）

### Methods sweep (debug_IP3/run_methods_sweep.py)
测试 7 种根本不同的 posterior sampling 架构

## 完整结果 (Box Inpainting, GT HF_unobs=0.632)

### Methods Sweep 排名

| Rank | Method | Config | PSNR_all | HF_unobs | Res | Time | 说明 |
|------|--------|--------|----------|----------|-----|------|------|
| 1 | PRINCIPLE | baseline_L10 | 13.25 | 0.584 | 37 | 155s | PSNR最优, box=smooth |
| 2 | DPS+Langevin | dps+L3_z5 | 12.41 | 0.543 | 1 | 48s | 混合最优, 3x快 |
| 3 | DPS+Langevin | dps+L3_z1 | 12.11 | 0.534 | 189 | 48s | |
| 4 | PRINCIPLE | baseline_L3 | 12.03 | 0.531 | 300 | 54s | |
| 5 | Alternating | alt_l100 | 9.15 | 0.585 | 47 | 15s | |
| 6 | Alternating | alt_l200 | 9.14 | 0.582 | 183 | 14s | |
| 7 | PGDM | pgdm_l200 | 9.06 | 0.606 | 24 | 16s | 有texture |
| 8 | PGDM | pgdm_l100 | 9.01 | 0.610 | 5 | 16s | 有texture |
| 9 | Resample | resamp_w0.7 | 8.96 | 0.584 | 0 | 12s | |
| 10 | PGDM | pgdm_l50 | 8.95 | 0.614 | 1 | 17s | HF接近GT |
| 11 | Resample | resamp_w0.5 | 8.91 | 0.590 | 0 | 13s | |
| 12 | Rep+DPS | rep+dps_z5 | 8.90 | 0.602 | 0 | 12s | |
| 13 | Resample | resamp_w0.3 | 8.87 | 0.595 | 0 | 13s | |
| 14 | Rep+DPS | rep+dps_z1 | 8.80 | 0.602 | 0 | 12s | |
| 15 | Replacement | replace | 8.77 | 0.603 | 0 | 12s | |
| 16 | PGDM 20s | pgdm_l100_20s | 8.63 | 0.620 | 2 | 31s | |
| 17 | DPS | dps_z5.0 | 8.51 | 0.614 | 5714 | 12s | |
| 18 | Replace 20s | replace_20s | 8.45 | 0.612 | 0 | 22s | |
| 19 | DPS | dps_z10 | 8.30 | 0.634 | 169 | 12s | HF=GT完美匹配 |

### 方法描述

1. **PRINCIPLE (baseline)**: CG-preconditioned Langevin inner loop, warm restart at each ODE step
2. **DPS**: x1 += ζ·A^T(y-Ax1)/||y-Ax1||, no inner loop, pure gradient correction
3. **Replacement**: observed pixels = measurement, unobserved = model prediction
4. **PGDM (Tikhonov)**: (A^TA/η² + λI)x1 = A^Ty/η² + λ·x1_hat, one CG solve per ODE step
5. **Resample**: annealed mixing of model prediction and measurement-consistent version
6. **Alternating**: odd ODE steps = unconditional, even = Tikhonov data consistency
7. **DPS+Langevin**: DPS gradient correction, then few Langevin steps for refinement

## 核心发现

### 1. PSNR vs Perceptual Quality 是 fundamental tradeoff
- Langevin 内循环优化 MSE → smooth（PSNR高）
- 模型 single-shot prediction → 有 texture 但 pixel 不 align（PSNR低）
- 在 PSNR metric 下，smooth 预测永远赢，因为 MSE penalizes misaligned texture > no texture

### 2. 模型能生成高频（DPS z=10 证明 HF=0.634 ≈ GT 0.632）
- 问题不在模型，在算法
- CG preconditioner 对所有频率等价压制 → 10 步 Langevin 把模型的高频预测磨成 smooth

### 3. Box region smooth 的三个原因
- CG preconditioner 在 unobserved 上限制 prior step to 0.001×g_prior per step
- 10 步 Langevin 累计只移动 ~0.04 per pixel（vs texture amplitude ~0.2）
- velocity model 的 re-evaluation 倾向于 regression to smooth mean

### 4. DPS+Langevin hybrid 是最佳平衡
- DPS z=5 gradient correction 解决 data consistency（res=1）
- 3 步 Langevin 提供最小必要的 score-based refinement
- 12.41 dB / 0.543 HF / 48s — 接近 baseline PSNR 的 94%，3x 更快

## 文件清单

```
debug_IP3/
├── langevin_v2.py          # v2: soft damping, tau-rho, score clip
├── langevin_v3.py          # v3: null-boost, Tikhonov
├── langevin_v4.py          # v4: frequency-aware (unsharp, noise anneal)
├── run_ablation.py         # v2 ablation
├── run_v3_ablation.py      # v3 ablation (structural changes)
├── run_v4_ablation.py      # v4 ablation (frequency-aware)
├── run_cg_sweep.py         # CG iterations × Langevin steps grid
├── run_methods_sweep.py    # 7 architecturally different methods
├── diagnose_smooth.py      # Per-step score/gradient diagnostics
├── results/                # v2 ablation images
├── results_v3/             # v3 ablation images + summary
├── results_v4/             # v4 ablation (partial, killed for GPU)
├── results_cg/             # CG sweep images
└── results_methods/        # Methods sweep images (this file)
```

## 推荐最终配置

### 如果优化 PSNR:
- baseline_L10_CG20: PSNR≈13.2, 速度 130s (vs 274s with CG50)
- 已合入的改动: soft sigma damping (σ_ref²=0.01), noise_scale=0, warm_restart=True

### 如果优化 perceptual quality:
- dps+L3_z5: PSNR=12.41, 速度 48s, 有 reasonable content in box region
- 或 PGDM λ=100-200: PSNR=9.0, 最好的 texture, 速度 16s

### 如果需要 DAPS-comparable quality (PSNR + FID 同时好):
- 需要更根本的改动：autograd-based DPS gradient (而非 CG)，或者 DAPS 的 annealed likelihood
- 当前框架的 limitation: CG preconditioner 结构性地限制了 prior 在 unobserved 上的表达能力
- 建议方向: 去掉 CG inner loop，改用 DPS-style gradient + replacement/projection 的组合，增加 ODE steps 数量来补偿
