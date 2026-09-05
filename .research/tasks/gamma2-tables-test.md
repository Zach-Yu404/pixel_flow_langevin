# gamma2-tables-test（2026-09-05，running）

## 用户原始要求（逐字）
「测试用spectral_labelled的setting下，测试这个gamma2_all和gamma2_labelled对比baseline看看好坏」

## 解读
S 用默认 spectral_class（= spectral_labelled）；三臂 γ² 表：baseline = gamma2_meas_alg4.json（7 图），
gamma2_all = val 50k 总表，gamma2_labelled = 按图片自身 synset 的类表。网格：5 task × 7 图（6 张网格图 + junco）
× seeds 42–44 = 315 格，schedule 强制 [2,2,1,1]（与 all_img_tests / S4 同口径），全噪声单样本；
指标 MSE(full/hole)/PSNR/SSIM/LPIPS(piq-VGG + alex)，rerun_imageNet/metrics.py。

## 实现
gamma2_stats/test/run_gamma2_test.py（原子 claim 工作窃取，results.csv 追加，幂等）；wrapper run_gpu.sh <gpu>
（GPU 0–3）；日志 g2test_gpu*.log。聚合：per (arm, task) 对 7 图 × 3 seed 的 mean±std，
以及逐格配对差（同 task/image/seed 下 arm − baseline）。

## 审查（workflow wf_18befe03）
key 逐点匹配、按类查表、数值范围均通过。采纳：修 CSV 表头竞争 + done_keys 容错、按类 class_idx 断言、
加 baseline_rerun 臂（同表同 seed 重跑 = GPU 运行噪声地板）；共 420 格。注意：baseline 表对这 7 张图 100% 在样本内，
gamma2_labelled 2%（1/50），gamma2_all 0.002% —— 对比时需说明。

## 追加：估计量视角（用户 2026-09-05）
「尝试预测full noise, no xi_0, no noise, full noise MMSE5, full noise MMSE10」→ run_gamma2_est.py：每 (arm, task, image)
10 seeds 全噪声 → single_avg10 / MMSE5 / MMSE10；no_xi0（ξ₀=0 全程，seed 42）；no_noise（ξ_y=ξ_h=ξ_s=0 全程，seed 42）。
105 格 × 12 次采样 ≈ 1.5 h（4 GPU）。输出 results_est.csv → summary_est.md / per_task_est.csv。

## 结果 A：3-seed 单样本（gamma2_stats/test/summary.md, per_task.csv, psnr_per_task.png；420/420 格）
逐格配对 Δ vs baseline（7 图 × 3 seed = 21 格/task，105 格总）：
| 指标 | gamma2_all Δ (wins) | gamma2_labelled Δ (wins) | baseline_rerun Δ |
|---|---|---|---|
| PSNR | −0.03±0.01 dB (0/105) | −0.07±0.13 dB (41/105) | +0.00±0.00 |
| SSIM | −0.0009±0.0005 (2/105) | −0.0025±0.0064 (45/105) | 0 |
| LPIPS-alex | +0.0011±0.0005 (2/105) | +0.0032±0.0065 (44/105) | 0 |
| hole MSE (inpainting) | +0.0007 (0/42) | +0.0013 (13/42) | 0 |
- 五个 task 方向一致：gamma2_all 系统性略差（−0.03~−0.04 dB，105 格无一胜），gamma2_labelled 均值更差但方差大
  （逐图有胜有负）。量级极小：PSNR 差 <0.1 dB，是 S 臂间差异（3 dB）的 1/100。
- baseline_rerun 与 baseline 逐格一致到 4 位小数：GPU 运行噪声在指标层面 ≈0，故 −0.03 dB 虽小但是真实系统性差异。
- 解读：baseline 7 图表对这 7 张测试图 100% 在样本内（γ² 偏小 25–55%），val 表把 (19) 端点解中 γ²H² 项加重、
  对速度的信任略降；在全噪声单样本下这带来 ~0.03 dB 的损失。γ² 表在这个量级上对结果几乎不敏感。

## 状态
running：估计量测试（105 格，single/MMSE5/MMSE10/no_xi0/no_noise）4 GPU 运行中。
