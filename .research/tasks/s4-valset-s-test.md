# s4-valset-s-test（2026-09-02 → 09-03，done）

## 用户原始要求（逐字）
「在这个s_stats下边创建test，参考/CBIG-Standard-ECE/Zach/MSFlow/PixelFlowICLR/Algorithm2/results/all_img_tests
中对allnoise(baseline), MMSE5, MMSE10, no_noise这几个baseline，用4种s测S_it = 2和[2, 2, 1, 1], [2, 2, 2, 1]
的PSNR, SSIM和LPIPS的结果」；后续「GPU 0，1，2，3只要可以跑的都跑上」。

## 展开
- 4 种 S = compute-s-stats-valset 产物：pooled_all / pooled_class（按图片 synset 取该类
  per-stage 标量）/ spectral_all / spectral_class（按 synset 取该类 centered 功率谱 →
  utils.SpectralSOp）。schedule ∈ {2222(=S_it 2), 2211, 2221}。
- 网格同 all_img_tests：5 task × 6 图（非 junco 6 张）；估计量 single_avg10（seed42-51
  单样本指标平均）/ MMSE5 / MMSE10（像素平均）/ no_noise（ξ_y=ξ_h=ξ_s=0 全程，seed42）。
  指标 rerun_imageNet/metrics.py（PSNR/SSIM piq range1，LPIPS piq-VGG + 官方 alex）。
- 共 4×3×5×6=360 cell，每 cell 4 行 → 1440 行。

## 实现
s_stats/test/run_s4_test.py（单文件；in-process _task_setup EIO 重试；原子 per-cell claim
test/_claims/ O_EXCL，>20min 陈旧回收，四 GPU 工作窃取）。wrapper run_single.sh(GPU2)、
run_gpu.sh <g>(GPU0/1/3，flock+30s 重试 60 次)。输出 test/s4_results.csv，
聚合 scratchpad/s4_aggregate.py → test/s4_per_task{.csv,_numeric.csv,.md}
（含 all_img_tests 参考行 ref:spectral_14img / ref:pooled_junco，[2,2,1,1]）。

## 结果（360/360 格，1440 行，四 shard DONE；s_stats/test/s4_per_task.md 全表）
6 图均值（5 task 平均）PSNR dB / SSIM / LPIPS-alex，schedule 2222：

| S | single | MMSE5 | MMSE10 | no_noise |
|---|---|---|---|---|
| pooled_all | 14.50/0.197/0.892 | 19.53/0.321/0.622 | 21.06/0.383/0.537 | 25.15/0.692/0.345 |
| pooled_class | 14.78/0.201/0.874 | 19.79/0.328/0.605 | 21.31/0.391/0.523 | 25.16/0.692/0.345 |
| spectral_all | 21.14/0.384/0.495 | 24.11/0.558/0.391 | 24.72/0.611/0.377 | 25.42/0.705/0.340 |
| spectral_class | 21.23/0.383/0.491 | 24.25/0.559/0.382 | 24.86/0.612/0.368 | 25.46/0.706/0.338 |
| ref:spectral_14img(2211, 校准泄漏) | 21.67/0.408/0.482 | 24.33/0.578/0.395 | 24.84/0.625/0.387 | 25.26/0.698/0.351 |
| ref:pooled_junco(2211) | 18.78/0.271/0.627 | 22.76/0.440/0.446 | 23.77/0.513/0.402 | 24.87/0.681/0.353 |

发现：
1. **标量 val-set S（0.27–0.31，junco 表 5.5×）在噪声路径上灾难性**：单样本比 junco 表再降
   4 dB（13.8–14.8 vs 18.8），MMSE10 仍差 2.5 dB；但 no_noise 与 junco/spectral 持平
   （24.6–25.2）。⟹ S 的量级只经 RTO 噪声 S^{-1/2}ξ_s 与 C⁻¹ 权重的噪声放大起作用，对
   确定性中心几乎无影响（与 all-img-tests「3ξ=0 后两臂收敛」一致）。
2. **spectral_all ≈ 旧 14 图 spectral**：单样本 −0.5 dB（旧表含 6 张测试图，校准泄漏），
   MMSE10 −0.1 dB，no_noise +0.16 dB，LPIPS(MMSE10/no_noise) 更优。全 val 谱是干净且不劣的替代。
3. **spectral_class 全面小胜 spectral_all**（+0.1 dB，box +0.4 dB，LPIPS −0.01），
   MMSE10 24.86 ≥ 泄漏参考 24.84，是本次最佳配置。
4. **schedule**：spectral 三种几乎无差（≤0.1 dB）；no_noise 下 2222 最优（random +0.3–0.4 dB）；
   pooled 下 2222 比 2211/2221 高 0.5 dB。均匀 S_it=2 不劣于 [2,2,1,1]。
5. 之前中途笔记「no_noise 28–29 dB」是个别 blur 图的单格值，6 图均值实为 25.2–25.5，已更正。

## 最终结果（1440 行 / 360 格，s_stats/test/s4_per_task.md；6 图均值，5 task 平均）
PSNR(dB) single / MMSE5 / MMSE10 / no_noise：
- pooled_all   2222: 14.50 / 19.53 / 21.06 / 25.15；2211: 13.94 / 19.04 / 20.66 / 24.62
- pooled_class 2222: 14.78 / 19.79 / 21.31 / 25.16；2211: 14.26 / 19.33 / 20.93 / 24.65
- spectral_all 2222: 21.14 / 24.11 / 24.72 / 25.42；2211: 21.13 / 24.03 / 24.62 / 25.26
- spectral_class 2222: 21.23 / 24.25 / 24.86 / 25.46；2211: 21.22 / 24.17 / 24.76 / 25.29
- ref:spectral_14img 2211: 21.67 / 24.33 / 24.84 / 25.26；ref:pooled_junco 2211: 18.78 / 22.76 / 23.77 / 24.87
SSIM single / MMSE10：pooled ~0.19 / 0.37；spectral ~0.38 / 0.61；ref14img 0.41 / 0.63；junco 0.27 / 0.51。
LPIPS-alex single / MMSE10：pooled 0.87-0.91 / 0.52-0.56；spectral 0.49-0.50 / 0.37-0.39；ref14img 0.48 / 0.39。

## 结论
1. **标量 val-set S 灾难性（带噪）**：s²≈0.27-0.31 比 junco 表大 5.5× ⟹ 先验精度 S⁻¹ 弱 5.5× ⟹
   RTO 抽样被似然噪声支配，单样本 13.8-14.8 dB（junco 表 18.8），SSIM 0.19；MMSE10 也只
   20.5-21.3。但 **no_noise 下 pooled ≈ spectral（25.15 vs 25.42）≈ junco（24.87）**：S 的量级
   几乎只通过噪声路径起作用，再次印证 all-img-tests「3ξ=0 后两臂收敛」。
2. **spectral val-set 是诚实的 spectral**：spectral_all 比含校准泄漏的 ref:spectral_14img 低
   0.5 dB（single）/0.1-0.2 dB（MMSE10）；泄漏只值 ~0.5 dB。spectral_class 再 +0.1-0.4 dB
   （box +0.4）、LPIPS 更低，MMSE10 24.86 ≈ ref 24.84 追平。**推荐 spectral_class（或
   spectral_all）替代 junco 表**；标量 val-set S 不可用。
3. schedule：spectral 三种几乎无差（±0.1 dB）；pooled 与所有 S 的 no_noise 下 2222 最优
   （+0.5 dB，random_inpainting +2 dB）——后期多步有利于确定性路径。
4. pooled 与 spectral 在 blur/SR 的 no_noise 上完全持平（25.5-25.9），差距只在
   random_inpainting（26.0 vs 27.9）：谱先验对散布测量的插值有实质帮助，对全域观测任务
   只影响噪声抑制。

## 状态
done。四 GPU 总时长 ~4.5 h（含 EIO/OOM 中断）；GPU1 曾因他人 72GB 任务 OOM 停启一次。
