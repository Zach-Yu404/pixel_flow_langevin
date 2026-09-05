# 当前状态（人类可读，保持 ≤1 页）

更新时间：2026-09-05（finalize）

## 刚完成：clean-alg4-project（tasks/clean-alg4-project.md，2026-09-05）
- PixelFlowICLR/Algorithm4：自包含干净版（alg4 包、run.py、config.json、vendored pixelflow、scripts、data、tests、README）。
  35 格验收与 results/alg4 参考一致（|Δ|≤8e-6），逐行审查无偏差。

## 已完成：finalize-alg4-config（tasks/finalize-alg4-config.md，2026-09-05）
- 用户决定 γ² 用 gamma2_all（config 一行）。审计后删掉 random_inpainting 的 num_langevin=15 覆盖、
  归档旧 results/alg4 → alg4_pre_final_0824，最终参考跑 `python main4.py` 已写 results/alg4（35 格）；FINAL.md（含参考表）交用户 check。

## 已完成：gamma2-tables-test（tasks/gamma2-tables-test.md，2026-09-05）
- spectral_class 下 γ² 表三臂 × 5 task × 7 图：val 表全噪声 −0.03 dB（105 格 0 胜）、MMSE10 −0.01、无噪持平/略优；
  per-class 表更差且方差大；GPU 噪声地板 0。结论：γ² 表不敏感，保留 baseline 或切 gamma2_all，不用 per-class。

## 已完成：compute-gamma2-valset（tasks/compute-gamma2-valset.md，2026-09-05）
- γ²(k,τ) 全 ImageNet val 50k（4 卡 16.5 h；74 类 fp32 + 926 类 TF32）：gamma2_stats/gamma2_all.json（drop-in）
  + gamma2_labelled.json（1000 类）。比 7 图旧表高 25–55%（stage 0–2、stage 3 早期），stage 3 晚期一致；
  demo 7 图对网络"更容易"。用户已决定：切 gamma2_all（2026-09-05）。

## 已完成：S 固定为 spectral_class（tasks/default-spectral-class-s.md，2026-09-03）
- main4 硬编码 s_stats per-class 谱 S（S_STATS/default_s2_fn/_bind_s2），config 删除 S_prior 段；
  与 S4 构造逐位一致（功率谱 torch.equal），完整采样差=GPU 运行噪声（~1e-3），CLI smoke 通过。
- 后续跑的所有 main4 结果默认都是这个 S；旧 junco 表只剩 diversity 消融臂。

## 已完成：s4-valset-s-test（tasks/s4-valset-s-test.md，2026-09-03）
- （已完成 09-03，见下）4 种 val-set S（pooled_all/pooled_class/spectral_all/spectral_class）× schedule
  {2222,2211,2221} × all_img_tests 网格（5 task × 6 图）× 估计量 {single_avg10, MMSE5,
  MMSE10, no_noise}；PSNR/SSIM/LPIPS(piq-VGG+alex)/MSE。四 GPU claim 工作窃取
  （s_stats/test/_claims/），输出 s_stats/test/s4_results.csv，聚合 s4_per_task.md。
- 结果（5 task 均值 PSNR 单样本/MMSE10/no_noise）：pooled_all 14.5/21.1/25.2，
  spectral_class 21.2/24.9/25.5，旧 spectral_14img 21.7/24.8/25.3，旧 pooled_junco 18.8/23.8/24.9。
  → 标量 val-set S 只在噪声路径上灾难，确定性中心持平；spectral_class+S_it=2 为最佳配置，
  与（泄漏的）旧 spectral 持平或更好；schedule 对 spectral 无差。

## 已完成（2026-08-31 ~ 09-02）
- compute-s-stats-valset：全 50k val 单遍统计（无 inference，433 s），4 文件在 s_stats/；
  ALL pooled s²=0.267–0.308，junco 表低 5.5×。
- all-img-tests + MMSE：MMSE10 比单样本 +2.6–5.7 dB；**3ξ=0 单趟 ≈ MMSE10 的 PSNR，
  SSIM/LPIPS 更优，成本 1/10**（MMSE_results/mmse_vs_noiseoff_per_task.csv）。
- xih-zero-probe 系列（ξ_h=0、3ξ=0、ξ₀:=ξ_h、x₀=0@f38、S_it sweep）与 cg-maxiter-test：
  噪声归因链闭合，见各 task 文件。

## 已撤回（用户指令，代码位级还原，记录保留）
- eq22-sigma2-scaling（results/alg4_eq22_sigma2_scaling/ 保留 4 份记录）；
- block2-adaptive-pcn（results/alg4_weighted_sigma_tau/ 已删）；
- drop-(1/σ²)HᵀH from C⁻¹ 探针。

## 阻塞 / 待用户决定
- m=0（19′）是否恢复。
- 记忆 v58+ 未提交/推送（v57 已在 IP_branch）；S4 结束后统一 commit+push。
