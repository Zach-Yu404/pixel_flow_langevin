# full-ip-compare

state: working · owner: claude · type: experiment

## 【用户原始要求】（2026-08-16，逐字）

> 现在对于5种inverse problem，对于wls和model的按照/standard/CBIG-Standard-ECE/Zach/MSFlow/PixelFlow/IP_package/rerun_imageNet/configs_best_out中best的config跑inverse problem，对于algorithm2，按照pdf中的算法来跑（主要差异在于\hat x_0^k 和\hat x_1^k的计算），三种统一实验设置，画出随时间的loss curve，统计NFE和x_\tau以及\hat x_1^k, \hat x_0^k的变化.mp4文件（放在一个.mp4）

## 【Claude｜方案】

- 三方法統一设置：同 7 张 playground GT、同 y/operator（configs_best_out 的 operator 块经
  demo_runner.build_setup_and_measurement,PYTHONHASHSEED=0）、同 seed=42、同 stage 网格。
- **WLS / Model**：现有 `run_posterior_sampling`（不 fork）+ configs_best_out 全套 kw,
  仅 `x1_init_mode` 二值切换（'wls' / 'model'——采样器内建分支）；record_trajectory=True
  （junco）取 x_τ 与 x₁ 轨迹,x̂₀ 统一按 (x_τ−H x₁)/σ_τ 事后重建（= Alg2 line 15 定义）。
- **Algorithm 2**：按 PDF p.13 逐行实现（score solve Cor.8 → 随机 b̃ (34) 精确抽样 →
  line 15 x₀ 重推 → line 17 x₀-Langevin → line 20 U⁽¹⁾=nearest×2 + fresh x₀）。
  γ²(k,τ) 用实测表（gamma2_meas.json,从 git 9ece9d6 恢复）；h₀=0.1（论文 §7 建议的
  量级方向,单值不扫）；S、grid、guidance 与各任务 best config 相同 → NFE 同预算。
- **NFE**：model.forward 计数器实测,不按公式推。
- 输出 `Algorithm2/results/full_ip/`：loss 曲线（每任务面板,3 方法,x=全局 t,
  y=MSE(x̂₁,x₁ᵏ_gt)+x̂₀ 误差副面板）、nfe.json、full_ip_metrics.csv、
  **一个** `full_ip_trajectories.mp4`（junco;5 任务×40 帧分段;行=3 方法,列=x_τ|x̂₁|x̂₀）。

## 运行记录
（待填）
