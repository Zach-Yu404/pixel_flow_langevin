# alg2-onestep-mse-vs-t

state: review · owner: claude · type: experiment

## 【用户原始要求】（2026-08-16，逐字）

> 基于现有 PixelFlow/MSFlow 代码，在 `PixelFlowICLR` 下/standard/CBIG-Standard-ECE/Zach/MSFlow/PixelFlowICLR/Algorithm2扩展已完成的 onestep_mse_vs_t 实验：
> 在 WLS 和 Model 两个估计器之外，加入 Algorithm 2 的 one-step 估计器（两次 CG），
> 测试其 \hat x_1^k 预测相对 GT x_1^k 的 MSE vs. t。
>
> 【任务与数据 —— 与 onestep_mse_vs_t 完全一致】
> 5 个任务：box_inpainting / random_inpainting / gaussian_blur / motion_blur / superresolution，
> 各自使用对应 LPIPS_king config 参数（含 g_bypass_stage3=true、eta=sigma_n 等）。
> 测试图片：…playground_runs 中已有实验使用的 ImageNet GT 图片；y 的构造方式与 seed 与 playground_runs 一致。
> GT x_1^k、stage 网格 t、GT x_t^k 的构造：逐行复用 onestep_mse_vs_t 已有的 helper 和循环，不要重新实现。
> 构造 x_t^k 时固定 x_0^k ~ N(0,I)（固定 seed），并把这个 x_0^k 保留下来。
>
> 【新增算子】apply_B = e_k·x − s_k·G(x)；apply_N = e_k(1−s_k)·x − s_k(1−e_k)·G(x)（G=apply_G, stage_idx=eff）。
> A_k/A_k^T 复用现有构造；无现成 adjoint 的按 _interpolate_adjoint 模式 autograd 取精确 adjoint，先做 adjoint test（<1e-4）。
>
> 【Algorithm 2 one-step】σ_τ<0.01 跳过；一次网络调用 v；
> line 11（score solve, CG）：[N²+γ²H²] x̂₀ = N(B x_t − H v)，γ²=0 主曲线；
> line 14（clean solve, CG, 确定性）：M = AᵀA/η² + H²/σ² (+ridge@τ=0: 1e-6·||M||，幂迭代 20 步)，
> b = Aᵀy/η² + H x_t/σ²，warm start = direct_estimate_x1，τ≈0 时 max_iter=200。
> 同一 x_t 上照算 WLS/Model 基线。
> 记录：三估计器 MSE；inpainting 分 observed/missing；score 误差 ‖x̂₀−x₀‖²/n；
> γ²_meas = mean_i‖v−d_exact‖²/n（d_exact = B x₁ − (e−s)x₀，按 (task,k,t) 存表）；
> 可选：γ²=γ²_meas 重跑 score solve（v 已缓存）。
>
> 【sanity】S1: v:=d_exact, γ²=0 → x̂₀ 必须恢复 x₀（恒等式）；S2: GT x_t + 无噪 y 跑 line 14，
> inpainting observed 区域误差 ~η 量级。不过就先修 wiring。
>
> 【输出】每 task 一张图：全局 t（四 stage 拼接、边界竖线），log-y，Alg2/WLS/Model
> （inpainting 加 obs/missing 版）；另存 score 误差与 γ²_meas 图；raw 数据（.pt/CSV）+ γ²_meas 表。
> 只做 one-step。复用现有代码。GPU（A100, Slurm），不碰 playground_runs。先检查代码再直接完成。
> 还是得到和之前一样的展示方式，进行对比。

## 【Claude｜方案】

- 实现位置：`PixelFlowICLR/Algorithm2/algorithm2.py`（用户已建空文件）+ `run_alg2.sbatch`。
- 复用：`import onestep_mse_vs_t as base`（金字塔/ε=x₀/rope/config 查找/循环骨架逐行同源）;
  y/operator = `demo_runner.build_setup_and_measurement`（playground 同代码路径,每任务 seed 契约）;
  数学件 = final_utils 的 apply_G/apply_H_tau/compute_sigma_tau/cg_solve/direct/wls/make_velocity_fn。
- 已解析验证 S1 恒等式：B·x_τ − H_τ·d_exact = N·x₀（σ_τB+(e−s)H_τ = N 恒等）。
- 关键事实：v、score solve、γ²_meas 与任务无关（不含 A/y）→ v 按 (stage,t) 缓存跨任务复用
  （kw 已验证逐位一致），line 14 才逐任务逐图。
- ⚠️ demo_runner 的 make_random_mask/make_box_mask 仍用 Python hash()（registry #6 残留）
  → 实验固定 PYTHONHASHSEED=0 保证可复现，构造方式/种子代码路径不变。

## 运行记录
- self-test（CPU）：adjoint worst 4.8e-9、S1 1.1e-7、S2 stage3 obs 0.011 —— 全过
- job 18591733（A100, 5m28s）→ results/ 全产物；详见 experiments/2026-08-16-alg2-onestep-mse-vs-t.md
