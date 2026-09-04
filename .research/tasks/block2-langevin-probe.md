# block2-langevin-probe（2026-09-03，done）

## 用户原始要求（逐字）
「现在测试对于equation 23, 用CG计算[N_k**2 + gamma**2 (H_tau^k )**2] \hat x_0 = N_k(B_k x_tau^k - H_tau^k v),
然后x0 = (x_tau - apply_H_tau(x1, tau, s_k, e_k, eff_si)) / float(sigma_tau)，再然后计算x_0^k = x_0^k - h_0/2(x_0 + \hat x_0)
+\sqrt(h_0) xi_0， x_tau = apply_H_tau(x1, tau, s_k, e_k, eff_si) + float(sigma_tau) * x_0」；
随后「只用对junco box进行测试」。

## 解读
即把 Algorithm 2 的 l.11（score_solve）+ l.15（x₀ 重算）+ l.17（Langevin 步）搬进 Algorithm 4 的 Block 2，
替换 (23) 的 exact draw。x̂₀ 用当前 inner 的 x_τ（l.8/(23)）与其 l.10 速度 v；x₀ 用 Block‑1 后的新 x₁ 重算；
ξ₀ 仍是同一次 RNG 抽样。h₀ 用户初未给：Alg 2 默认 H0=0.1，另测 0.5、1.0；随后用户指定 5e-2、1e-2、5e-3、1e-5。

## 实现
utils.run_posterior_sampling_alg4 新增 diag_block2_langevin=None（h₀；默认关闭 → 与 (23) 位级一致；
else 分支即原代码）。驱动 scratchpad/block2_langevin.py（junco box、
seed 42/43/44、[2,2,1,1]、默认 S=spectral_class），输出 results/alg4_block2_langevin/<arm>/junco_s<seed>/
{final.json, trajectory_metrics.csv, x1_final.png, traj_x1.npy}，聚合 block2_aggregate.py →
summary.{csv,md}、trajectory.png（各臂 seed42 轨迹 montage）、mse_hole_vs_frame.png。

## 结果（results/alg4_block2_langevin/summary.md；junco box，seeds 42/43/44 mean±std）
| 臂 | hole MSE | full MSE | PSNR(range2) |
|---|---|---|---|
| baseline (23) exact draw | 0.1369±0.0021 | 0.0362±0.0005 | 20.43±0.06 |
| h₀=1.0 | 0.0870±0.0110 | 0.0234±0.0028 | 22.35±0.53 |
| **h₀=0.5** | **0.0749±0.0050** | **0.0202±0.0013** | **22.98±0.27** |
| h₀=0.1 | 0.0994±0.0172 | 0.0263±0.0043 | 21.87±0.69 |
| h₀=5e-2 | 0.1241±0.0164 | 0.0326±0.0041 | 20.92±0.58 |
| h₀=1e-2 | 0.1538±0.0117 | 0.0403±0.0029 | 19.98±0.31 |
| h₀=5e-3 | 0.1621±0.0153 | 0.0425±0.0038 | 19.76±0.38 |
| h₀=1e-5 | 0.1794±0.0198 | 0.0469±0.0049 | 19.33±0.45 |
CG 全收敛；每格 ~17 s。产物：<arm>/junco_s<seed>/{final.json, trajectory_metrics.csv, x1_final.png,
traj_x1.npy}，trajectory.png（各臂 seed42 轨迹 montage）、mse_hole_vs_frame.png、summary.{csv,md}。

## 机制（符号疑点已解决：用户原式正确）
x_τ = Hx₁ + σx₀、v = Bx₁ − (e−s)x₀ + ε、恒等式 (e−s)H + σB = N ⟹ B x_τ − H v = N x₀ − Hε，
x̂₀ ≈ +x₀（收缩）。关键：l.15 重算的 x₀_cur 与 x̂₀ 是**同一个噪声的两个估计**，
σx₀_cur = x_τ − Hx₁ⁿᵉʷ，σx̂₀ ≈ x_τ − Hx₁ᵐᵒᵈᵉˡ，所以 x₀ − h/2(x₀ + x̂₀) 不是"推离"而是收缩：
x₀ ← (1−h)x₀ + √h ξ（AR(1)，平稳方差 1/(2−h) < 1，h<1 时链更"冷"），且
h=1 时 x_τⁿᵉʷ = H·½(x₁ⁿᵉʷ + x₁ᵐᵒᵈᵉˡ) + σξ₀ —— 即 (23)，但 Block‑1 样本换成样本与模型端点估计的平均，
送入下一次网络输入的 RTO 噪声减半（与「ξ_h 主导后期噪声」归因一致；增益 +2.5 dB ≈ MMSE10 的量级，
hole MSE 0.075 优于 3ξ=0 的 0.081）。我此前"固定 x̂₀ 的平稳律 N(−x̂₀,I)"的担心不成立（x̂₀ 随 x₀ 变）。
**h₀→0 极限不是 baseline 而更差**（1e-5：0.179 > 0.137）：h→0 时 x₀_new ≈ x₀_cur = (x_τ_old − Hx₁ⁿᵉʷ)/σ，
于是 x_τ_new = x_τ_old —— Block 2 退化为空操作，x_τ 的噪声实现整段不再刷新（无新鲜 ξ₀），
按空间 pinning 定律 box 洞内噪声被冻结带到晚期。h₀ 因此是「刷新比例」：exact draw = 完全刷新
（独立 ξ₀），h₀=0.5 = 半刷新 + 样本/模型端点平均，单峰最优；0.1 以下逐步退化到冻结极限。
诊断定性：它不是 (23) 的 exact draw，x_τ 的条件律被改（更冷 + 样本平均），属于 x_τ 噪声的显式阻尼。

## 撤回与追加（用户 2026-09-03）
「撤回flip的相关代码和内容，然后h_0 = 5e-2, 1e-2, 5e-3, 1e-5」：符号开关、对照臂及其结果目录/记录全部删除；
追加 h₀ ∈ {5e-2, 1e-2, 5e-3, 1e-5}（同 junco box、seeds 42-44）。

## 状态
done（2026-09-03）。代码：utils.py diag_block2_langevin（默认关，位级不变）。产物：results/alg4_block2_langevin/
{summary.md, summary.csv, trajectory.png, mse_hole_vs_frame.png, <arm>/junco_s<seed>/…, block2_langevin.py,
block2_aggregate.py}。待用户决定是否把 h₀=0.5 变体设为默认 / 扩到其他 task。
