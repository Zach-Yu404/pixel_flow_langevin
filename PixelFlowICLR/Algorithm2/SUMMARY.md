# Algorithm 2 — 代码与论文逐步对照（精简版）

论文：ICLR 草稿 p.13 Algorithm 2（Exact conditional draw for x₁ᵏ + Langevin for x₀ᵏ, no h₁）。
🔧 = Claude 建议的扩展（非论文原文，默认关闭）。

## 文件一览（PixelFlowICLR/Algorithm2/）

| 文件 | 功能 |
|---|---|
| `algorithm2.py` | **one-step 估计器实验**（实验 2）：单次 line 11 + line 14，对 GT x_t 测 MSE；含 S1/S2 sanity、γ²_meas 测量 |
| `full_ip_compare.py` | **完整迭代采样**（实验 3）：`sample_alg2` 逐行实现 p.13 全循环；vs WLS/Model 全逆问题对比 |
| `debug_box_h0.py` | box 洞区调试 sweep（h₀ / 🔧anchor 双模式，obs/hole 分离 MSE） |
| `verify_gamma2.py` | γ²=γ²_meas vs 0 差异复算（review 口径脚本） |
| `gamma2_meas.json` | 按 (stage, τ) 存的 γ²_meas 表（式 56 测量值） |
| `alg2_visual.py` / `unified_onestep.py` | 预测可视化 / 统一对比（y=GT 探针版） |
| `run_*.sbatch` | Slurm A100 作业脚本（PYTHONHASHSEED=0） |
| `results/debug_box_{h0,anchor,confirm}/` | 调试证据（h₀ 证伪、anchor 剂量响应、7 图复核） |

## 论文行 → 代码映射

| 论文行 | 内容 | 代码位置 |
|---|---|---|
| l.1 / l.3 | x₁⁰=0；x₀ᵏ~N(0,I) 每 stage | `full_ip_compare.sample_alg2`（"line 1"/"line 3" 注释） |
| l.5 | H_τ、σ_τ | **复用** `final_utils.apply_H_tau` / `compute_sigma_tau` |
| l.6 | B_k、N_k | `algorithm2.apply_B` / `apply_N`（:66/:71，g_bypass_stage3 贯通） |
| l.7 | M_τ = AᵀA/η² + H²/σ² + εI | 两文件内 `M0` 闭包；ε 仅 τ=0（Prop. 4），`power_iter_norm`(:76) 估 ‖M‖ |
| — | A_k/A_kᵀ | **复用** demo_runner 算子；blur/motion 用 `make_exact_AT`(:89) autograd 精确伴随（含 strict-det 守卫） |
| l.9 | x_τ = H x₁ + σ x₀ | `sample_alg2` 内循环首行 |
| l.10 | v = v_θ(x_τ,τ,k)（唯一网络调用） | **复用** `final_utils.make_velocity_fn`（CFG + stage 调制） |
| l.11 | score solve：[N²+γ²H²]x̂₀ = N(Bx_τ−Hv) | `algorithm2.score_solve`(:133)，CG=`final_utils.cg_solve`；γ² 查 `gamma2_meas.json` |
| l.12-13 | ξ_y, ξ_h；b̃（式 34） | `sample_alg2` l.13 段；**one-step 版**（`alg2_x1_solve`:146）按实验 spec 用确定性 b（无 ξ） |
| l.14 | 解 M_τ x₁ = b̃（精确采样，无 h₁） | `cg_solve`，warm start = direct_estimate_x1，τ=0 时 max_iter=200 |
| l.15 | x₀ ← (x_τ − H x₁)/σ | `sample_alg2` l.15 行 |
| l.16-17 | x₀ Langevin：x₀ − (h₀/2)(x₀+x̂₀) + √h₀ξ₀（式 36） | `sample_alg2` l.17 行；h₀ 已参数化（默认 0.1） |
| l.20 | U⁽¹⁾ nearest 上采样 + fresh x₀ | `sample_alg2` stage 末 `F.interpolate(nearest)` |
| σ_τ<0.01 跳过 | — | `SIGMA_MIN` 门（两文件一致） |

## 🔧 Claude 建议部分（默认关闭，采纳与否待用户裁决）

1. **Tweedie 锚定 Block 1**：`sample_alg2(anchor=λ)`——把 l.10 的 v 得到的
   `x̂₁_model = direct_estimate_x1(x_τ−τv, x_τ+(1−τ)v)` 作为精度 λ 的高斯伪观测：
   `M += λI`，`b̃ += λ·x̂₁_model + √λ·ξ_a`（保持 Lem. 5 精确采样语义，零额外 NFE）。
   动机：式 (62) 的条件分布在 ker(A) 内无 x₁ 先验 → box 洞区为无收缩随机游走
   （oracle 仿真：完美 score 下 std 仍 24× 超标；h₀/S 非杠杆）。
   证据：λ=25 时 7/7 图洞区 MSE 降 3.2×–13.7×（池化 1.097→0.178），obs 区无回退，
   洞区出现语义填充。若采纳，相当于论文式 (33)/(34) 各加一项先验伪观测。
2. **`make_exact_AT` 的 strict-deterministic 守卫**（已提交）：reflection_pad2d_backward
   无确定性 CUDA 实现，按仓库 warn_only 惯例包裹并恢复调用方模式。
3. **h₀ 参数化**（已提交）：原为模块常数 0.1；扫描证明 h₀ 非洞区杠杆（0.5/1.0 更差），
   保留参数仅供实验。

调试全链路证据：`.research/tasks/debug-box-alg2-hole.md` + 2026-08-17 handoff。
