# Algorithm 2 — 代码与论文逐步对照（精简版）

论文：ICLR 草稿 p.13 Algorithm 2（Exact conditional draw for x₁ᵏ + Langevin for x₀ᵏ, no h₁）。
🔧 = Claude 建议的扩展（非论文原文，默认关闭）。
变量/函数命名约定：论文符号优先（S/L/h0/gamma2/eta/epsilon/x1/x0/x_tau/sigma_tau/M_tau/b_tilde/
xi_y/xi_h/xi_0/s_k/e_k），论文未命名的沿用旧 inference 代码（velocity_fn/prompt_embeds/eff_si 等）。

## 文件一览（PixelFlowICLR/Algorithm2/）

| 文件 | 功能 |
|---|---|
| `algorithm2.py` | **one-step 估计器实验**（实验 2）：单次 line 11 + line 14，对 GT x_t 测 MSE；含 S1/S2 sanity、γ²_meas 测量 |
| `full_ip_compare.py` | **完整迭代采样**（实验 3）：`run_posterior_sampling_alg2` 逐行实现 p.13 全循环，**段落结构与旧 inference `run_posterior_sampling` 逐段对齐**（CFG setup → 尺寸推导 → stage 循环 → 内循环），便于并排 diff；vs WLS/Model 全逆问题对比 |
| `debug_box_h0.py` | box 洞区调试 sweep（h₀ / 🔧anchor 双模式，obs/hole 分离 MSE） |
| `verify_gamma2.py` | γ²=γ²_meas vs 0 差异复算（review 口径脚本） |
| `gamma2_meas.json` | 按 (stage, τ) 存的 γ²_meas 表（式 56 测量值） |
| `alg2_visual.py` / `unified_onestep.py` | 预测可视化 / 统一对比（y=GT 探针版） |
| `run_*.sbatch` | Slurm A100 作业脚本（PYTHONHASHSEED=0） |
| `results/debug_box_{h0,anchor,confirm}/` | 调试证据（h₀ 证伪、anchor 剂量响应、7 图复核） |

## 论文行 → 代码映射（`run_posterior_sampling_alg2`，变量名=论文符号）

| 论文行 | 内容 | 代码 |
|---|---|---|
| l.1 / l.3 | x₁⁰=0；x₀ᵏ~N(0,I) 每 stage | `x1 = zeros`；`x0 = randn_like_cpu(...)` |
| l.5 | H_τ、σ_τ | **复用** `final_utils.apply_H_tau` / `compute_sigma_tau` → `sigma_tau` |
| l.6 | B_k、N_k | `algorithm2.apply_B` / `apply_N`（g_bypass_stage3 贯通 `eff_si`） |
| l.7 | M_τ = AᵀA/η² + H²/σ² + εI | `M0` 闭包 + `epsilon`（Prop. 4，仅 τ=0，`power_iter_norm` 估 ‖M‖）→ `M_tau` |
| — | A_k/A_kᵀ | **复用** demo_runner 算子 → `Ak, ATk`（旧 inference 命名）；blur/motion 走 `make_exact_AT` autograd 精确伴随 |
| l.8 | for s = 1..S | `for s in range(S)`（S = kw num_langevin） |
| l.9 | x_τ = H x₁ + σ x₀ | `x_tau = apply_H_tau(x1,...) + sigma_tau * x0` |
| l.10 | v = v_θ(x_τ,τ,k)（唯一网络调用） | **复用** `make_velocity_fn` → `velocity_fn`、`v` |
| l.11 | score solve：[N²+γ²H²]x̂₀ = N(Bx_τ−Hv) | `algorithm2.score_solve(..., cg_tol, L)`；`gamma2` 查 `gamma2_meas.json` |
| l.12-13 | ξ_y, ξ_h；b̃_τ（式 34） | `xi_y`/`xi_h`；`b_tilde`；**one-step 版**（`algorithm2.clean_image_solve`）按实验 spec 用确定性 b |
| l.14 | 解 M_τ x₁ = b̃（精确采样，无 h₁） | `cg_solve(M_tau, b_tilde, ...)`，τ=0 时 max_iter=200 |
| l.15 | x₀ ← (x_τ − H x₁)/σ | 同式直译 |
| l.16-17 | ξ₀；x₀ Langevin（式 36，h₀） | `xi_0`；`x0 - (h0/2)(x0 + x0_hat) + √h0·xi_0` |
| l.20 | U⁽¹⁾ nearest 上采样 + fresh x₀ | stage 顶 `F.interpolate(nearest)` + fresh `x0`（对应旧 inference 的 renoise 过渡段位置，差异注释标明） |
| σ_τ<0.01 跳过 | — | `SIGMA_MIN` 门（两文件一致） |

## 函数命名对照

| 现名 | 来源 | 旧名 |
|---|---|---|
| `run_posterior_sampling_alg2` | 旧 inference `run_posterior_sampling` 命名系，签名/段落同构 | `sample_alg2` |
| `clean_image_solve` | 论文 "clean-image solve"（Lem. 5） | `alg2_x1_solve` |
| `score_solve` | 论文 "score solve"（Cor. 8） | （未变） |

## 🔧 Claude 建议部分（默认关闭，采纳与否待用户裁决）

1. **Tweedie 锚定 Block 1**：`run_posterior_sampling_alg2(anchor=λ)`——l.10 的 v 免费得到
   `x1_model = direct_estimate_x1(x_τ−τv, x_τ+(1−τ)v)`，作精度 λ 的高斯伪观测：
   `M_tau += λI`，`b_tilde += λ·x1_model + √λ·ξ_a`（保持 Lem. 5 精确采样语义，零额外 NFE）。
   动机：式 (62) 的条件分布在 ker(A) 内无 x₁ 先验 → box 洞区为无收缩随机游走
   （oracle 仿真：完美 score 下 std 仍 24× 超标；h₀/S 非杠杆）。
   证据：λ=25 时 7/7 图洞区 MSE 降 3.2×–13.7×（池化 1.097→0.178），obs 区无回退，
   洞区出现语义填充。若采纳，相当于论文式 (33)/(34) 各加一项先验伪观测。
2. **class_label 公平性修复**（已提交）：旧 `full_ip` main 里 Alg2 兜底用类别 10 跑 CFG，
   而 wls/model 用真实类别（best config kw 无 class_label）→ 旧 `full_ip_metrics.csv` 的
   alg2 行不公平，**引用前需重跑**；现与 wls/model 同款传 `class_label=int(d["class_idx"])`。
3. **`make_exact_AT` strict-deterministic 守卫**（已提交）：修 reflection_pad backward
   无确定性实现的崩溃。
4. **h₀ 参数化**（已提交）：扫描证明 h₀ 非洞区杠杆（0.5/1.0 更差），仅留作实验旋钮。

调试全链路证据：`.research/tasks/debug-box-alg2-hole.md` + 2026-08-17 handoff。
