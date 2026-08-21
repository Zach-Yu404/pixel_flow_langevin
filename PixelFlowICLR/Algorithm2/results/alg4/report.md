# Algorithm 4 — clean-endpoint sampler: implementation and first full run

草稿：`results/algorithm.4pdf.pdf`（"A clean-endpoint posterior sampler for cascaded
flow priors"；草稿内部把自己的 listing 编号为 "Algorithm 1"，本项目按文件名称 **Algorithm 4**）。
精读：`.research/references/2026-08-21-algorithm4-clean-endpoint-sampler.md`。

代码：`utils.py` 纯新增（`make_endpoint_operator` / `clean_endpoint_solve` /
`make_M_tau_den` / `measurement_residual` / `run_posterior_sampling_alg4`），
入口 `main4.py` + `config_alg4.json`。既有三个采样器（alg2 / wv / reg）一行未动。

跑法：`PYTHONHASHSEED=0 python main4.py --mode {verify|measure_s2|full_ip}`。

## 1. 与 Algorithm 2 的三处结构差异

都是草稿的主张，不是实现选择：

1. **l.8 在内循环之外**。Alg 2 每次内迭代都从 (x1, x0) 重建 x_τ；Alg 4 里 x_τ 是内循环的
   状态量，由 Block 2 (23) 更新，l.8 每个 τ 只进一次坐标。
2. **Block 1 的中心是 x̂₁ 而不是 x_τ**。整个 `C⁻¹ = H_τᵀH_τ/σ_τ² + S⁻¹` 都作用在干净端点上
   (22)，而 Alg 2 的右端带的是 `H_τᵀx_τ/σ_τ²`。
3. **Block 2 是直接抽样 (23)，没有 h₀**。§6.3：没有任何步长能复现精确抽样
   （drift 要 h=2、noise 要 h=1）。

`h0` / `ridge_rel` / `cg_max_iter_l14` 被 config 契约与采样器**双重拒绝**而非静默忽略。
`S_prior` 无代码默认值——草稿的主张是这个量被**测量**而非搜索，给默认值等于把被消除的
可调参数偷偷放回来。

## 2. verify：17 项 dense float64 审计全过

`python main4.py --mode verify`（CPU，无需 GPU）。两项值得单独说：

- **A2 — Prop. 7 数值成立**：(19) 的解与「(18) 求 x̂₀ 再乘 H_τ⁻¹」吻合到 `5e-15`
  （γ² = 0 / 0.01 / 0.2 三档）。
- **A4 — 去掉 ridge 的是 Prop. 4(a)，不是容差**：τ=0 处
  `λ_min(H₀ᵀH₀/σ²) = -4.6e-18`（这正是逼 Alg 2 必须加 ridge 的奇异性），
  而 `λ_min(M_τ^den)` **恰好等于 1/s²**。
- 另有 A1（(18)/(19) 共用算子一致）、A3（γ→0 回到 (17)、γ→∞ 回到 H_τ⁻¹x_τ、
  以及 v=d_exact 且 γ²=0 时 x̂₁ 精确等于 x₁）、A5（Lemma 9 的 Cov(ζ) 逐元等于 M_τ^den）、
  A6（l.16 恰好取回 l.14 抽的 ξ₀）、A7。

## 3. measure_s2：S_prior 的实测值

`python main4.py --mode measure_s2` → `results/alg4/s2_meas.json`。
stage pyramid 的逐像素方差（7 张 demo 图 pooled）：

| stage | 0 | 1 | 2 | 3 |
|---|---:|---:|---:|---:|
| s² | 0.2001 | 0.2111 | 0.2204 | 0.2282 |
| λ 等效 (1/s²) | 5.00 | 4.74 | 4.54 | 4.38 |

**注意口径**：这是在 demo/eval 图上测的，不是训练集，严格说是泄漏。
报告前必须换成训练集统计。图像归一化到 [-1,1]，
所以 Popoviciu 的 assumption-free 上界是 `s² ≤ (b-a)²/4 = 1`
（草稿 §8.2 写 1/4，那是 [0,1] 的情形）。

## 4. full_ip：5 任务 × 7 图（35 格，全部完成）

`results/alg4/full_ip_final.csv`、`full_ip_metrics.csv`、`loss_curves.png`。
S_prior 用上面的实测表；其余配置逐字沿用 `config.json` 的 `tasks_setup`
（含 `terminal_replace_weight = 0`，所以 post = pre）。NFE = 390/图（random_inpainting 585）。

### 各 stage 末的 mse_x1（7 图均值）

| task | stage0 | stage1 | stage2 | **stage3** |
|---|---:|---:|---:|---:|
| box_inpainting | 0.0697 | 0.0706 | 0.1364 | **0.9217** |
| random_inpainting | 0.0118 | 0.0220 | 0.0770 | **2.0161** |
| gaussian_blur | 0.0120 | 0.0648 | 0.2112 | **2.2952** |
| motion_blur | 0.0430 | 0.0842 | 0.2009 | **2.3413** |
| superresolution | 0.0128 | 0.0320 | 0.1928 | **2.3911** |

**跨 stage 不可直接比**：`mse_x1` 是对 `pyr[si]`，即该 stage 分辨率上的 GT，
stage 3 是全分辨率。stage 内逐步比较才是同口径的。

### §8.6 要求报告的量

- measurement residual `‖A_k x1 − y‖/(η√m)`，末步 7 图均值：
  box 0.910 / random 1.060 / gaussian 1.118 / motion 1.291 / **SR 3.747**。
  草稿说应稳定在 1 附近——**SR 明显不满足**，它的数据项本身就没拟合好。
- `x0_rms`（草稿 §7 的诊断 ‖x₀‖²/n ≈ 1）：1365 个 step 上均值 **0.9991**，
  范围 [0.9699, 1.0218]。**Block 2 的直接抽样与 l.16 的坐标变换完全按设计工作**
  ——失败不在这里。

## 5. 主要发现：全部亏损集中在 stage 3，且五个任务无一例外

### 5.1 stage 2 平稳，stage 3 单调劣化（stage 内同口径）

stage 3，7 图均值 `mse_x1`：

| step | τ | σ_τ | γ² | box | random | gaussian | motion | SR |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.000 | 0.4000 | 0.0094 | 0.096 | 0.121 | 0.165 | 0.161 | 0.188 |
| 1 | 0.111 | 0.3556 | 0.0109 | 0.340 | 0.333 | 0.844 | 0.773 | 0.931 |
| 2 | 0.222 | 0.3112 | 0.0137 | 0.366 | 0.897 | 1.145 | 1.123 | 1.307 |
| 5 | 0.555 | 0.1780 | 0.0183 | 0.671 | 1.446 | 1.657 | 1.654 | 1.793 |
| 8 | 0.888 | 0.0448 | 0.0447 | 0.922 | 2.016 | 2.295 | 2.341 | 2.391 |

stage 2 同样的表在整段上是**平的**（box 0.087→0.136、random 0.091→0.077、
SR 0.183→0.193），σ_τ 从 0.667 走到 0.250、γ² 从 0.0116 走到 0.0166。

### 5.2 起点不是极端 σ_τ 或极端 γ²

stage 3 内**最大的一次相对跳变发生在 step 0→1**：box 0.096→0.340（3.5×）、
SR 0.188→0.931（5.0×）。而那一步的 σ_τ 只从 0.400 降到 0.356、γ² 只从 0.0094 升到 0.0109
——都是该 stage 内最温和的条件。

所以：**劣化的"起始"不能用 σ_τ→0 或 γ² 变大解释**；
后续的单调恶化则与两者都一致（σ_τ 一路到 4e-4、γ² 一路到 0.145）。
这两句必须分开说。

### 5.3 与 anchored Algorithm 2 的对照（box/junco，同一格）

`s² = 1/λ`，所以 S_prior 与 Tweedie anchor 的 λ 在同一根轴上
（`results/alg4/s2_sens/comparison.csv` vs `test/results_reg_alg2/comparison.csv`）：

| | λ_eq 4.7(实测) | 1 | 25 | 100 | 400 |
|---|---:|---:|---:|---:|---:|
| **Alg4** stage0 | 0.227 | 0.728 | 0.114 | 0.084 | 0.064 |
| **Alg4** stage1 | 0.218 | 0.617 | 0.090 | 0.059 | 0.052 |
| **Alg4** stage2 | 0.473 | 0.504 | 0.138 | 0.106 | 0.088 |
| **Alg4** stage3 | 3.522 | 3.465 | 3.253 | 1.345 | 1.140 |
| **Alg2+anchor** stage3 | — | — | 0.115 | 0.102 | 0.112 |

（Alg2+anchor 在 stage0–2 的对应值：λ=25 → 0.117/0.095/0.117；
λ=100 → 0.085/0.067/0.094；λ=400 → 0.064/0.061/0.095。）

**stage 0–2 上 Algorithm 4 在同一 λ 下持平或略胜；stage 3 差 10–28 倍。**

### 5.4 S_prior 的方向与 §8.2 的建议相反（但不构成反驳）

hole 随 s² 单调下降：s²=1.0 → 3.465、实测 0.21 → 3.522、0.04 → 3.253、
0.01 → 1.345、0.0025 → 1.140。草稿 §8.2 建议"宁可取大"，
而这里 `s²=1.0` 在每个 stage 都不比实测值更好，有用的方向是**更小的 s²**。

这不反驳草稿：§8.2 明说低估 S 会把 Block 1 收缩向 x̂₁，
**重建指标反而变好而样本多样性丧失**，且该失败在 MSE 上看不出来。
要分辨"真的更好"和"塌缩到去噪器"，必须跑 §8.6 的诊断：
固定 y 重复采样，量 `ker(A_k)` 内的 spread。**该诊断尚未跑。**

## 6. 下一步（按信息量排序）

1. **`g_bypass_stage3` 的作用**。stage 3 是唯一 `G = I` 的 stage
   （`eff_si=3` 使 `apply_G` 变恒等），于是 H_τ、N_k 全退化为标量，
   interpolant 不再有任何低通作用；同时 l.18 的 `U⁽¹⁾` 是 **nearest** 上采样。
   Alg 4 的 Block 1 完全以 x̂₁ 为中心，没有 Alg 2 那个保留已抽状态的 `H_τᵀx_τ/σ_τ²` 项。
   这是目前最可能的机制，但**未经检验**。
   决定性实验是关掉 `g_bypass_stage3` 跑一格 ——
   **但 CONSTRAINTS §采样器纪律明令"`g_bypass_stage3=True` 不关"，
   所以这一步需要用户先裁决**（作为诊断而非调参是否可以破例）。
2. **§8.6 的多样性诊断**：固定 y 重复采样，量 ker(A_k) 内 spread。
   不跑这个，5.4 里"更小的 s² 更好"就不能写进任何结论。
3. **γ² 通道的 ablation**：把 γ² 冻结在 stage-0 水平重跑 stage 3，
   与实测 γ² 表对照，分离"velocity error 变大"与"σ_τ→0"两条通道（§8.4 的 crossover）。
4. **S_prior 换成训练集统计**，消除 §3 的泄漏口径。

## 7. 复现与已知运行问题

- `PYTHONHASHSEED=0` 是硬契约（demo_runner 的 mask seed 走 `hash()`）。
- 本机 ceph 挂载在并发下频繁 `Remote I/O error`，会打断 Python 的 import 扫描。
  因此 `run_full_ip` 改为**每格立即落盘 + 跳过已完成格 + 单格重试**，
  作图从磁盘回读。第一次全量跑正是因为沿用了 `main.py`"最后统一写盘"的结构，
  在 21/35 处崩溃后**丢掉了全部 21 格**。
- `results/alg4/_smoke/` 是验证"增量落盘 + 续跑"时留下的一格 scratch（未入 git）。
  **故意没有删**——见下一条：在这个挂载上删目录会把父目录读坏，而父目录正是本结果目录。
- **不要在这个挂载上删除目录**：删掉 `__pycache__` 之后，
  其**父目录**的 readdir 开始返回 EIO，反而把后续所有 import 打挂。
  用 `PYTHONDONTWRITEBYTECODE=1` 避免生成即可。
- 附带的复现性观察：一次误操作让**三个进程并发**跑了同一配置（同 seed），
  重复格之间各指标最大相对差 **4.3e-4** ——
  四位有效数字一致，与本项目"GPU 非 bit-exact"的既有口径吻合。
