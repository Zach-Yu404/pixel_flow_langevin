# algorithm.4 — A clean-endpoint posterior sampler for cascaded flow priors

来源：`PixelFlowICLR/Algorithm2/results/algorithm.4pdf.pdf`（15 页，pdfTeX，
文件内 CreationDate 2026-08-21 14:12）
精读日期：2026-08-21（claude，全文 15 页逐页读完，非摘要）

**编号约定（重要，草稿本身有冲突）**：仓库文件名是 `algorithm.4`，本项目内按
**Algorithm 4** 称呼。但 PDF **内部**把自己的伪代码 listing 编号为 "Algorithm 1"，
且 §8.1 Table 2 的三列表头字面写成 `Alg. 1 | Alg. 2 | Alg. 1`（第三列应为本文的算法）。
Table 2 caption 自述："Alg. 1 applies a Langevin step to each of the two conditionals
…; Alg. 2 draws the image conditional exactly after a change of variables and retains
a Langevin step for the interpolant; Alg. 1 is the construction of this note."
→ **草稿排版缺陷，引用 Table 2 时必须自行消歧**，并建议向作者反馈。
与前一版 `results/modified.pdf`（"CLIMB-Flow"，含 Alg.1/2/3）是同一系列的第四篇。

## Problem

线性逆问题 `y = A x + η ε` 的后验采样 `p(x|y) ∝ p(y|x) p(x)`，先验是 cascaded
pixel-domain flow（PixelFlow）。困难：先验只能通过 learned velocity 在训练过的
噪声水平上访问，`p(x_1^k)` 本身既不可求值也不可微。

标准解法是 variable splitting。用在 flow 先验上有两条路，差别在于**保留联合密度的哪一个因子**：

- **路线 I（interpolant-coupled）** = 本项目现有 Alg.1/2/3。通过 flat-prior 高斯
  `N(x_τ; H_τ x_1, σ_τ² I)` 耦合。learned marginal `p_τ` 留在 x_τ 的条件里 →
  必须用 Langevin → **同时**带来 step size（h₀、Alg.1 还有 h₁）和先验被
  `√2 σ_τ H_τ⁻¹` 平滑。
- **路线 II（clean-endpoint）** = 本文。通过真实的 denoising conditional 耦合。
  `p_τ` 约掉 → step size 和平滑**一起消失**。

## Core method

### 因式分解 (5) 与 Prop. 1–2

    π(x_1, x_τ | y) ∝ p(y|x_1) · p(x_1|x_τ) · p_τ(x_τ)
                      likelihood  denoising    flow marginal

- **Prop. 1**：把 x_τ 边缘掉得到的正是 `p(y|x_1) p_1(x_1)`，即真后验。
  **augmentation 是精确的，没有引入平滑，也就不需要 annealing 参数去消除平滑。**
- **Prop. 2**：两个条件分布
    - (6) `π(x_1 | x_τ, y) ∝ p(y|x_1) p(x_1|x_τ)`
    - (7) `π(x_τ | x_1, y) = p(x_τ|x_1) = N(x_τ; H_τ x_1, σ_τ² I)`
  `p_τ` 在 (7) 中与 `p(x_1|x_τ)` 里隐含的 `p_τ` 相消。**这个相消只在 (5) 这个方向成立。**
- 代价（"obstruction relocated, not removed"）：`p(x_1|x_τ) ∝ p_1(x_1) N(x_τ;H_τx_1,σ_τ²I)`
  仍含不可得的 `p_1`。全部困难被压缩进这一个因子 → §4 处理。

### 唯一的近似 (8)

    p(x_1|x_τ) ≈ N(x_1; x̂_1, C),  x̂_1 ≈ E[X_1|x_τ],  C ≈ Cov(X_1|x_τ)

两个梯度 (9)(10) 因此都是 affine → §6 证明 affine ⟹ 可无步长精确抽样。

### §4 coupling covariance C —— 本文的核心贡献

- **Lemma 3 (11)**：设 τ>0 且 `p_1` log-concave，则对每个 x_τ
      `Cov(X_1|x_τ) ⪯ σ_τ² H_τ⁻²`   ⟺   `Cov⁻¹ ⪰ H_τᵀH_τ / σ_τ²`
  证明用 Brascamp–Lieb variance inequality（`∇²V ⪰ Λ ≻ 0 ⟹ Cov ⪯ Λ⁻¹`）。
  `p_1` flat 时取等。**这是模型自己提供的上界，不是调出来的。**
- 饱和取 `C = σ_τ²H_τ⁻²` 最保守，但 τ=0 处 `H_0 = s_k G` 秩亏 → 在 ker(G) 上奇异，
  **每个 stage 开头都要加 ridge**。（= 现有代码 `ridge_rel`/`power_iter_norm`/`epsilon` 的由来。）
- **(12) 本文采纳的选择**：

      C⁻¹ := H_τᵀH_τ / σ_τ²  +  S⁻¹        （S = 先验协方差 surrogate，Cov(X_1) ⪯ S）

- **Prop. 4** 三条性质：
  (a) `C⁻¹ ⪰ S⁻¹ ≻ 0` 对每个 τ∈[0,1]（含 τ=0）成立 → **任何地方都不需要 ridge**；
  (b) `C ⪯ σ_τ²H_τ⁻²`，在 Lemma 3 意义下 admissible，且严格紧于饱和取法；
  (c) 若 `p_1 = N(0,S)` 则 (12) **正是精确的条件协方差**。
  → (12) 不是为了 conditioning 硬加的 ridge，而是高斯先验下的精确解。

### S 的两种取法

- **(13) 有图像统计**：`S(ω) = E|x̂_1(ω)|²`，训练图在该 scale 的平均功率谱；
  G 若被 DFT 对角化则 H_τ、C⁻¹ 同时对角，`S^{-1/2}` 成为逐点乘。
  各向同性退化：`S = s² I`，s² = 该 scale 训练图的**逐像素方差**（测量得到，非搜索）。
- **(14) 无图像统计**：取 `s² = c²σ_τ²` ⟹ `C⁻¹ = [H_τᵀH_τ + c⁻² I] / σ_τ²`，
  即对 `H_τᵀH_τ` 加常数 ridge `c⁻²`，只剩一个无量纲常数 c。
  本文明说它 **τ-依赖是错的**（精确 ridge 应是 `σ_τ²/s²`，随 σ_τ 减小；c⁻² 是常数），
  误差由 (24) 界住。
- **沿 schedule 的分工**：τ=0 时第一项是 `s_k²G²/σ_τ²`，在 ker(G) 上为 0，
  **先验 surrogate 恰好在 interpolant 完全不约束的方向上提供全部精度**；
  τ→1 时第一项 ~ `e_k²/σ_τ²` 增长而 S⁻¹ 不变，surrogate 相对贡献单调下降。
  注意：**ridge 是大 σ_τ 处才需要，不是小 σ_τ 处**（退化来自 H_τ 而非 σ_τ）。

### §5 clean endpoint x̂_1

- **Prop. 5 (15)(16)**：`x_τ = H_τ x̄_1 + σ_τ x̄_0`、`d_τ = B_k x̄_1 - (e_k-s_k) x̄_0`。
  两个向量方程两个未知条件均值 → 可精确消元。**因为两个 stage boundary 共享同一个
  噪声实现**；也正因如此，把两端当作独立观测做 least-squares 是错的（残差共线，没有独立
  误差可平均）——这从理论上解释了历史上 WLS 估计不如 direct 估计。
- **Prop. 6 (17)**：`N_k x̄_1 = (e_k - s_k) x_τ + σ_τ d_τ`，其中
  `N_k = e_k(1-s_k)I - s_k(1-e_k)G = σ_τ B_k + (e_k-s_k) H_τ`（(3)，与 τ 无关，Appendix A.1 验证）。
- **Prop. 7 (18)(19)** —— 把 `v_θ = d_τ + ε`, `ε~N(0,γ²I)` 的误差正则化进去，**一次求解**：

      (18)  [N² + γ² H_τ²] x̂_0 = N (B x_τ - H_τ v_θ)
      (19)  [N² + γ² H_τ²] x̂_1 = N [(e_k - s_k) x_τ + σ_τ v_θ] + γ² H_τ x_τ      (τ>0)

  **两式算子完全相同**。正则项 `γ²H_τ²` 来自 `X_0 ~ N(0,I)`（模型构造上成立的先验），
  不是对图像的假设。极限：γ→0 回到 (17)；γ→∞ 得 `x̂_1 = H_τ⁻¹x_τ`，即
  interpolant-coupled sampler 用的中心 → **两条路线在网络无信息的极限下重合**，
  差别随网络提供的信息增长。
- **(20) γ² 的测量**：held-out 数据上两端已知，`γ² ≈ (1/n)E‖v_θ(x_τ,τ,k) - d_τ‖²`，
  按 (k,τ) 制表随模型存储。**是测量不是搜索。** 若误差有色，`γ²I → Σ_ε`，
  (18) 变为 `[N(H_τΣ_εH_τ)⁻¹N + I] x̂_0 = N(H_τΣ_εH_τ)⁻¹(B x_τ - H_τ v_θ)`。

### §6 两个 conditional 都精确抽样

- **Lemma 8**：`∇log π = b - Mx`（M 对称正定）⟹ `π = N(M⁻¹b, M⁻¹)`。
- **Lemma 9（randomize-then-optimize）**：`M = Σ_j R_jᵀR_j ≻ 0`，
  `ζ = Σ_j R_jᵀ ξ_j`（ξ_j 独立标准正态）⟹ `Mx = b + ζ` 的解服从 `N(M⁻¹b, M⁻¹)`。
  M、M⁻¹、M^{1/2} 都不用显式构造，CG 即可。
- **(21) Block 1 的算子与右端**：

      M_τ^den = AᵀA/η² + C⁻¹ = AᵀA/η² + H_τᵀH_τ/σ_τ² + S⁻¹
      b        = Aᵀy/η²  + C⁻¹ x̂_1

- **(22) Block 1 精确抽样**（三个因子 `R₁=A/η, R₂=H_τ/σ_τ, R₃=S^{-1/2}`）：

      M_τ^den x_1 = Aᵀy/η² + C⁻¹x̂_1 + [ Aᵀξ_y/η + H_τᵀξ_h/σ_τ + S^{-1/2}ξ_s ]

  `S = s²I` 时第三项就是 `ξ_s / s`，不需要开方。
- **(23) Block 2 精确抽样，无需求解**：

      x_τ ← H_τ x_1 + σ_τ ξ_0,   ξ_0 ~ N(0,I)

  即"按 schedule 自己的噪声水平重新加噪"。
- **§6.3 为什么不是 Langevin 的极限**：对协方差 Σ 的高斯目标，预条件 Langevin
  只有 h=2 时 drift 才完全走到均值，而那时注入噪声协方差是 2Σ；
  未预条件步在 (10) 上的平稳方差 `2σ_τ²/(2 - h/2σ_τ²)` 只有 h→0 才等于 σ_τ²（链不动）。
  **drift 要 h=2、noise 要 h=1，没有任何步长能复现精确抽样** ⟹ (22)-(23) 不是 Langevin 的特例。

### §7 Algorithm listing（本文的伪代码）

每次内迭代 = **一次网络前向 + 两次线性求解**，与被替代的采样器同价。
- 第一次求解 (19) 只含 G 的多项式：`G=I` 时退化为标量 `e_k-s_k`；G 被 DFT 对角化时
  是逐点除 + 两次变换。
- 第二次求解 (22) 混入 `AᵀA`，需要 CG。
- 时间网格之间携带的状态是 **x_0**（l.16 `x_0 ← (x_τ - H_τx_1)/σ_τ`），不是 x_τ：
  `X_0` 的分布在每个 τ 都是 `N(0,I)`，而 `X_τ` 的分布随 τ 变；
  另外 x_0 提供诊断 `‖x_0‖²/n ≈ 1`（x_τ 没有对应量）。
- Stage 之间：`x_1^{k+1} ← U⁽¹⁾(x_1^k)`，`x_0^{k+1} ~ N(0,I)`（ancestral transition）。

## §8 Assessment（作者自评，值得原样记住）

**精确的三件**：augmentation（Prop.1）、endpoint 消元（Prop.6，共享噪声使其是恒等式而非拟合）、
两个 conditional 的抽样（§6，给定 C 与 x̂_1）。
**不精确的一件**：(8) 用高斯代替非高斯的 denoising conditional。除非 C 恰好等于真协方差，
(6) 与 (7) 是**不同联合密度**的条件分布，sweep 不保持任何单一分布不变
→ **Algorithm 4 本身也是 heuristic**，与它取代的 PnP / split-Gibbs 同类；
区别是**没有自由参数**。

Table 2（列名已按上文消歧）：

| | Alg.1（双 Langevin） | Alg.2/3（image 精确 + interpolant Langevin） | **Alg.4（本文）** |
|---|---|---|---|
| 先验被 `√2 σ_τ H_τ⁻¹` 平滑 | yes | yes | **no** |
| 对 `p(x_1|x_τ)` 作高斯近似 | no | no | **yes** |
| step-size bias | O(h₁)+O(h₀) | O(h₀) | **none** |
| endpoint 估计中的 γ² | yes | yes | yes |
| 是否留下不变分布 | yes（相对上述目标） | yes（同左） | **no** |
| 需要搜索的连续量 | 2 | 1 | **0** |

左两列与右列**差别在种类而非程度**：平滑是显式的、可量化、随 schedule 消失，
但采样目标不是后验；高斯 surrogate 保持目标正确、改为扰动 kernel，误差一般不可量化，
但 §8.3 证明 σ_τ→0 时消失。两种误差**都在 cascade 末端最小**，而末端决定重建结果。

- **§8.2 S 的敏感性 —— 宁可取大**。`s²→∞`：S⁻¹ 消失，退化为饱和界，仍 admissible（有界）。
  `s²→0`：C→0，Block 1 坍缩到 x̂_1（无界）。
  **低估 S 会把 Block 1 收缩向去噪估计，重建指标反而变好而样本多样性丧失
  → 这个失败模式在最常报告的指标上看不见**，必须用 §8.6 的诊断。
  各向同性 `S=s²I` 在高频带高估先验功率（自然图像谱衰减），而 S⁻¹ 恰在该带起作用
  ——按上面的不对称性，这个方向是安全的。无统计时：像素值落在 [0,1] 则 Popoviciu
  给出逐像素方差 ≤ 1/4，`s²=1/4` 不假设任何图像分布。
- **§8.3 σ_τ→0 的行为**：`Cov(X_1|x_τ) = σ_τ²H_τ⁻²[I + O(σ_τ²∇²log p_1)]`。
  局部 log-convex 处界被突破，但只超出 O(σ_τ²)；多模态造成的突破是 O(1) 量级，
  但需要 x_τ 落在两个模之间，其概率随 σ_τ 减小而衰减。两种机制同时退去
  → (11) 渐近变紧。**反向警告：界被破坏意味着真条件比 (12) 更分散，采样器在后验多模处
  过度自信，而那正是最关心多样性的区域。**
- **§8.4 与 interpolant-coupled 的比较**：Alg.4 的 Block-1 中心是 x̂_1，
  **全权重承担 velocity error**；interpolant-coupled 中心是 x_τ（精确），网络只通过
  Langevin drift 进入。**所以 Alg.4 的优势随 γ 增大而减小，γ 足够大时反转。**
  crossover 未作解析刻画，但 γ 由 (20) 按 (k,τ) 制表，**可在部署前经验定位**，
  且同一个 cascade 的粗尺度与细尺度可能落在 crossover 两侧。
- **§8.5 与已有工作的关系**：变量分裂 + 深度先验、smoothed-prior 分析、
  Lemma 9 的 randomize-then-optimize、PnP data step 都是标准的。cascaded 设定特有的三点：
  ①耦合是 degradation `N(x_τ;H_τx_1,σ_τ²I)` 且 H_τ 在 τ=0 秩亏（不是 `N(x;z,ρ²I)`）
  ——这是精度成为算子、以及需要 Prop. 4(a) 的原因；②耦合宽度不是设计参数而是 schedule 自己的 σ_τ；
  ③两个 stage boundary 共享噪声实现，单端点 diffusion 没有对应物，是 Prop. 6 成为精确消元的原因。
  对 VP / EDM 模型，`H_τ → α_t I`，消元塌缩为 `x̂_0=(x_t-σ_t ε̂)/α_t` 无需求解，
  **算法退化为已知的 data-consistency step；全部内容都在 `H_τ ≠ αI` 这一情形。**
- **§8.6 应报告而非调节的三个量**：γ²（(20)，按 (k,τ)）；S（各 scale 从训练数据得到）；
  measurement residual `‖A_k x_1 - y‖/(η√m)`，应稳定在 1 附近。
  **另需至少一个 calibration 诊断**：固定 y 重复采样在 `ker(A_k)` 内的 spread。
  C 太小 → Block 1 坍缩到去噪器输出 → 多样性丧失，而这会**改善**重建指标，故指标查不出来。

## Relation to this project（关键落点）

1. **(18) 与 γ² 表已经实现且完全一致**：`utils.py:score_solve` 就是 (18)；
   `Algorithm2/gamma2_meas.json` 就是 (20) 的按 (k,τ) 表
   （实测 stage0 0.0070–0.0106、stage1 0.0106–0.0132、stage2 0.0115–0.0166、
   stage3 0.0094–0.1448）。`apply_B`/`apply_N`/`apply_H_tau` 与 (2)(3)(1) 一一对应。
2. **(19) 是新的、可直接实现**：与 `score_solve` 共用同一个 CG 算子，只换右端。
   现有取 x̂_1 的两条路都有问题——`apply_H_tau_inv(x_τ - σ_τ x̂_0)` 在 τ=0 处必须抛异常
   （`diag-h-tau-inv` 任务的记录），`direct_estimate_x1` 不含 γ² 正则。
   (19) 一次求解同时解决两者，且 τ>0 全程可用。
3. **经验调出来的 Tweedie anchor 就是 (12) 的 S⁻¹ 项。**
   `run_posterior_reg_sampling_alg2`（`anchor_P="identity"`）做的是
   `M += λI`、`b += λ·x1_model`，与 (12)(21) 对照即
   **`λ I = S⁻¹`，亦即 `s² = 1/λ`**。
   `results_reg_alg2` 扫出来的 λ=100 ⟹ s²=0.01；λ=25 ⟹ s²=0.04。
   → **本文说这个量应当被测量而不是被搜索**：取该 scale 训练图的逐像素方差（或 (13) 的功率谱）。
   本项目图像归一化到 **[-1,1]**（`demo_runner.py:169`），所以 Popoviciu 的
   assumption-free 上界是 `s² ≤ (2)²/4 = 1`（不是 PDF 写的 1/4，那是 [0,1] 的情形）
   ⟹ 对应 `λ ≥ 1`。扫描最优的 λ≈100 对应 s²=0.01，**远小于任何合理的图像方差**
   ——按 §8.2，这正是"低估 S / 过度收缩到 x̂_1"的方向，也正是"重建指标变好但多样性丧失"
   的失败模式。**这是一个可证伪的预测，必须用 §8.6 的 ker(A_k) spread 诊断去查。**
4. **但现有代码是"加"，本文是"换"。**
   - 现有：`b = Aᵀy/η² + H_τᵀ x_τ/σ_τ² + λ x̂_1`（interpolant 项 **加** anchor）
   - 本文 (22)：`b = Aᵀy/η² + (H_τᵀH_τ/σ_τ² + S⁻¹) x̂_1`（interpolant 项被 **替换**，
     整个 C⁻¹ 都作用在 x̂_1 上）
   工作区里未提交的 `data_rhs_matchx1` 算的是 `Aᵀy/η² + H_τᵀH_τ x̂_1/σ_τ²`
   —— **正好是 (22) 的 b 缺了 `S⁻¹x̂_1`**。也就是说 Alg.4 Block 1 的零件已在工作区，
   缺的是：(a) M 与 b 两侧的 `S⁻¹` 项；(b) 噪声项 `S^{-1/2}ξ_s`；
   (c) 用 (23) 的直接重加噪替换 Block 2 的 Langevin；(d) 用 (19) 算 x̂_1。
5. **Prop. 4(a) 让 τ=0 的 ridge 整套机制消失。**
   现有 `ridge_rel`、`power_iter_norm`、`epsilon`、`√ε·ξ_eps`、
   `cg_max_iter_l14=200` 的特例，全部只因 M 在 τ=0 奇异而存在；`S⁻¹ ≻ 0` 一进 M 就都不需要了。
   这与已记录的结论 "the τ=0 ridge is not what holds the hole at 1.0"（commit `5734af2`）
   **相互印证**：ridge 从来不是修复项，缺的是先验精度本身。
6. **h₀ 整个消失。** `results_reg_alg2/regularization_final` 的 48 格网格结论是
   "h₀ 是主导因子，1e-4/1e-3 全线不可用，h₀ 太小 → Block 2 几乎不动 x₀ → 状态冻住"。
   Alg.4 的 Block 2 是 (23) 的直接抽样，**没有 h₀**——那个病理按构造被消除。
7. **符号冲突（必须消歧）**：本文的 `S` = 先验协方差 surrogate；
   本项目代码与报告里的 `S` = 内迭代次数（`num_langevin`，草稿 l.8 的 S）。
   两者在同一段文字里同时出现会直接读错，后续 note/代码一律写 `S_prior` 与 `S_iter`。
8. **§8.4 的警告直接适用于我们**：Alg.4 让 x̂_1 全权重承担 velocity error。
   我们已有按 (k,τ) 的 γ² 表，**可以在实现前先经验定位 crossover**——
   注意 stage 3 末端 γ² 冲到 0.145，是全表最大值，恰好在"末端决定重建"的位置。

## Open questions

- **S_prior 尚未测量**：需要各 stage 分辨率上训练图的逐像素方差（各向同性）
  或平均功率谱 (13)。这是实现 Alg.4 的前置数据，且是"测量"不是"调参"。
- **(13) 的 DFT 对角化对我们不成立**：本项目 `G = nearest_up ∘ bilinear_down`，
  已验证 `G²=G` 且自伴（即正交投影），**但不是 circulant**，DFT 不对角化它。
  好消息：正交投影使 `H_τ` 只有两个特征值（`apply_H_tau_inv` 已经利用了这点），
  所以 `S=s²I` 时 `C⁻¹` 是 G 的多项式，**有闭式逆，这一半不需要 CG**。
  谱形式的 S(ω) 则需要另行处理。
- **(19) 是否真优于 `direct_estimate_x1`**：现有 `test/score_x1hat.py` +
  `plot_x1hat_curves.py` 的框架可以直接量。
- **Table 2 的列名 bug** 是否已知——建议向草稿作者反馈。
- 本文只给单张图的构造，**cascade 之间 S 随 scale 怎么变**（(13) 说"at scale k"）
  需要按 4 个 stage 各测一次。

> 之后引用此文先读本 note；仅在需要验证细节时回原文
> `PixelFlowICLR/Algorithm2/results/algorithm.4pdf.pdf`。
