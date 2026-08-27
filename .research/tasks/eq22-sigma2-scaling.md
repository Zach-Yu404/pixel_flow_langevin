# eq22-sigma2-scaling（2026-08-27，done——双 agent 共识闭合）

## 用户原始要求（逐字）
「和Codex 协作要求

本任务必须由 **Claude Code 与 Codex 共同讨论、交叉审查并形成共识**，不能由 Claude 单独实现后直接宣布结论。

### 1. 实现前先讨论理论
在修改代码前，先把以下内容交给 Codex 独立推导和审查：
1. 将 Eq. (22) 两边整体乘 σ_τ² 后，新的 M̃_τ、b̃_τ、ζ̃_τ 是否全部正确；
2. 是否严格满足 M̃_τ=σ_τ²M_τ, b̃_τ+ζ̃_τ=σ_τ²(b_τ+ζ_τ)；
3. scaled system 的输出均值和 covariance 是否仍分别为 M_τ⁻¹b_τ、M_τ⁻¹；
4. 随机 RHS 应该使用 σ_τ²ζ_τ 而不是针对 M̃_τ 重新构造一组会改变输出 covariance 的噪声；
5. 整体 scalar scaling 是否只改变数值动态范围，而不改变 condition number；
6. PCG stopping criterion、relative residual、Jacobi/spectral preconditioner 是否需要同步缩放。
Claude 和 Codex 必须先形成明确共识，再开始实现。若有分歧，记录双方推导、证据和最终裁决，不能静默忽略。

### 2. 实现后由 Codex 独立 review
operator/RHS 是否完整乘上 σ_τ²；是否存在漏乘或多乘；random seed 和三组 ξ 是否与
baseline 完全一致；spectral SOperator 和 preconditioner 是否正确适配；baseline 路径
是否逐位保持不变；dense float64 与 Monte Carlo 验证是否足以支持等价性；trajectory
差异是否来自浮点数、solver tolerance 或随机流，而不是算法目标改变。
Codex review 发现的问题必须先修复并重新验证，再运行正式实验。

### 3. 结果也要共同解释
scaled Eq. (22) 是否严格保持原 Gaussian draw；是否真的改善 float32 数值稳定性；
CG iteration 或 residual 的变化来自哪里；trajectory 差异是否超过合理数值误差；
是否值得将 sigma2 设为默认实现；是否只能称为 numerical rescaling，而不能称为解决了
stage 2/3 的 frozen-center 问题。不得因为指标偶然改善就声称算法得到改进；若理论上
等价但实际结果明显不同，必须继续定位实现或数值原因。

### 4. 保存协作记录
结果目录保存 codex_theory_review.md / codex_code_review.md / claude_codex_consensus.md /
unresolved_questions.md，中文，含双方最初判断、关键公式审查、发现的实现问题、修改后
的验证结果、最终共识和仍未解决的问题。
若 Codex 因额度或环境原因暂时不可用，Claude 可以进入单 Agent 降级模式继续，但必须
标记 PENDING_CODEX_REVIEW，并且不能声称已经获得双 Agent 共识。Codex 恢复后必须基于
保存的 canonical state 完成补充 review。」

## 进度（全部完成）
- [x] Claude 独立初判（先于 Codex 回复落盘，7 条）→ Codex 独立理论审查
      （gpt-5.6-sol/ultra，170k tokens）→ **6/6 一致零分歧**（含 floor=c×1e-12
      缩放律、Cov=M̃ 重建噪声放大 1e16 的反例）
- [x] 实现：utils.py 单开关 eq22_sigma2_scale（默认逐位 baseline）：pcg_solve
      clamp_floor 参数化、make_M_tau_den_sigma2 逐项构造（σ>0 校验）、双分支
      frame-setup/Block-1、spectral 探针/Jacobi floor 同缩
- [x] 验证一轮：档案逐位复现（0.1339±0.0047/0.1382±0.0054 连 spread 同）、
      8 检查点恒等式 fp64 1e-16、PCG 逐步等价、缩放开 4-seed 四位全同
- [x] Codex 代码 review：无 blocker；1 should-fix（guard-fail 回退 None 破坏
      缩放律）→ 修复为 K̃=I/c（1D 逐位等）；2 nit（builder 移入 else、顺序）
- [x] 验证二轮销项全部缺口：[F]fallback 单测 [A2]修复后回归 [G]12 帧含谱臂
      fp64 与 f39 [S8]σ=1e-8（4.6e16→4.6 零 NaN）[DN]稠密 fp64（M̃−cM 5e-17、
      4000 直接解 9e-15、MC 协方差 = Cov(ζ)=M 得证）[T]tol 扫描（截断归因
      坐实）[BL]blur [CB]diag_noise_off 组合 [H]hash 归档
- [x] §3 共同解释：Codex 47 行最终意见与 Claude 六点**无实质分歧**

## 最终共识（详见 results/alg4_eq22_sigma2_scaling/claude_codex_consensus.md）
定性 = numerical rescaling（代数等价方程归一化）：精确算术逐实现恒等；实证
收益仅绝对动态范围安全裕量（f39 2.9e7→4.6，σ=1e-8 4.6e16→4.6）；基线本就
零溢出全收敛 ⟹ **不改默认，保留 opt-in 防御开关**；不解决 frozen-center。
非阻塞遗留见 unresolved_questions.md（batch>1/融合 S 操作/相对 breakdown
判据等）。附带发现：本机 GPU 栈同种子跨运行本就非逐位（hash 不同），逐位
基准只在指标层有效。产物：4 协作记录 + validation{,2}.md + 2 个可复跑脚本，
全在 results/alg4_eq22_sigma2_scaling/（<0.2MB）。

## 【Claude｜方案】（实现草案，待共识后执行）
- utils.py 单开关 `eq22_sigma2_scale=False`（默认 False ⟹ 逐位 baseline）。
- 打开时 Block-1 逐项重构（关键：**永不形成 1/σ² 中间量**，事后乘 σ²·M(v) 拿不到
  fp32 收益）：M̃v=(σ²/η²)AᵀAv+HᵀHv+σ²S⁻¹v；b̃=(σ²/η²)Aᵀy+HᵀHx̂₁+σ²S⁻¹x̂₁；
  ζ̃=(σ²/η)Aᵀξ_y+σHᵀξ_h+σ²S^{-1/2}ξ_s——同一组 ξ、同一 RNG 顺序。
- 预条件：对角探针在缩放算子上构造（Jacobi 与 spectral 探针同理，d̃=σ²d）。
- 验证：①默认关回归 4 帧逐位（0.1447/0.1143/0.1291/0.1495）；②小规模 dense
  float64 恒等（解与未缩放 dense 解差 ~1e-12；协方差恒等式）；③同种子 fp32
  MC 对照 max|Δ|（预期舍入级）；④双臂 4 seeds 正式对比（hole MSE、CG iters/resid、
  inf/nan 计数）；⑤晚期中间量幅值探针（baseline 的 1e16 级 vs 缩放后 O(1)）。
- 结果目录：results/alg4_eq22_sigma2_scaling/。
