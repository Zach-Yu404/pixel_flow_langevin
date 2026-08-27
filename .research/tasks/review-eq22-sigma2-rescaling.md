# review-eq22-sigma2-rescaling

state: review
type: review
owner: codex
issue: null
pr: null
created: 2026-08-26
execution_allowed: false

## 【用户原始要求】

> 你是独立理论审查方。请只依据下面给出的实现事实,独立推导并回答 6 个问题。给出推导或反例,不要只给结论。请用中文回答,输出为可直接存档的 markdown。
>
> ## 背景与实现事实(PixelFlow Algorithm 4, Block 1, utils.py 现状)
>
> 对每个内循环, Block 1 从 π(x1 | x_tau, y) 精确抽样,通过求解线性系统 (Eq. 22):
>
> - 算子: M v = (1/η²)·AᵀA v + C⁻¹ v, 其中 C⁻¹ v = (1/σ_τ²)·H_τᵀH_τ v + S⁻¹ v
> - 确定 RHS: b = (1/η²)·Aᵀy + C⁻¹ x̂₁ (端点中心)
> - 随机 RHS (Lemma 9 RTO): ζ = (1/η)·Aᵀξ_y + (1/σ_τ)·H_τᵀ ξ_h + S^{-1/2} ξ_s, 其中 ξ_y, ξ_h, ξ_s ~ N(0, I) 独立。Cov(ζ) = M, 因此解 x = M⁻¹(b+ζ) 是 N(M⁻¹b, M⁻¹) 的精确 draw。
> - S 有两种实现: 标量 s²(τ)·I,或 SpectralSOp(FFT 对角,apply_S_inv 除以功率谱 P(ω),apply_S_inv_sqrt 除以 sqrt(P))。
> - 求解器 pcg_solve(fp32, GPU),停机准则是相对残差 max_batch(||r||₂/||b||₂) < cg_tol=1e-5,max_iter=300。实现细节(逐字):
>   - `b_norm = dot(b,b).sqrt().clamp(min=1e-12)`
>   - `alpha = rz / dot(p, Ap).clamp(min=1e-12)`
>   - `beta 分母 rz.clamp(min=1e-12)`
>   - Jacobi 预条件: 对角估计 floor=1e-12,M_inv = 1/diag
>   - spectral 臂预条件探针: d = M(1) − S⁻¹(1) + mean(1/P)·1, M_inv = 1/d
> - 全程 float32。σ_τ 的调度最低到 sigma_min = 1e-8,即 1/σ_τ² 最大 1e16。η=0.05。
> - 基线路径的随机数顺序: 每个内循环依次抽 ξ_y, ξ_h, ξ_s(CPU 生成器),Block 2 再抽 ξ_0。
>
> ## 提议(待你独立审查)
>
> 把 Eq. (22) 两边整体乘 σ_τ²,得到缩放系统 M̃ x = b̃ + ζ̃,希望改善 fp32 数值动态范围。
>
> ## 请独立推导并回答:
>
> 1. 新的 M̃、b̃、ζ̃ 应当是什么形式(逐项写出)?是否全部正确可行?
> 2. 是否严格满足 M̃ = σ_τ² M 以及 b̃ + ζ̃ = σ_τ²(b + ζ)?在什么条件下成立(例如 ξ 实现是否必须相同)?
> 3. 缩放系统的解的均值与协方差是否仍分别为 M⁻¹b 与 M⁻¹?给出推导。
> 4. 随机 RHS 应该使用 σ_τ²·ζ(同一组 ξ 缩放),还是可以针对 M̃ 重新构造一组 Cov = M̃ 的 RTO 噪声?两者的输出协方差各是什么?
> 5. 整体标量缩放是否只改变数值动态范围而不改变条件数?fp32 下它能改变什么、不能改变什么(考虑三项的相对量级、加法吸收、reduction 溢出、乘法顺序:逐项重构 vs 事后乘 σ²)?
> 6. 上述 pcg_solve 的停机准则、相对残差、三处绝对 clamp(1e-12)、Jacobi floor、spectral 预条件探针,在缩放系统下是否需要同步修改?哪些量在精确算术下缩放不变,哪些在 fp32/clamp 下不是?给出晚期(σ_τ=1e-8, 1/σ²=1e16)的具体数值量级分析。
>
> 另外请回答: 这个提议在精确算术下是否改变算法的任何统计性质?它有资格被称为什么(算法改进/数值重缩放/其他)?你预期它对采样质量指标(hole MSE 等)的影响是什么?

## Scope 与证据边界

- 只做独立理论审查，不修改 `utils.py` 或任何采样代码，故 `execution_allowed: false`。
- 数学与数值分析只使用上述【用户原始要求】中列出的实现事实；Research OS 状态检查不作为技术结论证据。
- 输出为中文、可直接存档的 Markdown；六问均给推导、条件或反例。

## 【Codex｜Review】结果

完整归档稿：references/2026-08-26-codex-eq22-sigma2-rescaling-review.md。

核心结论：

1. 正确缩放是 \(\widetilde M=cM,\widetilde b=cb,\widetilde\zeta=c\zeta\)，
   \(c=\sigma_\tau^2\)。
2. \(\operatorname{Cov}(\widetilde\zeta)=c^2M=c\widetilde M\)，不是
   \(\widetilde M\)。误按后者构造噪声会使输出协方差成为
   \(c^{-1}M^{-1}\)；在 \(\sigma=10^{-8}\) 时放大 \(10^{16}\) 倍。
3. 精确算术下样本逐路径相同、条件数不变；它是数值重缩放，不是统计或采样算法改进。
4. fp32 下可缓解大中间量与 reduction 溢出，但不能修复相对加法吸收、零空间病态或条件数。
5. cg_tol 与相对残差无需缩放；三个绝对 clamp、Jacobi floor、spectral 探针必须按同一尺度
   重新审计。自然 Jacobi 尺度下，原 1e-12 对应 \(c\times10^{-12}\)，晚期为
   \(10^{-28}\)；更稳健的是相对 breakdown 判据与稳定 reduction。

## 结构化 peer 状态

research-doctor --agents-only 为 10/10 通过。随后按协议三次调用
.research/bin/research-peer plan --agent codex，均在创建隔离 worktree 时被本仓库 Ceph
挂载的 Remote I/O error 中断，未生成 plan/compare artifact。故本任务保留在 review
并标记 pending_review_by: claude；这不改变用户明确要求的 Codex 独立理论审查结果。
