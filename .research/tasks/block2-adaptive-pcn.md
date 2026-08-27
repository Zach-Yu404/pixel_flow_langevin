# block2-adaptive-pcn（2026-08-27，implementation review 阶段）

结果目录：**results/alg4_weighted_sigma_tau/**（用户修订，替代规格 §10 的
alg4_block2_adaptive_pcn；根目录必须有六臂同帧总对比 trajectory.png）。

## 用户原始要求（两条，逐字）

### 第一条（实验规格本体，经用户补发恢复）
「请在当前 Algorithm 4 上实现并测试一个 **自适应 pCN/OU Block-2 refresh**。目标是在
**不修改物理 σ_τ、不缩小目标 conditional covariance** 的前提下，减少 stage 2/3 后半段
每次迭代新加入的独立 ξ_0。只测试 image=junco, task=box_inpainting, S_it=2。
本任务必须由 Claude Code 和 Codex 在实现前共同推导、实现后交叉 review，并形成中文共识记录。
§1 理论目标：原 Block 2 独立重抽 x_τ=H x1+σ ξ0；不要直接 σ→wσ（改变 p(x_τ|x1)）。
改为对标准化 residual 用 pCN/OU kernel：m=H x1_new，z_old=(x_τ_old−m)/σ，
z_new=√λ z_old+√(1−λ) ξ0，x_τ_new=m+σ z_new。λ=0 即当前算法；对固定 x1_new 必须保持
N(H x1_new, σ²I) invariant（是不变 Markov kernel，非独立 conditional draw）。
§2 三模式 block2_refresh_mode：independent（λ=0，逐位复现 baseline）；
sigma_pcn（主候选）：σ_{k,0}=1−s_k，λ=clip[1−(σ_τ/σ_{k,0})², 0,1]
⟹ fresh scale σ√(1−λ)=σ²/σ_{k,0}，O(σ)→O(σ²)；
precision_pcn（次要对照）：q_{k,τ}=(1/n)Tr[HᵀH/σ²+S⁻¹]，q_{k,0}=stage 首个有效 τ 值，
λ=1−min(1, q_{k,0}/q_{k,τ})。实现前 Codex 独立检查 q 单调性、scalar/spectral trace
正确性、异常 schedule；不满足则保留诊断并停用，不得静默换公式。
λ 只准依赖 stage,tau,H,sigma_tau,S；禁止依赖图像/MSE/resid/v_θ/GT（否则需 MH 修正）。
§3 Line 16：x0←z_new 为主路径（避免小 σ 消减误差）；保留 residual 恒等
‖z_new−(x_τ_new−Hx1_new)/σ‖ 作为 correctness check。
§4 实验隔离：只改 Block 2；(19)/(22)/γ²表/S_it=2/scheduler/sigma_min/CG tol/
measurement/mask/seeds/terminal replacement 全部不动；不并用 antithetic/endpoint prior
shrink/early stop/tempering/随机项缩放/新 ridge/新 λ 搜索；block1_system_scaling=none。
§5 六臂：{independent,sigma_pcn,precision_pcn}×{pooled_junco,spectral}，≥4 seeds，
所有 arms 共享 Block-1 随机序列与对应 ξ0 序列。
§6 验证：A baseline 逐位；B dense float64 conditional invariance（E[x_τ]≈Hx1、
Cov≈σ²I、z_new~N(0,I)）；C stationary chain（非平稳初始化多步收敛 + 各 λ 的
autocorrelation/ESS）；D 状态审计（z_old 必须基于 x_τ_old−H x1_new，不得复用旧 x0）。
§7 逐 frame/inner 保存 frame/stage/step/inner/tau/sigma_tau/lambda_refresh/
fresh_noise_scale(=σ√(1−λ))/x1_in/x1_out/x_tau_in/z_old/xi0_fresh/z_new/x_tau_out；
并记录 hole/obs MSE、x̂₁ MSE、endpoint removal、Block-1 injection、‖z_old‖²/n、
‖z_new‖²/n、‖x0‖²/n、residual identity error、低/高频误差、meas resid、spread、
autocorr、ESS、NFE/CG/wall-clock。重点 frame 18–39、stage 2 后半、stage 3 全段；
生成完整 trajectory montage。
§8 十个机制问题：1 sigma_pcn 是否减少 ξ0 写入下一轮 x̂₁；2 stage2/3 frozen-center
contamination 是否减轻；3 改善是否来自更少 finite-step fresh-noise jump 而非目标
covariance 缩小；4 ‖x0‖²/n 是否仍≈1；5 marginal spread 保持但 autocorr 是否升高；
6 ESS 降多少；7 precision_pcn 是否稳定优于 sigma_pcn；8 pooled/spectral 结论是否一致；
9 是否只改善单条轨迹却牺牲 mixing；10 推荐保留哪种。报告必须区分：独立 exact
conditional draw / conditional invariant 的相关 pCN-OU kernel / 整体 Gaussian
surrogate 仍非原始 PixelFlow posterior 的严格 exact sampler。
§9 协作：实现前双方独立推导（invariance、λ 定义、为何不能缩放物理 σ、为何 λ 不能
state-dependent、l.16 carry z_new 坐标一致性），共识后实现；实现后 Codex 独立
review（z_old 基准、平方根权重、random stream 公平、baseline 逐位、trace 正确、
diagnostics 只读无 RNG 副作用、结论不夸大 exact posterior）；有问题修复后重验。
§10 结果文件清单 + 每 arm 7 张图 + 逐 frame 写盘断点续跑 + 最终回复 7 项。」

### 第二条（输出目录修订，逐字要点）
结果目录统一改为 results/alg4_weighted_sigma_tau/；根目录必须有
trajectory.png——同 frame/同图像范围/同归一化横向比较六臂，至少展示
stage1→2 转换前后、frame 18–39、stage2 后半、stage2→3 转换、stage3 全程、
final reconstruction；每子图标注 mode/S mode/frame/stage/tau/sigma_tau/
lambda_refresh/hole MSE；arm 子目录可存独立 trajectory 但根目录总图必须在；
文件清单同 §10（目录名替换）；逐 frame 写盘断点续跑；最终回复必须确认
trajectory: results/alg4_weighted_sigma_tau/trajectory.png。

## 进度
- [x] 规格恢复（第一条曾因进程重启丢失，用户补发；transcript 检索记录见对话）
- [x] 结果目录骨架
- [x] Claude 独立推导 → Codex 独立推导 → 共识 → 首版实现 → 验证 A–D
- [x] Codex code review：`request changes`；independent/sigma core 通过，precision fail-closed
      启用门缺失为 blocker B1（完整报告 `results/alg4_weighted_sigma_tau/codex_code_review.md`）
- [x] B1+S1-S5+N1/N2 修复（含首 RNG 前全表预检、门表 kwarg、S_it/σ_min 校验、
      SOperator trace 契约、schedule v2 全精度/溯源列）→ codex 复审→残项修复→
      终审：技术残项全部销项；request changes 仅余 SHA 绑定（用户裁决）与
      重跑收据（已补：configs/precision_recheck_receipt.csv，8/8 四位一致、
      gate 40/40 pass ≤8.5e-10）
- [x] 六臂 4 seeds 完成 + trajectory.png + report.md + 共同解释（十问收窄版）

## 结果（详见 results/alg4_weighted_sigma_tau/report.md）
hole mean4±std4（spread）：independent pooled 0.1499±0.0007(0.2646)/
spectral 0.1400±0.0052(0.2469)；sigma_pcn 0.1250±0.0146(0.2341)/
0.1297±0.0100(0.2436)；precision_pcn 0.1337±0.0014(0.2388)/
0.1376±0.0070(0.2469)。收益全在 stage 3（sigma pooled f29→f39 −0.0044，
independent +0.0239）；stage1-2 sigma 略差（早期 fresh noise 有益混合）；
x0_rms²≈1、恒等 ≤1.5e-4、CG 全收敛、NFE 不变；ESS 末段大降（f38
0.0032/0.0112，f39 ~e-7）、pooled spread −11.5%（diversity trade-off 成分
无法排除）。共识推荐：默认 independent 不变，pCN 留 opt-in
（sigma=均值最优/种子彩票，precision=pooled 稳健 −11%）；启用与否用户裁决。
不得表述为"更好的 posterior 采样"。

## 实现事实（双方推导的共同出发点）
- H_τ = (1−τ)·s_k·G + τ·e_k·I，G 正交投影（G²=G=Gᵀ），rank(G)/n = 1/4
- σ_τ = (1−τ)(1−s_k) + τ(1−e_k)（线性递减；σ_τ(0)=1−s_k=σ_{k,0}）
- 真实 stage 边界：s_k=(0,1/7,1/3,3/5)，e_k=(1/4,1/2,3/4,1)；只有 start_t
  被 scheduler rectify，σ stage 起点=(1,6/7,2/3,2/5)
- Tr[HᵀH]/n 闭式：a=τe_k, b=(1−τ)s_k ⟹ a² + (2ab+b²)/4（精确、确定性）
- Tr[S⁻¹]/n：scalar = 1/s²(τ)；spectral = mean(1/P) = SpectralSOp.inv_diag_mean()
- Block-2 现行代码：x_τ = H x1 + σ ξ0（每 inner）；l.16 帧末 x0=(x_τ−Hx1)/σ

## 2026-08-27 Codex implementation review

- 精确对象：base HEAD `d8abd9a1e2b4bb3bc5289e4ec6383efed44c4018`；工作树
  `utils.py` blob `e5b0663a2175eb024c7fe3e5fb169a42336ddf13` / SHA-256
  `b0b6dc49fe7b48fc6ab504b0fa1a8ee2c3af348595d8a6b0287caa01011c2a30`。
- Verdict：`request changes`。核心 pCN、legacy/RNG/trace/diagnostic 检查均通过；
  `precision_pcn` 没有共识 §94 的 runtime-vs-persisted fail-closed gate，CSV 亦缺执行该门所需
  的全精度/provenance/status 列，故 precision arm 必须保持停用。
- 其余 should-fix：mode 枚举 fail-fast；共识 e_k 事实更正；未收敛 stationary rho 字段；
  S_it/sigma_min 输入边界；泛型 SOperator trace 契约。完整分级与复算表见用户指定报告。
