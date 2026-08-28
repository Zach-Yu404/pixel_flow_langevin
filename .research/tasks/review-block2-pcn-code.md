# review-block2-pcn-code（2026-08-27）

## 【用户原始要求】（逐字）

> 你是独立代码审查方。Claude 已按共识实现 Block-2 自适应 pCN/OU refresh。请直接读源码与产物,独立完成审查,用中文把完整报告写入 /CBIG-Standard-ECE/Zach/MSFlow/PixelFlowICLR/Algorithm2/results/alg4_weighted_sigma_tau/codex_code_review.md(覆盖写,首行日期+模型名;发现按 blocker/should-fix/nit 分级,末尾明确"无 blocker"或列出 blocker)。
>
> ## 待审对象
> - 实现:/CBIG-Standard-ECE/Zach/MSFlow/PixelFlowICLR/Algorithm2/utils.py 中 run_posterior_sampling_alg4 的:①kwarg block2_refresh_mode(默认 independent);②frame-setup 的 λ 计算块(_lam/_sqrt_lam/_sqrt_fresh/_blk2_extra,precision 用闭式 trace:a=τe_k,b=(1−τ)s_k,Tr[HᵀH]/n=a²+(2ab+b²)/4,Tr[S⁻¹]/n=1/s² 或 inv_diag_mean());③Block-2 分支(independent 逐字 legacy;pCN:z_old=(x_τ−m)/σ 相对当次 x1,z_new=√λ z_old+√(1−λ)ξ0);④l.16 分支(independent legacy;pCN carry z_new);⑤_blk2_q0 每 stage 首帧记录;⑥row.update(_blk2_extra) 仅 pCN 模式。
> - 产物(同目录 results/alg4_weighted_sigma_tau/):claude_codex_consensus.md(含**事实纠错节**:Claude 之前给你的理论题面把 stage 边界写成均匀四分——错误;真实 σ stage 起点 = 1.0/0.857/0.667/0.4,你的条件表数值因此失效但单调性定理不受影响)、lambda_schedule.csv(按你要求的全列从运行时持久化,含谱 hash/floor dtype)、dense_invariance_verify.json、stationary_chain_verify.json。
> - 已有验证数字:independent 逐位复现哨兵 0.1447/0.1143/0.1291/0.1495(pooled,S_it=2,seed42);运行时 λ vs 持久化表 max 差 4.4e-9/4.9e-9(两模式);B:fp64 20 万样本 mean/cov/kurt/corr 全部 MC 噪声级;C:λ∈{0,.5,.9,.99} ρ/IACT/ESS 与理论吻合,λ=0.9999983 如实记录不收敛(恒等核病理)。冒烟:sigma_pcn pooled seed42 hole 0.1495→0.1047。
>
> ## 审查清单(规格 §9,逐项结论+证据)
> 1. z_old 是否相对更新后的 x1_new 计算(而非复用旧 x0)?
> 2. square-root 权重是否正确(√λ、√(1−λ);fresh scale=σ√(1−λ))?
> 3. random stream 是否公平(三模式 ξ0 抽样次数/顺序/形状一致;λ 计算无 RNG)?
> 4. baseline(independent)是否逐位不变(逐字 legacy 分支,含 l.16)?
> 5. scalar/spectral S 的 trace 是否正确(闭式 vs SpectralSOp.inv_diag_mean;rank(G)/n=1/4 的使用)?
> 6. diagnostics 是否只读、无 RNG side effect(_blk2_extra、row.update)?
> 7. 结论表述是否夸大"exact posterior"(检查 kwarg docstring 与共识文件表述)?
> 8. 另:precision_pcn 启用门(运行时 λ≡持久化表)是否满足你理论稿的 fail-closed 恢复条件?对照 lambda_schedule.csv 用真实边界抽查复算若干帧的 q/λ。
> 9. 任何其他问题(边界:τ=0 端点、σ=σ_min、与 diag_noise_off 组合、S_it=1 时 _z_new 定义)。

## 任务判定

- 类型：implementation 后独立 code review。
- `execution_allowed: false`：不修采样器，只读源码/产物、运行针对性无副作用验证，并覆盖写用户指定报告。
- 待审工作树基线：HEAD `d8abd9a1e2b4bb3bc5289e4ec6383efed44c4018`；开始审查时
  `utils.py` git blob `e5b0663a2175eb024c7fe3e5fb169a42336ddf13`、SHA-256
  `b0b6dc49fe7b48fc6ab504b0fa1a8ee2c3af348595d8a6b0287caa01011c2a30`。

## 当前状态

- 状态：done（review 完成，verdict=`request changes`；implementation 返回 working）。
- 双 agent preflight：`research-doctor --agents-only` 10/10 通过。
- 最终报告：`PixelFlowICLR/Algorithm2/results/alg4_weighted_sigma_tau/codex_code_review.md`。
- 报告 SHA-256：`9de412da5117611ba718e94e1d743096cfe443f746a56a614b403f07dbbda7c5`。
- 报告结论：independent/sigma_pcn 核心实现通过；blocker B1 = precision_pcn 未实现共识规定的
  fail-closed 运行门，修复并提交正式 code SHA 后须重审。
- 审查快照：`3d769e53cb1b47cf25972d4f44fb671c7394afa8`（无 branch，仅固定待审
  `utils.py` + 四个指定产物；不代替正式 implementation commit）。
- 结构化 review：`.research/bin/research-peer review --agent codex` 对上述快照连续三次在
  `.git/research-os/peer/worktrees` 遇到 Ceph `Remote I/O error`，未生成 artifact、未伪造。
- 独立验证：真实 scheduler/S 资产闭式复算 80 行；NumPy fp64 20 万×8 conditional MC；
  λ=0/.5/.9/.99 stationary chain；近恒等 4000 步未收敛反例。
