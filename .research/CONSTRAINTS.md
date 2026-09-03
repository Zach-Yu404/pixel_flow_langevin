# 用户约束（Authoritative）

> 规则：用户约束一出现立即写入本文件（附原始表达原文），并同步 STATE.yaml 相应开关。
> 任何 agent 不得基于旧约束继续工作。

## 长期约束

以下来自项目文档中的明确禁令（来源标注；非用户口头约束，但同等强制）：

### 复现公平性（来源：`github_project_local/LOCAL_REPRODUCTION.md`）
- **不得改 `batch_size`（=8）** 做跨 config 对比——seed 流位置靠它对齐
- seed 契约 all-seeds-42 不得破坏

### MRI（来源：`github_project_local/docs/mri_normalization_notes.md`）
- **MRI 数据已归一化，不得再归一化**

### 对比展示（来源：`github_project_local/README.md`）
- PSLD/ReSample 是 FFHQ 先验跑 ImageNet（OOD），**不得 naive 对比展示**
- DAPS unconditional vs 我们 class-conditional+CFG 的 prior 不匹配必须声明

### 仓库纪律（来源：research-init 共享规则 + 项目布局）
- 大文件（.pt/.ckpt/dataset）不进 git；绝对路径只进 `local.yaml`（gitignored）
- `github_project_local/` 是独立嵌套 repo，**不得** add 进外层 repo
- baseline 代码不 vendor

### 环境（来源：集群事实）
- 登录节点无 GPU；跑采样/训练必须走 Slurm

### 度量与引用(来源:rerun_imageNet/METRIC_AUDIT.md、celeba METRIC_AUDIT.md、baseline_audit_report.md)
- 跨论文 LPIPS 必须同变体(默认官方 AlexNet);**piq replace_pooling=True 的数字永不对外引用**
- FID:N=100 永不当绝对数;只信 pooled N=500 且仅作 ranking
- 永不引用:修复前 demo inpainting(~35 dB)、DDNM-gaussian 6.21、减步 baseline 伪影、MAX=1 时代的 PSNR
- DAPS 对其自身 torchvision GT 打分;两套表(piq-VGG vs alex)不得混

### 采样器纪律(来源:DESIGN.md、sampler_diff、debug_IP4)
- 永不 fork `run_posterior_sampling` / 算子——import 之(fork 即失去对齐保证)
- warm_restart=True 不关;λ_prox 与 λ_reg 同步动
- (2026-08-24 用户决定)apply_G 的 stage-3 恒等旁路已删除:G 在所有 stage 都是 down-up 投影;
  g_bypass_stage3 开关成为无效 no-op,gamma2 表按新口径重测(junco 单图)
- (2026-08-24 用户决定)gamma2 表分表:gamma2_meas.json = junco 单图/eps_for(Alg2/config.json 专用);
  gamma2_meas_alg4.json = Alg4 专用(config_alg4.json 指向它,stage-3 为 7 图/随机噪声口径)。
  两个会话不得写对方的表——此前 stage-3 行曾被两会话互相覆写
- config 不跨任务迁移(ns/tr/fd/srs 全是任务相关;motion 伴随必须 flip(K))
- inpainting tr=1;blur/SR tr=0(SR@tr=1 = bicubic,不是算法结果)
- MMSE 类实验:y 跨 sampler seed 固定(测量后、采样前重播种)

### 运维(来源:SESSION_HANDOFF*、INCIDENT_NOTES、random_memory)
- baseline runner 用方法专属 OUT env var(DDNM_OUT 等),通用 OUT_ROOT 会静默 clobber 别人的结果目录
- 永不 `cp -al` 硬链接实验数据目录(2026-05-03 事故:smoke 测试穿透覆写源数据)
- 可复现随机性禁用 `hash()`;bash 禁用变量名 `GROUPS`
- DAPS 采样不包 no_grad;piq 前 clip(0,1)
- celeba_results/ 根会被外部进程重置,持久物只放 code/ 或数据盘符号链接

## 当前任务约束

### 2026-08-27 · block2-pcn-final-review

【用户原始要求】
> Block-2 pCN 最终轮:①残项销项确认;②结果共同解释。请把两部分都追加写入 /CBIG-Standard-ECE/Zach/MSFlow/PixelFlowICLR/Algorithm2/results/alg4_weighted_sigma_tau/codex_code_review.md 末尾新节「## 终审与共同解释(2026-08-27)」,给最终 verdict(approve / request changes),≤80 行。
>
> ①残项修复(读码/读文件核验):
> - B1 残:utils.py 现已在**首个 RNG 抽样前**做全表预检(precision_pcn 分支内 RNG-free 重放整个 per-stage schedule,逐帧闭式 λ vs 表,tol 1e-7,失败即 raise);configs/pcn_verify.py 已换成当前版(XCHK 传 block2_lambda_table,可复现);lambda_schedule.csv dtype 溯源列改为 float32(numpy mean of reciprocal)/python-float(1/s2)。
> - S3 残:stationary JSON 未收敛行 ess_fraction=null;dense_invariance_verify.json 已加 _meta(N/dim/seed/dtype/estimator/script)。
> - N1 残:diag_noise_off 注释已改。
> - 验证重跑:两模式 XCHK 仍 0.00e+00;precision 正式 4-seed run 在新预检下全部通过。
>
> ②六臂结果(summary_metrics.csv;hole mean4±std4 / spread):
> independent: pooled 0.1499±0.0007/0.2646, spectral 0.1400±0.0052/0.2469
> sigma_pcn:   pooled 0.1250±0.0146/0.2341, spectral 0.1297±0.0100/0.2436
> precision:   pooled 0.1337±0.0014/0.2388, spectral 0.1376±0.0070/0.2469
> 机制诊断(inner/frame CSV 聚合,late=f30-39):x0_rms²≈1.00-1.04;resid 恒等 ≤1.5e-4(fp32);Block-1 injection 各模式几乎不变(pooled ~+0.032/inner);corr(z_new,z_old) late:0 / 0.75 / 0.55(pooled);fresh_scale f38:0.045 / 0.005 / 0.009;stage-end holes f19/f29/f39:independent/pooled 0.1101/0.1260/0.1499,sigma_pcn/pooled 0.1211/0.1295/0.1250(早中期更差、stage3 大幅better),precision/pooled 0.1098/0.1168/0.1337。
>
> 请就规格十个机制问题给出你的判断(与我共同解释;不一致处标明),特别是:改善是否为有限步非平衡效应而非 covariance 缩小(spread 数据:sigma_pcn pooled −11.5%,如何解读?);sigma_pcn 高种子方差(±0.0146)的定性;precision_pcn 低方差稳定 −11% 的定性;推荐保留哪种模式(或 needs-user);结论措辞红线(不得称更好的 posterior 采样)。

生效动作：任务级 `execution_allowed: false`；只读核验源码/产物，不改采样器或追加 GPU
实验；仅追加指定报告节及 Research OS 记录。终审节不超过 80 行，且不得称“更好的 posterior 采样”。

### 2026-08-27 · review-block2-pcn-code

【用户原始要求】
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

生效动作：任务级 `execution_allowed: false`；只读源码与指定产物，可运行无状态副作用的
针对性验证；只覆盖用户指定的审查报告，不修改采样器实现。

### 2026-08-27 · review-block2-pcn-theory

【用户原始要求】
> 你是独立理论审查方。用户要在 PixelFlow Algorithm 4 上实现"自适应 pCN/OU Block-2 refresh"。请只依据下面的题面事实独立推导,不要看 Claude 的推导。用中文,输出存档级 markdown,并将完整稿写入 /CBIG-Standard-ECE/Zach/MSFlow/.research/references/2026-08-27-codex-block2-pcn-theory.md(你有文件系统权限;同时正常 stdout 输出)。

完整逐字题面与 8 项问题见 `tasks/review-block2-pcn-theory.md`。

生效动作：任务级 `execution_allowed: false`；不读取 Claude 推导，不修改采样器；仅作独立
理论与数值审查。完整中文稿必须写入用户指定 reference，并将同一稿正常输出到 stdout。

### 2026-08-26 · review-eq22-sigma2-rescaling

【用户原始要求】
> 你是独立理论审查方。请只依据下面给出的实现事实,独立推导并回答 6 个问题。给出推导或反例,不要只给结论。请用中文回答,输出为可直接存档的 markdown。

生效动作：只做理论 review，不修改实现；`execution_allowed: false`。技术推导的证据边界限于
用户题面所列实现事实；最终输出为中文、可直接存档的 Markdown，并逐问给出推导、条件或反例。

### 2026-08-15 · onestep-mse-vs-t（PixelFlowICLR 实验 1）
【用户原始要求】（节选，全文见 tasks/onestep-mse-vs-t.md）
> 只做 one-step prediction，不做 Langevin、迭代优化或 ODE rollout。
> 尽量复用现有 scheduler、stage construction、WLS、model prediction 和 operator 代码，
> 不要重新实现已有逻辑，也不要修改 `playground_runs`。
> 实验需要 GPU，请按服务器现有方式申请 GPU 后运行，不要直接在 login node 跑。
> GPU可以申请A100

生效动作：`playground_runs/` 只读；实验代码全部 import 现有实现；GPU 经 Slurm `gpu` 分区
`--gres=gpu:a100:1` 申请；login node 仅允许 CPU self-test（无模型前向）。

<!-- 格式示例：
### 2026-08-13 · 关于 temporal-consistency 任务
【用户原始要求】
> 先不要实现
生效动作：STATE.yaml execution_allowed: false
-->

## 用户约束（2026-09-03）：handbook / 复现说明必须包含 S_stats 的计算步骤
原文：「这个记住（计算S_stats)，之后如果写handbook要计算，记得给这个加进去」
- Algorithm 4 的先验协方差 S 固定为 s_stats 的 per-class 谱统计
  （decisions/2026-09-03-fixed-spectral-class-S.md）。其数据文件
  `PixelFlowICLR/Algorithm2/s_stats/spectral_power_labelled.npz`（334MB）按 .gitignore 不入库，
  `main4.py` 在任何 sampling 模式下都硬性依赖它。
- 因此任何 handbook / README / 复现流程写到「运行 main4」之前，**必须**列出一步：
  `cd PixelFlowICLR/Algorithm2 && python s_stats/compute_s_stats.py`
  （单文件、一遍 50k ImageNet val、无 inference、~7 分钟；输出 4 个文件：
  s_pooled_statistics_{labelled,all}.json、spectral_power_{labelled,all}.npz），
  并注明数据路径依赖（ImageNet val + LOC_synset_mapping.txt，见 main4.S_STATS）。
