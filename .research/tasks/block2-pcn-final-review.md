# block2-pcn-final-review（2026-08-27）

## 【用户原始要求】（逐字）

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

## 任务判定

- 类型：最终 code review + methodology/result interpretation。
- `execution_allowed: false`：只读源码与指定产物；仅追加用户指定报告节及 Research OS 记录，不修改采样器或重跑 GPU 实验。
- 交付：目标新节标题必须逐字为 `## 终审与共同解释(2026-08-27)`，总行数不超过 80 行，结尾给 `approve` 或 `request changes`。

## 【Codex｜方案】（独立初稿；未读本轮 Claude 方案）

1. 逐项核对 B1/S3/N1 的源码顺序、fail-closed 语义、验证脚本接口、CSV provenance 与 JSON 字段。
2. 从 `summary_metrics.csv` 和 inner/frame CSV 独立复算六臂与机制聚合，不把 spread 单指标等同于 posterior covariance。
3. 按规格十问逐条给判断，区分：固定中心的条件核不变性、有限步路径效应、终态跨种子 spread 与链内 covariance/ESS。
4. 先固化 Codex 结构化 plan，再获取 Claude 的独立结论并 compare；至多两轮处理实质分歧。
5. 追加不超过 80 行的终审节，做标题唯一性、行数、数字、措辞红线与 diff 核验。

## 当前状态

- 状态：review（交付已写；verdict=`request changes`，待正式重跑收据、远端精确 SHA 与 Claude 结构化 compare）。
- 双 agent preflight：`research-doctor --agents-only` 10/10 通过。
- 报告已在目标文件末尾追加唯一标题节，共 71 行（上限 80）；完整报告 SHA-256
  `05cc71aa629ffa7f8696861249918ce0550d7832d28994ac17b1cfdf4b0959c9`。
- 技术销项：B1/S3/N1 当前源码/产物均通过；80 行 λ 表闭式逐列复算误差 0；precision
  保存产物 320/320 gate pass。正式产物时间早于入口全表预检修复，故“新预检下四种子重跑”
  缺持久化收据；`utils.py` 仍为未提交工作树 blob `51c1e478…`，不能 formal approve。
- 六臂/机制复算：去重后 960 frame + 1920 inner；与 `summary_metrics.csv` 完全一致。
  唯一解释收窄：pooled terminal spread 并未保持（sigma −11.5%、precision −9.8%），只能
  解释为 finite-horizon ensemble/mixing trade-off，不能说目标 conditional covariance 被缩小。
- 推荐：`independent` 默认；只能留一个 pCN 主候选时选 `precision_pcn` opt-in，
  `sigma_pcn` 仅作机制 ablation；严禁“更好的 posterior 采样”。
- 结构化协作：独立 Codex 方案在读取本轮 Claude 意见前已固化。`research-peer plan` 对 Codex
  三次、Claude 一次均被 `.git/research-os/peer/worktrees: Remote I/O error` 阻断；曾将
  runtime 可恢复备份为 `.git/research-os/peer.eio-20260827-1915` 并重建，错误仍复现。
  未伪造 plan/compare artifact；结果级 Claude compare 尚待补齐。
