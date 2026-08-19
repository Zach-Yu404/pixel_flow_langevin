# review-alg2-debug-directory

state: done（增量复审仍 request changes，等待下一轮修正） · owner: codex · type: review · issue: 3 · pr: null · created: 2026-08-18

## 【用户原始要求】

> 按 AGENTS.md 执行 sync protocol。先 gh issue view 3 读正文，确定 review 对象：关联 PR，或 Issue 里指定的 commit 区间（无 PR 时）。用 base+diff+受影响代码/tests 审查（不重读整仓）。结论以【Codex｜Review】中文评论发在 PR（无 PR 则发在 Issue #3）：approve 或列出必须修改项。若列出必须修改项，另外执行 gh issue edit 3 --add-label 触发:needs-fix。同步 .research/ 相关 task 文件并提交。

## 已确认 Interpretation

- Issue #3 无关联 PR，正文明确指定 review 区间 `b0c9351..6a5a927`。
- 审查结论发布到 Issue #3；仅在存在必须修改项时添加 `触发:needs-fix`。
- 范围限定为 base、区间 diff、受影响代码/tests 与相关 `.research/` 证据，不重读整仓。

## Scope

包含：Algorithm2 科学/数学正确性、配置内联与默认值等价性、数据流/随机性、实验结论及测试证据。

不包含：区间外代码、当前工作区其他未提交改动、scope 外风格治理。

## 约束

- 审查优先级遵循 `research-code-review`：科学正确性 > 正确性 > 实验有效性 > tests > 风格。
- 所有意见须给出文件/行号、错误原因与期望行为；scope 外问题不阻塞本 review。

## 计划

### 【Codex｜方案】

1. 以 `b0c9351` 为 base 核对区间提交、完整 diff 与实际受影响入口。
2. 分别检查 Alg2 公式/算子约定、配置值迁移、随机性/实验公平性、结果可复算性和回归测试。
3. 运行无 GPU 的定向静态/单元检查；GPU 结果只按已保存产物与日志证据核验。
4. 发布中文 review；随后同步 canonical memory、写 handoff 并提交。

## 执行记录

- sync snapshot：`memory_version=18`，`HEAD=6a5a927`。
- `git pull --ff-only`：因环境将 `.git/FETCH_HEAD` 设为只读而失败；本轮以当前 HEAD 与 GitHub 实时状态为准。
- Issue #3 review 对象：`b0c9351..6a5a927`（无 PR）。
- stale reconcile：远端 `IP_branch=736fae3`，仅比 review head 多一个 `CLAUDE.md` watcher 说明提交；
  远端 `.research memory_version=18`、约束无变化，不扩张 review 区间。
- 定向验证：`py_compile`、JSON 解析通过；pixelflow conda 环境下 `verify` V1–V8 运行完成，
  V1–V7 全部通过；五任务 one-step self-test 通过（adjoint worst `5.59e-09`、S1 worst
  `1.08e-07`、stage-3 S2 `0.0110`）。
- 配置静态对照：最终公共 sampler 参数、五任务 operator/sigma、random `num_langevin=15`、
  box/random trw=1、`cg_max_iter_l14=200` 与旧来源的实际消费值一致。
- Review 已发布到 Issue #3：`https://github.com/Zach-Yu404/pixel_flow_langevin/issues/3#issuecomment-5334464116`；
  已执行 `gh issue edit 3 --add-label '触发:needs-fix'` 并回读确认。

## 结果

### 【Codex｜Review】（2026-08-18）

审查对象：Issue #3 指定的 `b0c9351..6a5a927`（无 PR）。结论：**request changes**。

必须修改：

1. `utils.py:303-331` 在 τ=0 使用 `M=M0+epsilon*I`，随机 RHS 却只有协方差 `M0` 的
   `xi_y/xi_h`，漏掉 `sqrt(epsilon)*xi_epsilon`。因此所谓 exact draw 的协方差是
   `M^{-1}M0M^{-1}` 而非 `M^{-1}`；需明确目标并修公式，补 τ=0 empirical covariance test。
2. `utils.py:313-335` 的 `x0_langevin_steps>1` 只计算一次 `x_tau/v/x0_hat`，随后复用冻结
   score，不是对当前 x0 的多步 Langevin。K=3/5 只能称 frozen-score ablation；若要解释为
   Block-2 mixing，需每子步重算并计入 NFE，同时真正 sweep 外层 S。
3. debug 根因/anchor 证据需要重做或降级：scalar oracle 未保存脚本/参数且把固定 τ 的宽化
   target 直接与原始 prior std=0.2 比；“line 15 清零进展”与
   `x_tau_next=x_tau+sigma*Delta_x0` 的代数不符。历史 anchor/K 分支还会额外消费同一 RNG，
   后续噪声流错位，且仅 seed 42；当前证据只支持固定 demo/seed 的探索性结果。
4. active config/API 有 silent no-op：`anchor` 实现已注释，但 config 和 debug sweep 仍接受
   非零值并输出 `anchor=a` 标签。停用期间必须移除或 fail fast；恢复时再完整实现与验证。
5. terminal projection 没进入 full_ip 持久结果：`main.py:425` 丢弃返回的 post-projection
   x1，而 rows/traj 均在投影前记录。需至少保存 post final reconstruction/metrics，并明确 pre/post。
6. 单配置重构仍可静默漂移：相对 `out` 由 `main.py:79-80` 锚到调用者 CWD（从仓库根运行会
   写到根 `results/`）；缺失的显式 config 会按 basename 静默回退；`.get` 与 `**unused_kw`
   会吞漏键/错拼。需锚定 Algorithm2/config 目录、显式路径 fail-fast、校验 required/allowed keys，
   并补 CPU config contract test（含 CG/trw/task override）；`TRAJ_IMAGE` 也应进入 config。
7. 证据口径必须修正：trw 两条 CSV 轨迹 40 步中 `mse_x1/hole/obs` 分别 25/26/24 步不等，
   最大差 `2.38e-6/1.05e-5/1.08e-7`；f9299d7 与 6a5a927 末端 hole/obs 也不逐位相等。
   只能写四位小数一致/数值等价，除非保存 tensor hash 并 `torch.equal`。S2 以 GT 作 warm start，
   solver 原样返回也会通过，需改为零/随机 warm start并检查残差/稠密参考及 τ=0 路径。
8. `demo_runner.py` 的 Python `hash()` 仍控制 mask；删掉所有固定 `PYTHONHASHSEED=0` 的 sbatch
   后，直接入口会静默不复现。应改稳定 digest seed，或至少对非 verify 模式 fail-fast 并记录
   mask checksum/bbox。V8 目前不参与 `checks["ok"]`，输出须标明只有 V1–V7 为 gating。

已确认：`apply_B/apply_N/compute_sigma_tau/score_solve` 的符号与系数、当前 G 的自伴性、
class label 修复、terminal replacement 代数、最终 config 的静态值均正确。修正后只复审增量。

## 【Codex｜增量复审】（2026-08-18）

- 对象：无 PR；只复审修正链 `6a5a927..a8ac308`，后续无关提交与工作区未提交改动不纳入。
- 结论：**request changes**；评论：
  `https://github.com/Zach-Yu404/pixel_flow_langevin/issues/3#issuecomment-5337002138`；
  已添加并回读确认 `触发:needs-fix`。
- 已确认正确：生产 τ=0 RHS 已加 `sqrt(epsilon)*xi`；`_resolve_out`/显式 config 路径正确；
  full_ip 已保存 post 指标；V8 变量污染已被独立命名与 S1 guard 阻断；py_compile/JSON 通过。
- 仍阻塞：
  1. V9 在 NumPy 中复制公式、不调用生产 RHS，mutation 后仍可 PASS；默认 res=16/n=40000
     在本审查环境于 V9 OOM（exit 137），需共用 helper/生产路径并降低峰值内存。
  2. K 轴共用单 RNG，额外 `xi_0` 平移后续 Block-1 随机流；h0=0 的 K=1/K=3 toy
     仍输出不同。debug_box 还未安装/reset/保存 NFE，需拆流并补 invariance/call-count tests，
     或移除未验证的真多步轴。
  3. config 仅验键：字符串 false、负 trw、历史 CG/trw 值漂移均可通过；默认 debug_box
     会把非零 `algorithm.anchor` 硬改 0，绕过 sampler guard。缺类型/范围/canonical-value test；
     hash guard 正确，但 mask checksum/bbox 与 subprocess seed test 仍缺。
  4. S2 只改零 warm start、断言 observed RMSE，未测线性残差/稠密参考/τ=0 ridge/CG cap。
  5. V8 stdout 未标 diagnostic/non-gating，末尾仍笼统打印 `ALL CHECKS PASSED`。
  6. n=4 单图统计不能外推 5% 噪声地板或“确证/方向成立”；跨机最差只同到 6 位有效数字，
     `1.1920929e-7` 在 0.969 附近是 2 ULP；相关 task 仍保留未作废 bit-exact 旧结论。

修正后仍只看本轮增量；证据项若统一降级措辞，无需为本 review 重跑 GPU。
