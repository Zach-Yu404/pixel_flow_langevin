# 2026-08-18 Issue #3 修正应答（待在集群侧发到 Issue）

**本机（新挂载 GPU 机）无 gh、无 GitHub 凭证**（HOME 不共享），以下正文需在集群 login node
执行：`gh issue comment 3 -F .research/handoffs/2026-08-18-issue3-fix-response.md`
（去掉本行以上的说明），并 `gh issue edit 3 --remove-label '触发:needs-fix' --add-label '状态:review'`
触发增量复审。

---

【Claude｜修正】针对 `b0c9351..6a5a927` 的 8 条 request changes，已修 5 条、降级 2 条、
1 条按用户 scope 约束改在 Algorithm2 内解决。所有验证均在新挂载机实跑（4×A100，无 Slurm）。

**1. τ=0 ridge 漏 `sqrt(epsilon)` 噪声 —— 已修 + 补经验协方差测试**

`utils.py` 的 l.13 增加 `if epsilon: b_tilde += sqrt(epsilon) * randn_like_cpu(x1)`。
新增 **V9** 经验协方差测试（`verify` 模式，dense float64，n=40000 抽样）。测试设计上
放大 `ridge_rel` 到 0.5，否则 1e-6 的缺陷远小于蒙特卡洛地板、无法判别；并让每个变体
与**它自己的**理论协方差对照：

| 变体 | 理论 cov | 实测相对误差(谱范数) | 判据 |
|---|---|---|---|
| 修正后（含 √ε·ξ） | `M^-1` | 1.203e-01 | PASS (MC 地板 0.259) |
| 修正前 | `M^-1 M0 M^-1` | 1.150e-01 | PASS |
| 两理论之间的差距 | — | mc_tol/gap = 0.262 | PASS (<1，即差距大于地板) |

即：**旧实现的抽样协方差确实是 `M^-1 M0 M^-1`**，评审判断成立。V1–V9 全通过。

**2. K>1 是 frozen-score —— 已正名 + 补真多步实现**

新增 `x0_langevin_recompute`（config `algorithm` 节）。默认 `False` = 旧的 frozen-score
语义（保持历史可比）；`True` 时每个子步重算 `x_tau/v/x0_hat`，额外 NFE 由现有计数器计入。
`.research` 中"K↑ 单调恶化"已改写为"frozen-score 消融下的观察"，并注明真多步与 S sweep 未跑。

**3. 根因/anchor 结论 —— 已降级**

`tasks/debug-box-alg2-hole.md` 的"最终结论"改为 **junco 单图 / seed 42 的探索性观察**，
逐条标注：h₀/γ² 的结论限于已测设置；K 的结论限于 frozen-score；oracle 缺可复算脚本、
"line 15 清零 Block-2 进展"的代数表述与 `x_tau_next = x_tau + sigma*Δx0` 不符，需重做；
历史 anchor/K 分支额外消费同一 RNG、各 variant 噪声未对齐。

补充实测支持这次降级：修 #1 后 τ=0 多了一次抽样、噪声流平移，同一 seed 的洞区
MSE 从 0.9691 变成 **1.0276（+6%）**，obs 不变（0.0018）。洞区指标对噪声流的敏感度
本身就说明单 seed 数值不可作结论依据。（seed 7/123/2024 的扫描结果见 task 记录。）

**4. anchor silent no-op —— 已 fail fast**

`run_posterior_sampling_alg2` 开头即校验：anchor 块仍被注释停用期间，传非零值直接
`ValueError` 并说明原因，不再静默无效。实测触发正常。

**5. full_ip 丢弃 post-projection x1 —— 已修**

改为 `x1_final, rows, traj = ...`，新增 `full_ip_final.csv`
（`task,image,trw,pre_mse,post_mse`，trw>0 再加 `post_hole/post_obs`），并在注释里
写明 rows/traj 是 pre-projection、该 CSV 是 post。smoke 实跑（box_inpainting/junco）：
`pre_mse 1.0176944732666016 → post_mse 1.0162274837493896`，`post_obs 0.0`。

**6. 单配置漂移 —— 已补严格契约**

- `_resolve_out` 改为锚定 `HERE`（Algorithm2 目录），不再跟随调用者 CWD。
- 显式 `--config` 不存在直接 `FileNotFoundError`，删掉 `HERE/basename` 静默回退。
- 新增 `CONFIG_SCHEMA` + `_check_config_keys`：顶层与各节的键集合必须**精确匹配**，
  `tasks_setup` 每个任务必须有 `sigma_n/operator/terminal_replace_weight`，
  `tasks` 条目必须有对应 setup，`traj_image` 必须在 `images` 内。
  实测：把 `h0` 写成 `h_0` → `KeyError: missing=['h0'] unknown=['h_0']`；
  删掉 `terminal_replace_weight` → 同样报错。
- `TRAJ_IMAGE` 已移入 config（`traj_image`），从 utils 的硬编码删除。

**7. 证据口径 —— 已修正**

删除全部 "bit-exact / 逐位一致" 表述。跨机复现的正确说法：集群与新挂载机
**同到 8 位有效数字，非逐位相同**（两机 conda env 各自安装，新机 torch 2.11.0+cu128）。
"投影不动洞区"改为恒等式证据：隔离 GPU 试验中
`torch.equal(x1p*hole, x1*hole) == True`、洞区 MSE 相对差 `0.000e+00`。
S2 warm start 已从 GT 改为**零**；改后数值不变（stage3 0.0110、stage1 0.0616/0.1116），
说明原结论不是被 warm start 掩盖的。

**8. hash() 种子 —— 在 Algorithm2 内 fail fast**

用户对本任务的 scope 约束是"只在 Algorithm2 下改动"，`demo_runner.py` 属 IP_package，
故未改其 `hash()` 实现。改为入口守卫：非 verify 模式下
`PYTHONHASHSEED != "0"` 直接 `RuntimeError` 并给出正确命令行；verify 不建 mask 故豁免。
实测触发正常。V8 只作诊断打印、不参与 `checks["ok"]` 的现状已在输出中体现（V1–V7 + V9 为 gating）。

**未做 / 需评审者注意**：真多步 Langevin 与 S sweep 的实测、oracle 脚本重做尚未进行
（已在 task 记录中标为未完成，不作为论文论据）。本机无 GitHub 凭证，本回复由集群侧代发。
