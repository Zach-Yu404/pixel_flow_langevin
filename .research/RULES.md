# Research OS 共享规则（Claude + Codex 共同遵守）

rules_version: 6

本文件是 CLAUDE.md 与 AGENTS.md 共同引用的唯一规则源。两个 agent 必须完整遵守。
冲突时优先级固定：**用户明确指令 > 本规则 > agent 自身默认行为**。

---

## 0. 最高原则

```
Chat ≠ Memory    Agent ≠ Memory    Server ≠ Memory
Project Canonical State (.research/) = Memory
```

- 用户原始要求 = Authoritative Source。任何 interpretation 都不能扩大、缩小或改变它。
- 凡不确定的重要事情——尤其是路径、服务器、迁移、权限、scope——先问用户，不要猜。
- 不降低模型、不降低 reasoning effort；通过减少重复上下文和重复工作节省 token。

## 1. 项目隔离

- 1 Project = 1 Git Repo = 1 Canonical Memory Namespace。
- 禁止：搜索 sibling 项目目录、读取其他项目的 `.research/`、自动复用其他项目的研究结论。
- Global 层（~/.claude、~/.codex、research-init）只放：角色、通用 workflow、通用 skills、
  MCP/plugin 配置、安全规则、工作偏好。**任何项目知识必须留在该项目 repo 的 `.research/` 内。**
- 例外：用户明确要求 cross-project work。

## 2. Canonical Memory（.research/）

结构与语义见 `.research/README.md`。关键约定：

- `STATE.yaml` — 机器可读当前状态（memory_version、mode、active_tasks、constraints 快照）。
- `CURRENT.md` — 人类可读当前状态摘要（≤1 页）。
- `CONSTRAINTS.md` — 用户约束全文，含【用户原始要求】原文。**约束一出现立即写入，先于一切其他工作。**
- `tasks/<verb-object>.md` — 每个任务一个文件，第一节永远是【用户原始要求】逐字记录。
- `decisions/`、`experiments/`、`references/`、`handoffs/`、`history/` — 见各自 TEMPLATE。
- `local.yaml` — machine-specific 路径，**gitignored**。由 `research-discover-paths`
  **自动扫描生成**（checkpoints/datasets/大文件），不要求用户手填。agent 在
  "初始化记忆"、换机器后、或发现文件与记录不符时自动重跑并汇报发现结果。
  **只在自动发现无法解决时才问用户**（如：多个候选 dataset 分不清用哪个、
  引用的 checkpoint 本机不存在、任务需要 discovery 覆盖不到的外部存储）。
- Canonical memory 内一律使用相对路径；绝对路径只允许出现在 `local.yaml`。
- 路径类提问的分界：**破坏性/迁移/写入外部位置必须问**（§8 不变）；
  **只读的路径发现自动做**，不拿"请提供路径"打扰用户。

## 3. 开始工作前（Sync Protocol）

任何 meaningful work 开始前：

1. `git pull`（有 remote 时）；读取 `STATE.yaml` + `CURRENT.md` + `CONSTRAINTS.md`（Level 1）。
2. 记录自己看到的 `memory_version` 与 `HEAD` commit 到 STATE.yaml 的 `last_seen.<agent>`。
3. 按需分层加载：
   - L2：相关 decisions / consensus / 最新 handoff
   - L3：相关源码符号、tests、git diff、相关 commit
   - L4：完整历史讨论 / 全文论文 / 大 log —— 仅在确实需要时
4. 不要整仓库通读；代码检索用 symbol → function → caller → deps 逐层展开。

发布结果前（写回 memory / 提 PR / 发 GitHub 评论前）：

1. 重新读取 `STATE.yaml`。若 `memory_version` 比自己记录的新 → **stale**。
2. Stale 处理：读取 delta（新 constraints、新 decisions、新 handoff）→ reconcile → 继续。
   **绝不允许覆盖更新的用户约束。**

## 4. 任务类型识别

任务默认**不是** implementation。先判断类型：
explain / research / paper-understanding / url-analysis / code-understanding / debug /
design / plan / review / implementation / modification / test / experiment / mixed。

- 用户说"先不要实现"/"只 plan" → `STATE.yaml` 立即写 `execution_allowed: false`，
  另一个 agent 下次 sync 必须看到。**没有 execution_allowed: true 不进入 implementation。**
- explanation-only 任务不创建 branch、不改代码。

## 5. 任务命名

`verb + core object`，小写连字符：`add-warped-noise`、`fix-flow-direction`、
`debug-stage3-instability`。禁止 `TASK-031` / `EXP-0042` 式编号名。
GitHub Issue/PR 标题用简短中文（如「加入 warped noise」）。

## 6. 双 Agent 协议

**角色**（默认分工，非绝对）：
- Planning/understanding 阶段：Claude 与 Codex 均为 Peer Research Agent。
- Implementation 阶段：Claude = Primary Implementer；Codex = Primary Reviewer/Debugger。

**独立 Plan**：重要任务双方各自独立读取 shared facts 后独立生成 plan。
禁止一方看另一方的 plan 后直接附和——先写完自己的再比较。

**Consensus**：达成一致后写非常短的【共识】到 `decisions/` 并同步 STATE.yaml。
共识 = authoritative task understanding。

**Disagreement**：交换 objection → 各自 revise。数轮后仍无共识 → 状态置 `needs-user`，
停止，向用户列出：分歧点 / Claude 推荐 / Codex 推荐 / 核心 tradeoff。**不允许投票表决。**

**对话内自动互调**：用户在其中一个 agent 的对话里布置任务时，到达需要另一方的节点
（独立方案、共识比较、review），当前 agent **自动 headless 调用另一方**，无需用户操作：
- Claude 调 Codex：`codex exec --skip-git-repo-check "<任务上下文+要求>"`
- Codex 调 Claude：`claude -p "<任务上下文+要求>"`
调用时给足上下文（任务名、Issue 号、读哪些文件），并明确要求对方先走 sync protocol、
输出以【Codex｜方案】/【Claude｜方案】等标记落到 `.research/` 或 GitHub。
**独立性纪律不变**：请求独立方案时不得把自己的方案放进 prompt。
对方 CLI 不可用/超时 → 按 §12 degraded mode 继续，不阻塞用户。

**Shared Facts vs Independent Reasoning**：
- 机械性事实（ls 结果、依赖关系、test 输出、log 摘要）写入 shared memory，只做一次。
- 对事实的分析、判断、方案必须各自独立完成，不同步中间推理。

## 7. 状态机

```
new → planning → (needs-user) → ready → working → review → done
```

只用这 7 个状态。GitHub label 与之一一对应（见 orchestrator/github/labels.yaml）。

## 8. 必须进入 needs-user 的情况

无法达成共识；用户要求有真实歧义；路径/repo/storage/migration mapping 未明确；
scope 变化；重大 architecture 变化；destructive 操作；重大依赖或环境变化；
单 agent 想绕过新 implementation 的 consensus。

**路径 / 服务器迁移 / 数据迁移**：找到现路径 → 明确新路径 → 列出受影响内容 →
展示计划 → 等用户明确确认 → 执行。无确认 = NO EXECUTION。

## 9. Handoff

每个 meaningful phase 结束写 `handoffs/<date>-<task>.md`，至少含：
目标 / 当前状态 / 完成了什么 / 修改文件 / branch / commit / tests / 结果 /
assumptions / known concerns / remaining work / recommended next step。
Handoff 必须让另一个 agent（或全新 session）不依赖本次对话即可接手。

## 10. GitHub 协议

- Issue = 任务讨论、research、planning、agent 通信、历史。
- PR = implementation、review、debug、fix。
- Repo `.research/` = 压缩后的 canonical truth；GitHub = 完整耐久通信历史。
- **所有人类可读记录一律中文**（标题、正文、评论、review、handoff、结果）。
  技术 identifier（函数名、路径、branch、SHA、命令、数学符号、原始报错）保留英文。
- Agent 通信统一标记：
  `【Claude｜方案】【Codex｜方案】【Claude｜问题】【Codex｜问题】【Claude｜同意】`
  `【Codex｜同意】【Claude｜有异议】【Codex｜有异议】【Claude｜交接】【Codex｜Review】`
  `【结果】【需要用户决定】`
  简洁，不闲聊。
- Commit message 遵循 repo 既有 convention；无 convention 时用简洁中文或中英混合。

## 11. 并行执行

- 1 execution task = 1 owner = 1 branch = 1 worktree。
- 同一 task 不允许两个 implementation owner。
- 依赖关系记录在 STATE.yaml 的 `depends_on`；有依赖的任务不得错误并行。
- Worktree 绝对路径写 `local.yaml`（gitignored），不进 canonical memory。

## 12. Single-Agent Degraded Mode

一方因 token/quota/服务不可用时：

- `STATE.yaml` 写 `mode: single-agent-degraded` + `unavailable_agent` + 时间。
- 另一方**可以**继续：explanation、research、读论文、代码理解、memory 检索、
  debug 调查、结果分析、test 检查、进度汇报、文档、plan 准备、证据收集，
  以及**已有 consensus + 已批准 scope 内**的 implementation / fix / tests / experiments。
- **新的重大 implementation** 默认停在 `waiting-for-second-agent-consensus`，
  除非用户明确授权单 agent 继续。授权必须记录：
  ```yaml
  single_agent_execution_authorized: true
  authorized_agent: claude   # 或 codex
  scope: <具体范围>
  ```
  授权只作用于该 scope，不扩展到未来任务。
- 单方完成的工作标记 `pending <agent> review`，照常写 handoff。
- 恢复方 catch-up：sync → 读 STATE delta → 相关 commits → 最新 handoff → **只 review 增量**，不重做。

## 13. Token Efficiency

优先级固定：Correctness > User constraints > Consensus semantics > Recoverability >
减少冗余上下文 > 减少重复工作 > 减少 token。前四项不可为省 token 牺牲。

手段：incremental sync、targeted retrieval、canonical summaries、diff-based review、
symbol-based code retrieval、compact handoff、paper note 复用（`references/`）。
禁止：重复整仓通读、重复读全文论文、重复跑同样的机械检查（写成 Shared Fact 复用）。

## 14. 实验与工件

- 重要实验记录：commit、config、command、seed、Python/包版本、CUDA、GPU、
  dataset/checkpoint reference、metrics、notes（用 `experiments/TEMPLATE.md`）。
- Git 不保存大文件（dataset/checkpoint/视频/大结果）。用 `.research/artifacts.yaml`
  manifest 指向外部存储：logical_name、location、checksum、size、producer task、producer commit。
- Artifact store 的选择先问用户。
- 环境管理工具（conda/uv/poetry/docker）先 inspect 项目现状再定，不自作主张。

## 15. 进度查询

用户问进度时：**只读真实状态**（STATE.yaml、CURRENT.md、GitHub Issue/PR、commit、
tests、handoff），不凭对话印象回答；不打断正在运行的 worker。输出格式：

```
已完成 / 当前 / 下一步 / 阻塞 / 需要我处理
```

没有明确 checklist 不报百分比。

## 16. 统一记忆口令（对 Claude 和 Codex 完全一致）

用户用自然语言操作记忆时，两个 agent 执行**相同**的动作。识别语义而非死抠字面：

| 用户表达（示例） | 动作 |
|---|---|
| "初始化记忆" | 跑 `research-discover-paths`（自动发现 checkpoints/datasets）→ **导入已有记忆源（见下）** → 读现有代码/git 历史 → 填充 PROJECT/ARCHITECTURE/CURRENT + artifacts.yaml（含 checksum）→ 只有读不出来的才**问用户** → commit + push |
| "把 X 导入记忆"（X=笔记/导出/旧 memory 路径） | 读 X → 按内容归位（事实→ARCHITECTURE、状态→CURRENT、计划→tasks、论文→references、约束→CONSTRAINTS）→ commit + push |

**初始化时必查的已有记忆源**（自动做，不等用户点名）：
1. **项目内已有文档**：README、notes、实验计划（如 `experiments_plan.md`）、TODO 等
   —— 吸收要点进 `.research/`，原文件保留，PROJECT.md 里登记为"权威入口"引用而非复制。
2. **Claude Code auto-memory**：`~/.claude/projects/<本项目路径 slug>/memory/*.md`
   —— 属于本项目的知识**迁移进 `.research/`**（项目知识必须在 repo 内，见 §1 隔离），
   迁移后在原文件留一行指针；与本项目无关的条目不动。
3. **Codex 侧历史**：`~/.codex/` 内本项目相关的记忆/笔记（若可读）——同上迁移。
4. **用户指定的其他来源**（旧对话导出、散落笔记）——用户给路径就读，读完汇报归位结果。
导入完成后向用户汇报：从哪些来源导入了什么、写到了哪些文件、有哪些内容因矛盾/过时被搁置待确认。
| "同步记忆" / "拉一下最新状态" | `git pull` → 重读 L1（STATE/CURRENT/CONSTRAINTS）→ 对比自己 `last_seen` → **向用户汇报 delta**（新增约束/决策/handoff/commit），不做其他事 |
| "更新记忆" / "把刚才的记进去" / "记住 X" | 把本次对话的新结论写入对应位置：事实→ARCHITECTURE 或 context/facts；决定→decisions/；约束→CONSTRAINTS + STATE 开关；进展→CURRENT + 相关 task 文件。memory_version +1 → commit + push → 一句话汇报写了什么、写到哪 |
| "写个 handoff" / "存个 checkpoint" | 按 §9 写 handoffs/ → 更新 CURRENT + STATE → commit + push |
| "修正记忆：X 不对" | 定位错误条目 → 改正（decisions 被推翻时不删除，标注"已否决+原因"）→ memory_version +1 → commit + push |

**Prompt-first**：用户的主入口是自然语言，CLI（`research-init/bin/*`）只是实现工具。
用户说"接入这个项目"/"做个体检"/"同步最新规则"/"连一下 GitHub"时，agent 自己找到并
执行对应工具（本机没有 research-init 就先 `git clone` 它），把结果汇报给用户——
**不许反过来让用户去敲命令**（除了必须用户本人完成的：登录、sudo、输密码）。

纪律：
- 写入前先 `git pull` + stale 检查（§3），避免覆盖另一个 agent 刚写的内容。
- 每次记忆写入必须 commit（有 remote 则 push）——**另一个 agent 只能看到已提交的记忆**。
- 汇报要具体到文件："已写入 decisions/2026-08-14-use-clean-endpoint-cg.md 并 push"，
  而不是"已更新记忆"。

## 17. Session 与恢复

- 假设任何 context window 随时消失。重要阶段完成即写 canonical memory / GitHub。
- 恢复 = `git clone` → 按 `local.yaml.example` 确认机器路径 → 读 L1 → 继续。
- 不依赖旧对话、旧服务器绝对路径、旧 agent context。
