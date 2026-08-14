# .research/ — Canonical Project Memory

本目录是这个项目的唯一权威记忆（Claude 与 Codex 共同读写）。规则全文见 `RULES.md`。

| 文件/目录 | 内容 | 读取层级 |
|---|---|---|
| `STATE.yaml` | 机器可读当前状态：memory_version、mode、active_tasks | L1，每次必读 |
| `CURRENT.md` | 人类可读当前状态摘要（≤1 页） | L1，每次必读 |
| `CONSTRAINTS.md` | 用户约束全文（含原始表达） | L1，每次必读 |
| `PROJECT.md` | 项目是什么、目标、范围 | L1（新 session 首次） |
| `ARCHITECTURE.md` | 代码/方法架构事实 | L2 |
| `tasks/` | 每任务一文件，首节=【用户原始要求】 | L2（相关任务） |
| `decisions/` | 决策与【共识】记录 | L2（相关决策） |
| `experiments/` | 实验记录（可复现信息） | L2/L3 |
| `references/` | 论文/URL 精读笔记（复用，避免重读原文） | L2/L3 |
| `handoffs/` | 阶段交接记录 | L2（最新一份） |
| `history/` | 已完成任务归档、重要事件日志 | L4 |
| `context/` | Shared Facts（机械性观察，避免双 agent 重复劳动） | L2 |
| `artifacts.yaml` | 大文件 manifest（指向外部存储） | 按需 |
| `local.yaml` | 本机路径映射，**gitignored**，换机器重建 | L1（执行类任务） |

更新纪律：改动 STATE.yaml 时 `memory_version` +1；发布结果前先检查 stale（见 RULES.md §3）。
