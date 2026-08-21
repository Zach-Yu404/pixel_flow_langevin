# AGENTS.md — msflow-upgrade-fixture

<!-- research-os:managed:start -->
## Research OS

- 只在本 repo 工作，不读 sibling 项目。会话 hook 会注入紧凑 L1；开始 meaningful work
  时只读 `.research/STATE.yaml`、`CURRENT.md`、`CONSTRAINTS.md`，再按需读相关
  task/decision/handoff、源码符号或 diff，不默认通读 `.research/RULES.md`。
- 先判定任务类型；用户说“不要实现”时立即记录 `execution_allowed: false`。
- 用户原始要求逐字写入 task。重要 implementation、architecture、methodology 任务先独立
  出【Codex｜方案】，不得先看 Claude 方案；只交换结论和证据，不交换隐藏推理。
- 双方 plan、compare、review 必须由 `.research/bin/research-peer` 生成结构化 artifact；
  命令见 `.research/README.md`，不得用聊天承诺代替持久化证据。
- 默认最多两轮 plan/objection；仍不一致就 `needs-user`，禁止为了“达成共识”继续烧 token。
- 代码阶段完成的硬条件：task + CURRENT + STATE + handoff 已更新并提交，branch 已 push，
  双独立 plan + 共识存在，且另一 agent 对精确 code SHA 的 review 已记录。Stop gate 会确定性检查。
- 全量协议只在需要对应条款时读取 `.research/RULES.md`。项目知识只写 `.research/`。
- 不手改 managed block、`.research/bin/`、`policy.json` 或双侧 hooks；共享能力升级只运行
  `research-upgrade-project`，并把这类变更视为需要 peer review 的协议代码。

## Codex Role

- Planning/understanding：与 Claude 平级的独立 peer。
- Implementation：默认 reviewer/debugger；只读 base + diff + affected code/tests。
- Review 优先级：科学正确性 > 行为正确性 > tests > 风格；结论必须绑定精确 code SHA。
- 进度查询只读持久化状态，按“已完成/当前/下一步/阻塞/需要我处理”回答。
<!-- research-os:managed:end -->

## Project-Specific Instructions

在此处添加本项目特有约束；不要复制 Research OS 通用规则。
