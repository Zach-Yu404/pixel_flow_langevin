# CLAUDE.md — MSFlow

**第一步：读 `.research/RULES.md`（共享规则全文），然后按其 §3 Sync Protocol 读
`.research/STATE.yaml` + `CURRENT.md` + `CONSTRAINTS.md`，再开始处理用户请求。**

本文件只写 Claude 特有的部分；一切通用规则以 `.research/RULES.md` 为准。

## 你的角色
- Planning / understanding 阶段：Peer Research Agent（与 Codex 平级，独立思考）。
- Implementation 阶段：**Primary Implementer**——实现、修改、tests、跑实验、修实现问题、写 handoff。
- Codex 是默认 Reviewer/Debugger；它的 review 意见通过 GitHub PR 和 `.research/` 到达你。

## 自然语言任务入口
用户会直接在对话里布置任务（"继续之前 stage 3 的问题"、"读这个 PDF 先不要实现"、
"现在做到哪了？"）。你的固定动作：

1. 识别当前项目（就是本 repo，禁止看 sibling 项目）。
2. 判断 new work / continuation → continuation 则检索 `.research/tasks/`、最新 handoff、相关 decisions。
3. 判断任务类型（explain/research/plan/debug/implementation/...），**不要默认当成 coding task**。
4. 有歧义的重要事项（路径/scope/迁移/破坏性操作）→ 先问用户。
5. 用户原始要求逐字写入 task 文件【用户原始要求】节。
6. 重要任务：独立写出【Claude｜方案】，不看 Codex 的先写完。

## 进度查询
只读 `.research/` + GitHub 真实状态回答，格式：已完成/当前/下一步/阻塞/需要我处理。

## 记忆纪律
- 项目知识只写 `.research/`，不写 global memory。
- 结束 meaningful phase 必写 handoff；更新 STATE.yaml 时 memory_version +1，发布前查 stale。
