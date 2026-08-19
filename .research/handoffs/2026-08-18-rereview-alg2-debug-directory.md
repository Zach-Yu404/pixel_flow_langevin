# 2026-08-18-rereview-alg2-debug-directory

【Codex｜交接】

目标：复审 Issue #3 对首轮 8 条必须修改项的修正增量，发布结论并同步 canonical memory。

当前状态：`6a5a927..a8ac308` 增量复审为 request changes；等待下一轮修正。

已完成：
- 按 Sync Protocol 核对 L1、Issue 正文/最新修正评论、远端 HEAD 与 memory_version。
- 用 commit 快照审查代码、测试与 CSV/.research 证据，未混入工作区后续改动。
- 发布 Issue 评论 `5337002138`，添加并回读确认 `触发:needs-fix`。
- 记录 6 类阻塞与已确认正确的窄项到 review task、相关 task、CURRENT/STATE。

修改文件：
- `.research/STATE.yaml`
- `.research/CURRENT.md`
- `.research/tasks/review-alg2-debug-directory.md`
- `.research/tasks/cleanup-algorithm2-directory.md`
- `.research/tasks/debug-box-alg2-hole.md`
- `.research/tasks/test-terminal-replace-weight.md`
- 本 handoff

branch：`IP_branch`
commit：本 handoff 所在 memory 提交
tests：`py_compile main.py utils.py` 与 JSON 解析通过；CSV 四 seed 统计独立复算一致；
config AST 反例证实字符串 false/负 trw/非零 anchor 可通过；生产控制流 toy test 证实 h0=0 时
K 改变仍平移后续 RNG，且 recompute NFE 次数符合 K 但 debug_box 不记录。默认 verify
`res=16,n=40000` 在本环境于 V9 OOM（exit 137）；V9 静态/mutation 审计确认不引用生产 RHS。

结果：生产公式与若干数据流修正成立，但回归测试、实验契约和证据口径仍不足，不能 approve。

assumptions：
- 无 PR；首轮区间已审，re-review 仅覆盖 `6a5a927..a8ac308`。
- 后续 `8058e07/3c66cc5/2d99535/ea4653b` 与未提交 measurement-alignment 工作不在本轮 scope。

known concerns：
- 当前运行环境对原工作树 `.git` 只读；常规 pull/commit 不可用，memory 提交需从同 repo 的
  临时可写 clone 基于远端 HEAD 完成，再核验远端文件。
- 证据 task 中的旧强口径已由本 handoff/增量复审段标记为待修，但实现者仍需正式改正文。

remaining work：
- 按 Issue 评论修 V9/S2/config+seed/K+NFE/output gating；或移除未验证功能并进一步降级结论。
- 修正后派发 Codex，只复审新增增量。

recommended next step：先补小型 CPU production-path/config/CG/seed tests 与入口校验；证据项优先
改为固定 seed 描述性观察，可避免不必要的 GPU 重跑。
