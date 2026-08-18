# 2026-08-18 review-alg2-debug-directory

【Codex｜交接】

目标：审查 Issue #3 指定的 `b0c9351..6a5a927`，发布 review 结论并同步 canonical memory。

当前状态：request changes；等待 Claude 修正后 Codex 只复审增量。

已完成：
- 按 sync protocol 读取 L1/L2、Issue 正文、base+diff、受影响代码/tests；确认无 PR。
- 核对 Alg2 数学、配置内联、GPU CSV/log 与实验口径；发布 8 类必须修改项。
- pixelflow 环境实跑 CPU verify 与 one-step self-test；静态复算 trw/等价性 CSV。
- Issue 评论：`https://github.com/Zach-Yu404/pixel_flow_langevin/issues/3#issuecomment-5334464116`；
  已添加并回读确认 `触发:needs-fix`。

修改文件：
- `.research/STATE.yaml`
- `.research/CURRENT.md`
- `.research/tasks/review-alg2-debug-directory.md`
- `.research/tasks/{cleanup-algorithm2-directory,debug-box-alg2-hole,test-terminal-replace-weight}.md`
- 本 handoff

branch：`IP_branch`
review base/head：`b0c9351` / `6a5a927`
sync remote head：`736fae3`（仅新增 CLAUDE watcher 说明，不影响 review）
commit：本 handoff 所在 memory 提交

tests：
- `python -m py_compile PixelFlowICLR/Algorithm2/{main,utils}.py`：通过
- `jq empty config.json gamma2_meas.json`：通过
- `conda run -n pixelflow python ... main.py --mode verify`：V1–V7 全过；V8 仅 probe
- one-step CPU self-test：adjoint worst `5.59e-09`；S1 worst `1.08e-07`；stage-3 S2 `0.0110`
- CSV 独立复算：trw 轨迹非逐位；f9299d7→6a5a927 非 bit-exact

结果：实现的若干窄项正确，但科学抽样、实验解释、配置契约与结果持久化存在阻塞；不 approve。

assumptions：
- review 截止严格按 Issue 正文为 `6a5a927`；远端后续 watcher 文档提交不纳入代码审查。
- 不重跑 GPU；只核验已保存的 CSV/log 与 CPU tests。

known concerns：
- 本环境 `.git` 只读，常规 pull/commit 不可用；memory 发布需以远端 `736fae3` 为父提交，
  避免本地 `6a5a927` 上产生分叉。
- Algorithm2 的 PNG 与 oracle 原始脚本/输出未进入 artifacts manifest，视觉/oracle 强结论不可审计。

remaining work：
- Claude 修复 Issue #3 评论中的必须修改项并请求增量复审。

recommended next step：优先修 τ=0 抽样协方差、K/anchor silent semantics、config contract、
full_ip post 输出与证据口径；昂贵实验可选择多 seed 重跑，或先降级结论。
