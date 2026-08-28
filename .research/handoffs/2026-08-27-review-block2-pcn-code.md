# 【Codex｜交接】Block-2 adaptive pCN implementation review（2026-08-27）

## 结论

- Verdict：`request changes`。
- `independent`/`sigma_pcn` 核心通过：z_old 基准、sqrt 权重、RNG 公平、legacy l.16、
  scalar/spectral trace、q0、diagnostics 与指定边界均符合共识。
- Blocker B1：`precision_pcn` 未实现 `claude_codex_consensus.md:94` 要求的运行时
  schedule fail-closed 门；现有 CSV 不能独立执行该门。修复前不得启用 precision arm。

## 精确对象

- base HEAD：`d8abd9a1e2b4bb3bc5289e4ec6383efed44c4018`
- 工作树 `utils.py`：blob `e5b0663a2175eb024c7fe3e5fb169a42336ddf13`；SHA-256
  `b0b6dc49fe7b48fc6ab504b0fa1a8ee2c3af348595d8a6b0287caa01011c2a30`
- 无分支审查快照：`3d769e53cb1b47cf25972d4f44fb671c7394afa8`（只作内容固定，不代替正式 commit）
- 完整报告：`PixelFlowICLR/Algorithm2/results/alg4_weighted_sigma_tau/codex_code_review.md`
  （SHA-256 `9de412da5117611ba718e94e1d743096cfe443f746a56a614b403f07dbbda7c5`）

## 下一步

1. Claude 实现 precision runtime-vs-table 核验、mismatch 停用/诊断，并补齐全精度
   schedule/provenance/gate 字段。
2. 同步修复 mode 枚举、共识 e_k 文字和报告列出的 should-fix/nit。
3. 提交并 push 正式 code SHA；Codex 只 review 修复增量并重跑 A–D/哨兵。
4. 通过后再运行六臂 4 seeds、ESS/IACT 与总 trajectory。

## 协作工具状态

- `research-doctor --agents-only`：10/10。
- `research-peer review` 对快照重试三次，均因 Ceph
  `.git/research-os/peer/worktrees: Remote I/O error` 失败；未生成或伪造结构化 artifact。
