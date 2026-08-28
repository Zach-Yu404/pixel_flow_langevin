# Handoff：Block-2 pCN 终审与共同解释（2026-08-27）

## 目标与当前状态

- 目标：核验 B1/S3/N1 最终残项，独立复算六臂/机制，按规格十问追加不超过 80 行的中文终审节。
- 状态：`review`；交付完成，formal verdict=`request changes`。
- 分支/HEAD：`IP_branch` / `d8abd9a1e2b4bb3bc5289e4ec6383efed44c4018`；未提交/未 push。

## 完成内容与修改文件

- 追加 `PixelFlowICLR/Algorithm2/results/alg4_weighted_sigma_tau/codex_code_review.md`：
  唯一新节标题正确，共 71 行；完整文件 SHA-256
  `05cc71aa629ffa7f8696861249918ce0550d7832d28994ac17b1cfdf4b0959c9`。
- 新建/更新 Research OS：task、STATE v65、CURRENT、CONSTRAINTS 与本 handoff。
- 未修改采样器、验证脚本或实验结果；工作树原有其他修改均未触碰。

## 核验与结果

- `research-doctor --agents-only`：10/10。
- 当前 `utils.py` blob `51c1e478973c2df9c304df5a8f63ea93cf80d5b6` / SHA-256 `49b09c4c…`。
- B1：全 per-stage schedule 预检在首次 RNG draw 前，闭式 λ vs table、tol `1e-7`、失败 raise；
  运行时逐帧门保留。S3/N1 当前产物/注释通过。
- `lambda_schedule.csv`：80 行/27 列，全部闭式与 sqrt/fresh 列复算误差 0；dtype provenance 正确。
- 正式 precision 保存行 320/320 gate pass，最大 diff `8.53e-10`；但产物 19:02、入口预检源码
  19:04，不能证明“正式四种子已在新预检下重跑”。当前逻辑是 seed-independent，故确定性兼容。
- 去重后 960 frame、1920 inner；六臂均值/std/spread 与 CSV 完全一致。pooled spread：sigma
  −11.5%、precision −9.8%；spectral 为 −1.3%/约 0%。结论是 finite-horizon 路径与
  diversity/mixing 权衡，不是固定 conditional covariance 缩小。

## 协作与已知问题

- Codex 独立方案先固化，未读取本轮 Claude 方案。
- `research-peer plan`：Codex 三次、Claude 一次均因 Ceph
  `.git/research-os/peer/worktrees: Remote I/O error` 失败。可恢复 runtime 备份为
  `.git/research-os/peer.eio-20260827-1915` 后仍复现；无结构化 plan/compare artifact，未伪造。
- 未提交工作树且无远端 exact SHA，不满足 formal approve/Stop gate。

## 下一步

1. 用当前源码保存两模式 XCHK stdout 与 precision 两臂×四种子的入口预检通过收据。
2. Claude 完成结果级独立 compare；基础设施恢复后由 `research-peer` 生成结构化 artifact。
3. 提交并 push 实现/产物，随后 Codex 对远端精确 code SHA 做增量复审；通过后才改为 `approve`。
