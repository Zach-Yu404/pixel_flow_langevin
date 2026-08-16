# 当前状态（人类可读，保持 ≤1 页）

更新时间：2026-08-16

## 正在进行

- 当前无待审查增量。Issue #1 的 `onestep-mse-vs-t` 已完成两轮修正与最终复审。

## 2026-08-16 协作层修复记录

- 用户发现"GitHub 上看不到项目 & Codex 零参与"。根因三件：
  1. default branch 停在 `main`（初始骨架），工作全在 `IP_branch` → 已切 default=IP_branch。
     repo 名是 `pixel_flow_langevin`（非 MSFlow）——改名与否待用户定
  2. watcher 8月14日启动后即死：orchestrator 轮询裸奔 gh 调用，一次 TLS 超时整个进程崩
     → research-init 已修（watch 模式容错重试，commit 4a47632）并重启
  3. Issue/PR 从未创建过，label 状态机零触发 → Codex 自然零参与；另修掉
     `状态:review` 每轮重复调 codex 的烧 token bug（commit e881ef0）
- research-init 三个修复已推上游（含 doctor 的 BSD stat bug）

## 最近完成

- 2026-08-16：**Issue #1 review 完成并关闭——双 agent 回路首次完整闭环**：3 轮
  （request changes ×2 → approve），全程 watcher/orchestrator 自动调度。
  ①结论表述过强 ×2 + "bitwise identical" 证据越界 → 修正 526ca24（修正前逐项复算核实）；
  ②README 生成模板漏改（会回归旧口径）→ 修正 458d5af；③approve（模板与磁盘逐字节一致）。

- 2026-08-15：**实验 1b：可视化 + 结果瘦身**——每个 t 的一步恢复图合成 `onestep_predictions.mp4`
  （40 帧 = 4 stage × 10 t，帧内 [GT|x_t|WLS|Model] × 7 图 + 逐图 MSE，job 18568633）；
  145 张冗余曲线 PNG 合并为 1 张 `mse_vs_t_summary.png`（合并前逐位一致性校验）。
  结果目录收敛到 6 个文件；png/mp4 按仓库纪律 gitignored，见 artifacts.yaml
- 2026-08-15：**PixelFlowICLR 实验 1（onestep-mse-vs-t）完成**——one-step x̂₁ MSE vs t，
  5 任务 × 7 图 × 4 stage，A100 job 18567584。核心发现：Model(direct) 一致优于 WLS
  （stage0 差 ~50×，stage3 打平）；5 任务曲线因 kw 相同而完全一致（one-step 不经过 operator）。
  详见 experiments/2026-08-15-onestep-mse-vs-t.md；数据 PixelFlowICLR/results/onestep_mse/

- 2026-08-14：**全量记忆导入**——13 路并行读完项目内 130+ 份 .md(含前代 agent 的 SESSION_HANDOFF/global_memory)+ 关键代码/未提交 diff → 写入 experiments/{imagenet-5task-tuning, imagenet-baselines, multimodal-celeba-afhq-mri}.md、context/{authoritative-entries, contradictions-registry}.md、ARCHITECTURE/CONSTRAINTS 增补
- 2026-08-14：接入 Research OS（.research/ 初始化、GitHub label、watcher 启动、codex 装好并登录、dual-agent 就绪）
- 2026-07：旧机器 → CBIG 集群迁移完成；MRI prior checkpoint 恢复；`github_project_local/` 复现 benchmark 打包并验证（AFHQ SR bit-exact；ImageNet/CelebA batch-parity bit-exact；MRI 归一化契约验证）
- 2026-04~05（迁移前，见 git log 与 debug_IP4）：五任务 ImageNet 调参收敛（`memory_blur.md` 有 best config 与禁区）、MRI val_final_8 参考协议跑完、`IP_package/Experiments.md` 写成 paper 风格

## 下一步

- （待用户确认研究方向：例如跑 `--full` 复现、补 baseline 表格、写 paper draft、或继续某个任务的调参）
- 处理外层 repo 未提交状态（见"阻塞"第 2 条）

## 阻塞 / 需要用户决定

1. **in-flight 工作已从 diff 复原(机械意图)**：未提交改动 = random_inpainting **第二阶段 OAT sweep**(数据契约 .pt→PNG + stage2_configs + 断点续跑)、train.py 训练韧性(resume/epoch-ckpt/epoch-eval)、CFG 空类 token 泛化、legacy 采样配置开 CFG=2.0(详见 ARCHITECTURE"工作区未提交改动"节)。**待用户确认**:stage2 sweep 跑到哪一步了、动机实验是什么(MRI 先验训练?)
2. **外层 repo 工作区大量未提交变更**：`PixelFlow_train_code/` 整目录删除（−5226 行）、`IP_package/`、`debug_IP4/` 多数实验目录 untracked。是否按现状提交（大文件已被 .gitignore 排除）由用户决定
3. ~~codex CLI 待装/待登录~~ 已解决（2026-08-14）：v0.147.0 装于 `~/.local/bin/codex`，
   ChatGPT 账号已登录，5 个 MCP server 双侧注册，STATE 已切回 `dual-agent`
