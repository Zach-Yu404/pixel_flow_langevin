# 当前状态（人类可读，保持 ≤1 页）

更新时间：2026-08-18（晚）

## 正在进行

- Issue #3 Codex review（8 条 request changes）**已全部处理**：5 修 + 2 降级 + 1 条按
  scope 约束在 Algorithm2 内解决（hash 种子改入口 fail-fast，不动 IP_package）。
  随后自跑三路对抗审计，发现并修掉修正本身引入的 3 个缺陷（V9 静默破坏 V8 的 γ² 探针、
  `--config config.json` 仍走默认分支、operator/kw 子字典未校验）。
  commits `fd57a2a` / `24907e6` / `ffe344f` / `a8ac308`。
  **待办：本机无 GitHub 凭证，需在集群侧发
  `.research/handoffs/2026-08-18-issue3-fix-response.md` 到 Issue #3 并切回 `状态:review`。**
- **新证据：洞区 MSE 的 seed 地板 = 5.3%**（n=4，sd 0.0237）。所有 <5% 的洞区差异结论作废
  （含"γ² 影响 ~1%"）；h₀/K/anchor 的 2–6 倍效应仍有效。

## 2026-08-18（晚）：新挂载机路径兼容（用户指令）

- 环境：同一 ceph 盘挂到 `/CBIG-Standard-ECE/`（旧前缀 `/sfs/ceph/standard/`、`/standard/`
  在本机不存在）；**无 Slurm**，本地 4×A100 直跑；HOME 不共享（`/home2/rqx6rs`，本机无 gh）。
- 真正断点是**绝对符号链接**（共享盘上存的是旧机写的 `/sfs/...` 目标）：74 个已改为相对目标，
  一次修改两机通用。致命项 `IP_package/pixelflow`（模型包本身）此前在新机是死链。
- 代码侧加 `SHARE_ROOTS` + `resolve_path()`（存在即原样返回 → 旧机行为零变化；否则按候选根
  依次重试；全失败列出所有尝试过的路径）。
- 端到端验证：新机 GPU 复现集群 trw 实验，同到 8 位有效数字（非逐位——两机 env 独立安装）。
  详见 `tasks/path-compat-mounted-server.md`。

## 2026-08-18（Codex review Issue #3）

- 审查区间 `b0c9351..6a5a927`。CPU V1–V8 与 one-step self-test 可运行并通过，但未覆盖
  τ=0 抽样协方差、config 值迁移、post-projection 落盘和 fixed-score K 扩展等关键风险。
- 必改主项：τ=0 ridge 随机 RHS 漏 `sqrt(epsilon)` 噪声；K>1 复用冻结 score；active anchor
  配置是 silent no-op；full_ip 丢弃 post-projection x1；相对 out 会写到调用者 CWD；缺严格
  config/seed 契约测试；oracle/anchor/K 结论需补可复算与公平对照或降级口径。
- 已提交 CSV 反证“bit-exact/逐位一致”：trw 两次独立运行最大轨迹指标差约 `1.05e-5`；
  f9299d7 与 6a5a927 末端 hole/obs 也仅四位小数一致。详见
  `tasks/review-alg2-debug-directory.md` 与同日 handoff。

## 2026-08-18（晚）：GPU 终验闭环 + trw 实验 + 协作层二次修复

- **cg_max_iter_l14 回归**：终验 18650178 得 hole 0.9704≠0.9691 → 根因=参数内联把 τ=0 的
  l.14 CG 上限从硬编码 200 写成 config 里的 50；修复后 job 18651717 复现到四位小数
  0.9691/0.0018（非 bit-exact），终验数值判据通过。同批修回 box/random trw=1.0。
- **terminal_replace_weight 0 vs 1.0 实测**（用户指令，job 18651908）：两次独立运行的
  pre-projection 轨迹数值接近但非逐位一致；trw=1 洞区不变 0.9691、观测区 post obs=**0.0**。
  顺带发现 demo 管线
  inpainting 的 y **无加性噪声**（η-模型失配）。详见 tasks/test-terminal-replace-weight.md
- **协作层二次修复**：Issue #3 首次 codex 派发静默失败（exit 0 无评论）+ 标签被误置
  状态:done——三个缺陷：①错误评论原文引用 marker 导致后续子串校验误判 ②疑似双 watcher
  竞态 ③watcher 在 login node 反复被清（846355 已死，日志无痕）。修复=orchestrator 补
  输出留痕/行首 marker 校验/单实例锁 + **watcher 迁移 scrontab**（计算节点已验证 gh/codex 可用）。

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

- 2026-08-18~19：**Algorithm2 目录治理四连（原执行记录；Issue #3 复审待修）**——results 170M→18M 只留关键证据；
  main 剥离 WLS/Model（纯 Alg2）；config.json 成为唯一配置源（4 模式 JSON 合一 → 全参数内联，
  路径指针废除）；用户抓出 projection（terminal_replace_weight box/random=1.0）遗漏已补；
  S2 stage-1 漂移判决=hash 种子调用失误（契约 PYTHONHASHSEED=0）。GPU 指标仅四位小数一致，
  并非 bit-exact；其余必改项见本页顶部。commits f9299d7/e3f23b9。
  GPU 终验 job 18650178 被 A100 维护窗口挂起（判据 hole 0.9691）。
  详见 tasks/cleanup-algorithm2-directory.md（含补记说明）

- 2026-08-17：**debug-box-alg2-hole 原调试阶段结论（Issue #3 复审待修）**——固定 demo/seed 下
  Tweedie anchor 的已存 CSV 显示洞区 MSE 池化 1.097→0.178（6.2×），但 oracle 目标、K>1
  实现、RNG 配对和多 seed 证据均不足，暂不能据此确认“结构性缺陷/唯一有效干预”。详见
  tasks/debug-box-alg2-hole.md 与 review-alg2-debug-directory.md。

- 2026-08-16：**Issue #2 re-review 通过**：Algorithm 2 one-step 估计器实验的 3 处证据口径
  已修正；Codex 增量复审确认 10.94%/19.60% 可复算，结论 approve。
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

- 修复 Issue #3 的必须修改项后请求 Codex 只复审增量。
- （待用户确认研究方向：例如跑 `--full` 复现、补 baseline 表格、写 paper draft、或继续某个任务的调参）
- 处理外层 repo 未提交状态（见"阻塞"第 2 条）

## 阻塞 / 需要用户决定

1. **in-flight 工作已从 diff 复原(机械意图)**：未提交改动 = random_inpainting **第二阶段 OAT sweep**(数据契约 .pt→PNG + stage2_configs + 断点续跑)、train.py 训练韧性(resume/epoch-ckpt/epoch-eval)、CFG 空类 token 泛化、legacy 采样配置开 CFG=2.0(详见 ARCHITECTURE"工作区未提交改动"节)。**待用户确认**:stage2 sweep 跑到哪一步了、动机实验是什么(MRI 先验训练?)
2. **外层 repo 工作区大量未提交变更**：`PixelFlow_train_code/` 整目录删除（−5226 行）、`IP_package/`、`debug_IP4/` 多数实验目录 untracked。是否按现状提交（大文件已被 .gitignore 排除）由用户决定
3. ~~codex CLI 待装/待登录~~ 已解决（2026-08-14）：v0.147.0 装于 `~/.local/bin/codex`，
   ChatGPT 账号已登录，5 个 MCP server 双侧注册，STATE 已切回 `dual-agent`
