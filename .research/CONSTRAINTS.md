# 用户约束（Authoritative）

> 规则：用户约束一出现立即写入本文件（附原始表达原文），并同步 STATE.yaml 相应开关。
> 任何 agent 不得基于旧约束继续工作。

## 长期约束

以下来自项目文档中的明确禁令（来源标注；非用户口头约束，但同等强制）：

### 复现公平性（来源：`github_project_local/LOCAL_REPRODUCTION.md`）
- **不得改 `batch_size`（=8）** 做跨 config 对比——seed 流位置靠它对齐
- seed 契约 all-seeds-42 不得破坏

### MRI（来源：`github_project_local/docs/mri_normalization_notes.md`）
- **MRI 数据已归一化，不得再归一化**

### 对比展示（来源：`github_project_local/README.md`）
- PSLD/ReSample 是 FFHQ 先验跑 ImageNet（OOD），**不得 naive 对比展示**
- DAPS unconditional vs 我们 class-conditional+CFG 的 prior 不匹配必须声明

### 仓库纪律（来源：research-init 共享规则 + 项目布局）
- 大文件（.pt/.ckpt/dataset）不进 git；绝对路径只进 `local.yaml`（gitignored）
- `github_project_local/` 是独立嵌套 repo，**不得** add 进外层 repo
- baseline 代码不 vendor

### 环境（来源：集群事实）
- 登录节点无 GPU；跑采样/训练必须走 Slurm

## 当前任务约束
（无）

<!-- 格式示例：
### 2026-08-13 · 关于 temporal-consistency 任务
【用户原始要求】
> 先不要实现
生效动作：STATE.yaml execution_allowed: false
-->
