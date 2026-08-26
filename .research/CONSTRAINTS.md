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

### 度量与引用(来源:rerun_imageNet/METRIC_AUDIT.md、celeba METRIC_AUDIT.md、baseline_audit_report.md)
- 跨论文 LPIPS 必须同变体(默认官方 AlexNet);**piq replace_pooling=True 的数字永不对外引用**
- FID:N=100 永不当绝对数;只信 pooled N=500 且仅作 ranking
- 永不引用:修复前 demo inpainting(~35 dB)、DDNM-gaussian 6.21、减步 baseline 伪影、MAX=1 时代的 PSNR
- DAPS 对其自身 torchvision GT 打分;两套表(piq-VGG vs alex)不得混

### 采样器纪律(来源:DESIGN.md、sampler_diff、debug_IP4)
- 永不 fork `run_posterior_sampling` / 算子——import 之(fork 即失去对齐保证)
- warm_restart=True 不关;λ_prox 与 λ_reg 同步动
- (2026-08-24 用户决定)apply_G 的 stage-3 恒等旁路已删除:G 在所有 stage 都是 down-up 投影;
  g_bypass_stage3 开关成为无效 no-op,gamma2 表按新口径重测(junco 单图)
- (2026-08-24 用户决定)gamma2 表分表:gamma2_meas.json = junco 单图/eps_for(Alg2/config.json 专用);
  gamma2_meas_alg4.json = Alg4 专用(config_alg4.json 指向它,stage-3 为 7 图/随机噪声口径)。
  两个会话不得写对方的表——此前 stage-3 行曾被两会话互相覆写
- config 不跨任务迁移(ns/tr/fd/srs 全是任务相关;motion 伴随必须 flip(K))
- inpainting tr=1;blur/SR tr=0(SR@tr=1 = bicubic,不是算法结果)
- MMSE 类实验:y 跨 sampler seed 固定(测量后、采样前重播种)

### 运维(来源:SESSION_HANDOFF*、INCIDENT_NOTES、random_memory)
- baseline runner 用方法专属 OUT env var(DDNM_OUT 等),通用 OUT_ROOT 会静默 clobber 别人的结果目录
- 永不 `cp -al` 硬链接实验数据目录(2026-05-03 事故:smoke 测试穿透覆写源数据)
- 可复现随机性禁用 `hash()`;bash 禁用变量名 `GROUPS`
- DAPS 采样不包 no_grad;piq 前 clip(0,1)
- celeba_results/ 根会被外部进程重置,持久物只放 code/ 或数据盘符号链接

## 当前任务约束

### 2026-08-15 · onestep-mse-vs-t（PixelFlowICLR 实验 1）
【用户原始要求】（节选，全文见 tasks/onestep-mse-vs-t.md）
> 只做 one-step prediction，不做 Langevin、迭代优化或 ODE rollout。
> 尽量复用现有 scheduler、stage construction、WLS、model prediction 和 operator 代码，
> 不要重新实现已有逻辑，也不要修改 `playground_runs`。
> 实验需要 GPU，请按服务器现有方式申请 GPU 后运行，不要直接在 login node 跑。
> GPU可以申请A100

生效动作：`playground_runs/` 只读；实验代码全部 import 现有实现；GPU 经 Slurm `gpu` 分区
`--gres=gpu:a100:1` 申请；login node 仅允许 CPU self-test（无模型前向）。

<!-- 格式示例：
### 2026-08-13 · 关于 temporal-consistency 任务
【用户原始要求】
> 先不要实现
生效动作：STATE.yaml execution_allowed: false
-->
