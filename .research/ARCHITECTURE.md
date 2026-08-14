# 架构事实

> 只记录**事实**（当前代码/方法确实如此），不记录提案。提案进 tasks/ 或 decisions/。
> 用相对路径引用代码，如 `src/solver.py:42`。

## 方法概览（PRINCIPLE, Algorithm 1）

多 stage（K=4）PixelFlow 逐分辨率生成；每个 stage 内每个时间步做 S 步 Langevin 内循环：

1. velocity prediction → 端点估计 xs_hat / xe_hat
2. **WLS + CG** 求干净图像估计 x1_hat（M = coeff·G²+coeff·I，SPD）
3. flow score `s_flow = (H·x1_hat − x_tau)/σ_safe²`
4. 联合梯度 g_x1（观测项 + prior 项）、g_eps
5. **预条件 ULA**：delta = CG((AᵀA/η²+λI)⁻¹, rhs)，更新 x1_k；eps_k 加噪更新
6. 重构 x_tau = H_t(x1_k) + σ_t·eps_k

细节/变量对照/调参：`PixelFlow/PRINCIPLE_MANUAL.md`（权威）。

## 代码结构

- `PixelFlow/ms_posterior_sampling_article_version.py` — 主采样脚本（PRINCIPLE 版）；`ms_posterior_sampling.py` 为旧版对照（主变量 latents、autograd ULA、无预条件）
- `PixelFlow/pixelflow/` — PixelFlow 模型/pipeline/scheduler（上游 fork）
- `PixelFlow/train.py` — 训练入口（MRI 等自训 prior 用）
- `PixelFlow/IP_package/` — 打包的 per-task 实验（tasks/{superresolution,gaussian_blur,motion_blur,random_inpainting,box_inpainting}/configs/*.json、runs、baselines/{DAPS,PSLD,…}）
- `PixelFlow/debug_IP4/` — per-task 调参工作区；`memory_blur.md` 汇总 blur 结论；`MRI/val_final_8/` 是 MRI 参考协议（其 dnnlib/torch_utils 是从 Sahil 的 fastmri 项目**复制**来的，避免跨用户依赖）
- `github_project_local/`（**独立嵌套 repo**）— 复现 benchmark：`src/` 不重新实现，只 re-export 已验证的 `IP_package/rerun_imageNet/{runner,operators,metrics,seeding}.py` 与 `ms_posterior_sampling_*` 采样器；`benchmark/run_{ours,afhq,celeba,mri}.py`

## 关键约定（最容易出 bug 的隐性假设）

- **Seed 契约**：全部 seed=42；每个 batch 的 measurement 与 sampler 调用前重置所有 RNG（`rerun_imageNet/seeding.py`）。**batch_size=8 固定不可改**——图像在 RNG 流中的位置靠它对齐，改了跨 config 公平性即失效
- **算子契约**：DAPS-aligned——SR `antialias:true`、motion 直接注入 DAPS `motionblur.Kernel`（seed 42）、box 用 DAPS `random_sq_bbox`、random inpainting 70% missing。见 `github_project_local/docs/operator_alignment_notes.md`
- **伴随算子**：A_kᵀ 用 `torch.autograd.grad` 精确转置（保证 AᵀA SPD、CG 收敛）；G=nearest_up(bilinear_down) 自伴随（误差~1e-7），不需要单独 Gᵀ
- **MRI 数据已归一化，不得再归一化**（`github_project_local/docs/mri_normalization_notes.md`）
- **model.pt 是 flat state_dict**（2.7GB），`strict=True` 加载；class-conditional 1000 类 + CFG guidance 2.0（DAPS 是 unconditional——对比时存在 prior 不匹配）
- **非确定性**：CUDA `upsample_bicubic2d_aa_backward` 非确定 → SR 单图 run-to-run ~0.5dB 浮动，100 图均值稳定；这是 kernel 属性不是 bug。截断 smoke 的均值 ≠ 100 图均值，要 per-image 对比
- `measurement_mode`: `"measure"`（y=A(gt)+σε，默认）与 `"call"`（无噪声，σ_n 必须 0）语义不同
- `sigma_tau_safe = max(σ_τ, 1e-4)` 只用于 score 的 1/σ² 除法；原始 σ_τ 用于乘法与重构
- **调参禁区**（跨任务，来自 `debug_IP4/memory_blur.md`）：`tr=1`、`ns=1`、`hx≥0.30`、`L=1`、`h_eps∈{1e-1,1e-4}` 都会大幅劣化
- 输出目录结构与 Langevin 日志信号（loss/delta_norm/grad_norm 判读）见 PRINCIPLE_MANUAL §8

## 环境事实

- conda env `pixelflow`（py3.12, torch 2.6.0+cu124, piq 0.8.0, lpips 0.1.4）——解释器与 dataset/checkpoint 的本机绝对路径一律见 `local.yaml`（gitignored）
- CBIG 集群登录节点**无 GPU**；采样/训练需 Slurm 申请（复现实验原硬件 A100-80GB）
- 2026-07 从旧机器迁移到 CBIG 集群；dataset root 注意**嵌套双层** `Zach_dataset/Zach_dataset`（见 `local.yaml`）；MRI prior 已恢复（`github_project_local/checkpoints/mri/CHECKPOINT_RESTORED.txt`）
