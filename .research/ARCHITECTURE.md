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

## Sampler 世代谱系(2026-08-14 导入)

- 三代主线:`ms_posterior_sampling.py`(legacy)→ `_article_version.py` → **`_article_version_final.py`(规范实现,rerun/benchmark 都 import 它的 `run_posterior_sampling`,绝不 fork)**
- debug fork:`debug_IP4/ms_sampler_v5.py` + `langevin_v5.py`(批量 import article utils,数学同源)加了 12 个 sweep 旋钮——**除 terminal_replace_weight 外全部实证死路**(见 sampler_diff_v5_vs_article.md);run_ip4 对 blur/SR 靠 monkey-patch make_Ak_fns,get_mask 返回全 1 占位
- 机制事实:warm_restart 每 ODE 步硬覆写 x1、eps 重推导 → 后验修正的跨步记忆**全部走 eps**;eps 熵是单调链(stage 1 是坍缩关键);掩码算子的 DPS 梯度在洞内恒 0;1/σ² 内层 floor=0.01(1e-3 会在 stage 3 尾部发散),外层 σ_τ<0.01 直接跳过 Langevin;warm-restart 的 x1 用 v-pred/direct 估计,不用 WLS(差 2-30 倍)
- Legacy 版 on-disk 已坏(DownUp_operation 手补 bug,stage 0 除零),不可作参考

## 度量约定(全项目,详见 rerun_imageNet/METRIC_AUDIT.md 与 contradictions-registry)

- piq PSNR/SSIM data_range=1.0 于 (x+1)/2 clamp 后的 [0,1];SSIM 为 RGB 平均(非 Y 通道);PSNR=80.0 表示逐位相同(EPS 上限)
- **三把 LPIPS 尺子**(piqT/alex/vgg):对外报 alex;piqT 只作历史连续性;metrics.py 三列并出
- FID 只信 pooled N=500,且仅作 ranking(N=100 有偏);metric 从内存 float 算,不从重读的 uint8 PNG
- 度量信任排序(inpainting):目测 > PSNR_unobs(偏爱平滑填充)> PSNR_all > HF 能量(mask 边界污染,不可信)

## 已知代码陷阱

- `rerun_imageNet/operators.py` `_daps_motion_kernel` 硬编码旧机器 DAPS 路径 → `use_daps_kernel=true` 在本机 import 即炸(**待修**)
- 可复现随机性禁用 Python `hash()`(PYTHONHASHSEED 每进程随机)→ 用 `zlib.crc32`
- bash 脚本禁用变量名 `GROUPS`(readonly 内建,赋值静默失败)
- `CUDA_VISIBLE_DEVICES=N` 与 `--gpu N` 不可同用(重映射为 cuda:0);8 GPU 同时起 B=8 c2img 会静默 SIGKILL(1 job/GPU)
- 批组成混淆:cuDNN 按 batch 形状选算法,长轨迹放大到 ~0.5-0.8 dB/图;逐位复现必须 batch parity
- pgrep/pkill 自匹配(waiter 命令行含 watched pattern)→ `[d]emo_runner` 括号技巧

## 工作区未提交改动的意图(2026-08-14 从 diff 读出)

- `train.py`(+93):--resume 全量恢复、epoch 级 checkpoint、rank-0 epoch-end FID/LPIPS/SSIM eval(try/except 包裹)、NCCL timeout 2h、崩溃写 error_rank{RANK}.log(残留:eval_status 张量未 all-reduce)
- `pipeline_pixelflow.py`:CFG 空类 token 改 `getattr(transformer,'num_classes',1000)`——服务非 1000 类模型(MRI/CelebA 先验)
- `ms_posterior_sampling.json`:guidance_scale 0→2.0
- `debug_IP4/random_inpainting/{prepare_data,run_chunk}.py`:OAT 数据契约 .pt→PNG(gts/masks/meta_data.json,~0.004 量化漂移可忽略),run_chunk 加 --configs_module(stage2_configs)与断点续跑守卫——**random_inpainting 第二阶段 OAT sweep 是当时的 in-flight 工作**

## 环境事实

- conda env `pixelflow`（py3.12, torch 2.6.0+cu124, piq 0.8.0, lpips 0.1.4）——解释器与 dataset/checkpoint 的本机绝对路径一律见 `local.yaml`（gitignored）
- CBIG 集群登录节点**无 GPU**；采样/训练需 Slurm 申请（复现实验原硬件 A100-80GB）
- **2026-08-21 本机（挂载服务器）无 gh、无 codex、无 GitHub 网络**；`git status` 全量扫描
  因 ceph `Remote I/O error` 稳定失败（`-uno` 可用）。详见 `context/facts.md`
- 2026-07 从旧机器迁移到 CBIG 集群；dataset root 注意**嵌套双层** `Zach_dataset/Zach_dataset`（见 `local.yaml`）；MRI prior 已恢复（`github_project_local/checkpoints/mri/CHECKPOINT_RESTORED.txt`）

## 草稿谱系与 PixelFlowICLR 代码映射（2026-08-21 导入）

四篇草稿，同一系列，都在 `PixelFlowICLR/Algorithm2/results/`：

| 草稿 | 文件 | Block 1 | Block 2 | 需搜索的连续量 |
|---|---|---|---|---|
| Alg.1 | 早期 draft p.12 | 预条件 Langevin（h₁） | Langevin（h₀） | 2 |
| Alg.2 | draft p.13 | 换元后**精确抽样** | Langevin（h₀） | 1 |
| Alg.3 | `modified.pdf`（"CLIMB-Flow"） | 同 Alg.2，改了求解式 | Langevin（h₀） | 1 |
| **Alg.4** | `algorithm.4pdf.pdf` | **精确抽样，中心是 x̂₁** | **直接重加噪，无步长** | **0** |

- Alg.1/2/3 都是 **interpolant-coupled**：通过 flat-prior 高斯 `N(x_τ;H_τx_1,σ_τ²I)` 耦合，
  learned marginal `p_τ` 留在 x_τ 的条件里 → 必须 Langevin → step size 与
  `√2 σ_τ H_τ⁻¹` 的先验平滑同源、同时存在。
- Alg.4 是 **clean-endpoint**：通过真实 denoising conditional 耦合，`p_τ` 相消，
  两者一起消失；代价是把 `p(x_1|x_τ)` 当高斯，需要其协方差
  `C⁻¹ = H_τᵀH_τ/σ_τ² + S_prior⁻¹`。精读见
  `references/2026-08-21-algorithm4-clean-endpoint-sampler.md`。
- **注意 `algorithm.4pdf.pdf` 内部把自己的 listing 编号为 "Algorithm 1"，
  且 Table 2 表头字面是 `Alg. 1 | Alg. 2 | Alg. 1`（排版 bug）。**

代码：

- `PixelFlowICLR/Algorithm1/main_alg1.py` (+`config_alg1.json`) — Alg.1，只写 Block 1，
  其余算子全部 import `Algorithm2/utils.py`
- `PixelFlowICLR/Algorithm2/utils.py` — **数学唯一库**。`apply_B`/`apply_N`/`apply_H_tau_inv`
  ↔ 草稿 (2)(3)/(p.20)；`score_solve` ↔ **(18) 且与 Alg.4 完全一致**；
  `make_M_tau`+`data_rhs` = interpolant-coupled 的 Block 1；
  `gamma2_meas.json` ↔ **(20) 的按 (k,τ) 表**（stage0 .0070–.0106、stage1 .0106–.0132、
  stage2 .0115–.0166、stage3 .0094–**.1448**）
- 三个采样器族（后两个在工作区未提交）：
  `run_posterior_sampling_alg2`（Alg.2 本体）、
  `run_posterior_wv_sampling_alg2`（velocity 作为对 x₁ 的第二个高斯观测，
  `M += NᵀN/(σ_τ²γ²)`）、
  `run_posterior_reg_sampling_alg2`（Tweedie anchor：`M += λP`、`b += λP·x1_model`；
  **`P=I` 时 `λI` 就是 Alg.4 的 `S_prior⁻¹`，`s²=1/λ`**）
- `PixelFlowICLR/Algorithm2/measurement.py` — 唯一改测量的地方；
  inpainting 的加噪/居中对齐 baseline 靠它，`demo_runner` 不动
- **符号冲突**：草稿 Alg.4 的 `S` = 先验协方差 surrogate；代码/报告里的 `S` =
  内迭代次数 `num_langevin`。一律写 `S_prior` / `S_iter`。
