# onestep-mse-vs-t

state: working · owner: claude · type: experiment

## 【用户原始要求】（2026-08-15，逐字）

> 基于现有 PixelFlow/MSFlow 代码，新建 `PixelFlowICLR`，完成 one-step (\hat{x}_1^k) prediction 的 MSE vs. (t) 实验。
>
> 测试 5 个任务：box_inpainting / random_inpainting / gaussian_blur / motion_blur / superresolution
>
> 每个任务直接使用其对应的 `LPIPS_king` config 参数。
>
> 测试图片使用：`/standard/CBIG-Standard-ECE/Zach/MSFlow/PixelFlow/IP_package/playground_runs` 中已有实验使用的 ImageNet GT 图片。
>
> 对每个 task、image、stage (k)：
> 1. 用 GT 得到真实 (x_1^k)。
> 2. 遍历该 stage 从开始到结束的所有 (t)，按现有代码构造 GT (x_t^k)。
> 3. 对同一个 (x_t^k) 分别只预测一次：WLS → (\hat{x}_{1,WLS}^k)；Model → (\hat{x}_{1,model}^k)
> 4. 计算二者相对于真实 (x_1^k) 的 MSE。
> 5. 画出 MSE 随 (t) 的曲线，WLS 和 Model 放在同一张图。
>
> 只做 one-step prediction，不做 Langevin、迭代优化或 ODE rollout。
> 尽量复用现有 scheduler、stage construction、WLS、model prediction 和 operator 代码，不要重新实现已有逻辑，也不要修改 `playground_runs`。
> 实验需要 GPU，请按服务器现有方式申请 GPU 后运行，不要直接在 login node 跑。
> 先检查代码确认实现位置，然后直接完成实验并保存曲线和 raw MSE 数据。
>
> （追加，实验准备期间）GPU可以申请A100

## 【Claude｜方案】

- 新目录 `PixelFlowICLR/`（repo 根下，与 PixelFlow 平级）：`onestep_mse_vs_t.py` + `run_onestep_mse.sbatch` + README。
- **全部数学复用** `IP_package/ms_posterior_sampling_article_version_final_utils`：
  `apply_H_tau`/`compute_sigma_tau`（x_t 构造）、`direct_estimate_x1`（=Model 预测）、
  `wls_estimate_x1`（=WLS 预测）、`make_velocity_fn`（模型+CFG 速度场）；
  `PixelFlowScheduler` 给 stage 边界 (s_k,e_k) 与 within-stage t 网格。
- **x1_gt^k 构造**：链式 bilinear 减半（`pixelflow/data_in1k.py:63-68` 的训练时语义）。
- **x_t^k 构造**：`H_t(x1_gt) + σ_t·ε`（采样端 line 634 同式）；每 (image,stage) 一份 ε
  （训练 collate 中 xs/xe 共享同一 noise 的语义），ε 种子 = crc32(image|stage)^42，跨任务相同
  → 同一 x_t 喂给各任务 config，任务间差异只来自 kw。
- 每个 t 只做**一次** velocity 调用 → 同时得到 WLS 与 Model 两个 x̂₁（无 Langevin/迭代/rollout）。
- 图片：playground_runs/*/meta.json "images" 的并集 = 7 张（5 任务同集）；
  loader 复用 `demo_runner.load_demo_images`（IP_package/demo + labels.json，含 class_idx）。
- box_inpainting 无裸 `LPIPS_king.json` → 用其唯一 LPIPS_king 变体 `LPIPS_king_srs1e-4__fd1_tr1.json`。
- GPU：Slurm `-p gpu --gres=gpu:a100:1 -A cbig-ece`（用户指定可用 A100）。
- 备注：one-step 预测本身不经过 operator/y（WLS 是 xs_hat/xe_hat 的加权最小二乘，与测量无关）；
  任务间差异体现在 LPIPS_king kw（步数/shift/guidance/rho 等）。已在 CPU self-test 验证全链路。
