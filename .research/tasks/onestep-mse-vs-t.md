# onestep-mse-vs-t

state: done · owner: claude · type: experiment

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

## 【用户原始要求】（2026-08-16，逐字）

> 按 AGENTS.md 执行 sync protocol。先 gh issue view 1 读正文，确定 review 对象：关联 PR，或 Issue 里指定的 commit 区间（无 PR 时）。用 base+diff+受影响代码/tests 审查（不重读整仓）。结论以【Codex｜Review】中文评论发在 PR（无 PR 则发在 Issue #1）：approve 或列出必须修改项。若列出必须修改项，另外执行 gh issue edit 1 --add-label 触发:needs-fix。同步 .research/ 相关 task 文件并提交。

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

## 【Codex｜Review】（2026-08-16）

结论：request changes。审查范围为 Issue #1 指定的 `cba4e45..4167561`。

实现侧核对：`x1_gt` 链式 bilinear 金字塔、`x_t=H_t(x1)+sigma_t*eps`、WLS/direct estimate、
CFG 与 `g_bypass_stage3` 的调用均与受影响依赖实现一致；1400 行结果按完整实验键覆盖 280 组，
每组 5 个任务的 MSE 数值一致；9 个实际消费的 LPIPS_king 参数也一致。

必须修改：

1. `.research/experiments/2026-08-15-onestep-mse-vs-t.md:45-48` 的“一致优于”和“都单调下降”
   与 raw data 不符。stage 3 有 WLS 40 次胜出；stage 0 pooled WLS/Model 在首步均上升，逐图仅
   WLS 21/28、Model 22/28 曲线单调。需改成数据支持的限定表述。
2. `PixelFlowICLR/consolidate_results.py:53-56` 只比较两个标量 MSE，却宣称 predictions bitwise
   identical；`onestep_visual.py:15-18` 与实验记录沿用该表述。相同 MSE 不足以证明预测张量逐位
   相同。需把表述收紧为“MSE 数值逐位一致，且由相同消费参数/相同输入/未经过 operator 的代码
   路径推知任务无关”，或保存并比较预测 tensor/hash 后再声称 prediction bitwise identity。

验证：`py_compile` 通过；结果完整性/跨任务 MSE 一致性检查通过。当前受限执行环境运行 CPU
self-test 时进程未产出结果，未重复 GPU 实验。

## 【Codex｜Review】增量复审（2026-08-16）

范围：仅 `4167561..526ca24`。实验记录中的胜负范围、非严格单调表述以及现有结果 README/
docstring 的 prediction tensor 证据口径均已修正；从 `all_mse.csv` 独立复算得到 stage 0–2
Model 各 350/350、stage 3 Model 310/WLS 40，严格单调 WLS 21/28、Model 22/28，均与记录一致。

仍有一项必须修改：`PixelFlowICLR/consolidate_results.py:117-118` 的 README 生成模板仍写
“the 5 tasks are bitwise identical”，且未注明只比较了 MSE、未直接比较 prediction tensor。
该脚本重跑会覆盖已经修正的 `results/onestep_mse/README.md`，重新引入上一轮要求删除的证据越界。
请让生成模板与已提交 README 使用相同的“MSE rows bitwise identical + tensors not directly
compared”口径；修复后只需复审该增量。

## 【Codex｜Review】最终增量复审（2026-08-16）

结论：approve。范围严格为 `526ca24..458d5af`。唯一改动已将
`PixelFlowICLR/consolidate_results.py` 的 README 生成模板限定为“5 个任务的 MSE rows
逐位一致”，并明确 prediction tensors 未直接比较；模板与磁盘 README 逐行一致，不会在脚本
重跑时恢复旧的证据越界表述。

验证：`python -m py_compile PixelFlowICLR/consolidate_results.py` 通过；AST 提取的 README
模板与磁盘 README 长度均为 1031 字节且逐行无 diff；`git show --check 458d5af` 通过。
