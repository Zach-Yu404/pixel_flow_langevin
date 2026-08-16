# 2026-08-15-onestep-mse-vs-t

任务：onestep-mse-vs-t（见 tasks/onestep-mse-vs-t.md）
目的：量化 one-step x̂₁ᵏ 预测质量随 within-stage t 的变化，对比 WLS（wls_estimate_x1）
与 Model（direct_estimate_x1）两条路径，5 任务 × 7 图 × 4 stage，为 ICLR 分析图供数据。

## 复现信息
| 项 | 值 |
|---|---|
| commit | （PixelFlowICLR 新增文件，提交前记录；核心依赖代码未改动） |
| command | `sbatch PixelFlowICLR/run_onestep_mse.sbatch`（内部 `srun python onestep_mse_vs_t.py --out results/onestep_mse --chunk 4`） |
| config | 各任务 `IP_package/tasks/<task>/configs/LPIPS_king*.json`（box 用 `LPIPS_king_srs1e-4__fd1_tr1.json`）；消费字段：ode_steps_per_stage=10, shift=1.0, guidance_scale=2.0, rho_s/rho_e=1.0, lambda_x=0.01, cg_tol=1e-5, cg_max_iter=50, g_bypass_stage3=true |
| seed | ε 种子 = crc32(f"{image}\|stage{k}") ^ 42（每 image+stage 一份，跨任务相同；记录于脚本 `eps_for`） |
| Python / 关键包 | conda env pixelflow（py3.12, torch 2.6.0+cu124, diffusers 0.32.2） |
| CUDA / GPU | Slurm `-p gpu --gres=gpu:a100:1 -A cbig-ece`（A100，用户指定） |
| dataset ref | IP_package/demo（15 张 ImageNet val demo 集）之 playground 7 图子集：breastplate_armor, crane_structure, ibex_horns, junco, lakeside_beach, sea_anemone, shetland_sheepdog |
| checkpoint ref | c2img `IP_package/pretrained_models/c2img/model.pt`（→ PixelFlow/pretrained_models/c2img，2,706,601,240 B，见 artifacts） |

方法要点（与 task 文件一致）：x1_gt^k=链式 bilinear 减半（data_in1k 训练语义）；
x_t^k=H_t(x1_gt)+σ_t·ε（采样端同式，ε 每 image+stage 一份）；每 t 一次 velocity 调用
→ WLS 与 Model 两个 x̂₁；MSE 对 x1_gt^k。无 Langevin/迭代/rollout。
CPU self-test（无模型，零速度场）已通过：160 rows，stage 边界 s/e=[0,.25],[.143,.5],[.333,.75],[.6,1]。

## 运行记录
- job id：18567584（2026-08-15 提交，`-p gpu --gres=gpu:a100:1 -A cbig-ece`）

## 结果

Job 18567584 COMPLETED（A100, udc-an34-13, 6m40s）。1400 rows = 5×7×4×10 ✓ 全有限正值。
产物：`PixelFlowICLR/results/onestep_mse/`（all_mse.csv + 每任务 raw_mse.json + 28 曲线 + overview_mean.png）。

Pooled mean MSE（5 任务 × 7 图；t 为 within-stage）：

| stage | t=0 WLS / Model | t=0.56 WLS / Model | t=1.0 WLS / Model |
|---|---|---|---|
| 0 (32px) | 7.795 / 0.160 | 7.729 / 0.099 | 7.644 / 0.063 |
| 1 (64px) | 1.127 / 0.072 | 1.077 / 0.035 | 1.049 / 0.018 |
| 2 (128px) | 0.144 / 0.026 | 0.128 / 0.011 | 0.119 / 0.005 |
| 3 (256px) | 0.0095 / 0.0094 | 0.0038 / 0.0036 | 0.00002 / 0.00000 |

胜负计数（更低 MSE）：stage 0–2 Model 全胜（350/350 each）；stage 3 Model 310 / WLS 40。

## 结论与备注

1. **Model（direct_estimate_x1）一致优于 WLS（wls_estimate_x1）**，stage 0 差距近 50×，
   随 stage 递减，stage 3（G bypass → 两式几乎重合）打平。WLS 在小 s_k/e_k（早期 stage）
   严重 ill-conditioned，是其误差主源。
2. 两种预测的 MSE 都随 within-stage t 单调下降（噪声减小 → 预测更准），符合预期。
3. **5 个任务的曲线完全相同**（数字逐位一致）：one-step 预测不经过 operator/y；
   5 份 LPIPS_king 的相关 kw（steps=10/shift=1/gs=2/rho=1/cg 同）恰好一致，
   ε 按设计跨任务共享 → 任务维度在此实验中无区分度。如需任务分化，
   需引入测量一致性（operator 参与的预测）或各任务不同的 kw。
4. 修复过程记录：首跑输出被 demo_runner import 时的 os.chdir(IP_package) 副作用
   写到 IP_package/results/（已搬回 PixelFlowICLR/results/ 并在脚本里以 ORIG_CWD 修复；
   IP_package 现场已还原，playground_runs 全程未动）。

## 追加(2026-08-15 晚):1b 可视化 + 结果合并

- **onestep_predictions.mp4**(job 18568633,A100 udc-an34-13,~2.5min 渲染):40 帧 =
  4 stage × 10 t,每帧 4 行[GT x₁ᵏ | x_tᵏ | WLS x̂₁ | Model x̂₁]× 7 图,WLS/Model 行下标注
  逐图 MSE;2fps,20s,1462×892 H.264,4.0MB。脚本 `onestep_visual.py`(import 复用 1 的全部机制,
  只跑 box config——已验证 5 任务预测逐位一致,故标注 task-independent)。
  编码用 imageio-ffmpeg 静态二进制(`pip install imageio-ffmpeg` 进 pixelflow env;
  系统 /usr/bin/ffmpeg 缺 libvmaf.so.0 不可用)。
- **PNG 去冗余**(`consolidate_results.py`,先做逐位一致性断言,失败即中止不删):
  145 张曲线 PNG(5 任务 × 28 + 5 overview)→ **1 张** `mse_vs_t_summary.png`
  (4 stage 面板,逐图细线 + 均值粗线,log-y);5 份 raw_mse.json → 1 份 + task_configs.json
  (5 任务 kw 溯源)。最终目录 6 个文件(csv/json×3/png/mp4 + README)。
- 教训复刻:我自己核对一致性时先用了跨进程 `hash()` 比较——正是 registry #6 登记的
  PYTHONHASHSEED 陷阱,数值逐项对比才是对的。
- png/mp4 被仓库 .gitignore 全局排除(既有纪律),位置在 artifacts.yaml 登记;
  csv/json/脚本入库。
