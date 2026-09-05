# compute-gamma2-valset（2026-09-04 → 09-05，done：全量 50k，TF32）

## 用户原始要求（逐字）
「这个gamma是怎么计算的？」→「如果想计算整个validation set 50k张图片的gamma2得多久」→「目前有3个GPU可以用」
→（方案选择）「先抽样1w张没一类500张fp32，要计算全部的和每个class的gamma2」
解读：抽样 10k 张 = 每类 10 张（ImageNet val 每类仅 50 张，"500" 视为笔误），fp32，输出 all 与 per-class 两张表。

## 基准（A100-80GB，fp32，CFG）
每图每 τ 点前向：stage0 7 ms / stage1 28 ms / stage2 124 ms / stage3 681 ms；batch 1→32 仅快 3–8%。
全 50k×4 stage×10 τ ≈ 117 GPU-h（fp32）；TF32 ≈ 60 GPU-h（v 相对误差 3e-3，γ² 偏差 ~1e-5）；
bf16 快 5×但 v 误差 9%（不可用）。10k 抽样 fp32 ≈ 23 GPU-h → 3 卡 ≈ 8 h。

## 实现
gamma2_stats/compute_gamma2_stats.py（单文件）：数据管线与 compute_s_stats 逐字一致（cca、金字塔、label map）；
每类取按文件名排序的前 10 张；x₀ = eps_for(name, k)（与 main.py 同 crc32 种子方案）；
τ 网格 = 采样器 per-stage schedule（_stage_schedule，ode_steps_per_stage、shift）；CFG guidance 取 config
sampler_kw，class embedding 为图片本类；REAL G（eff_si=None）；fp32（matmul TF32 关）。
γ²(k,τ) = mean_images mean_pixels ‖v − (B x₁ − (e−s) x₀)‖²，另存 per-image 平方均值用于跨图 std。
3 GPU 按 class 原子 claim（gamma2_stats/_claims，1 h 陈旧回收）共享，每类一个 shards/<syn>.json，可断点续跑；
--merge 生成 gamma2_all.json（drop-in 替代 gamma2_meas_alg4.json 的 table 格式）、gamma2_labelled.json、gamma2_all.csv，
并打印与 7 图参考表的逐点对照。wrapper gamma2_stats/run_gpu.sh <gpu>（EIO 重试 200×60 s），日志 g2_gpu{0,1,2}.log。

## 改为全量（用户 2026-09-04）
「算了，直接5w张吧，fp32一起跑」→ 每类 50 张（全部 val），fp32；每类前向分 25 张一批（CFG 后 50，
stage3 峰值显存约 25 GB 留余量）；wrapper 重试预算 2000×60 s；每类 10 张的中间 shard 已删除（2 类）。
预计 117 GPU-h → 3 卡 ≈ 39 h（加 EIO 余量约 2 天）。

用户补充「你可以后续的尝试用更大的batch，另外可以用fp32」：fp32 本就是现口径；batch 基准显示 stage3
B=8→32 仅快 0.5%（卡已饱和），故不打断运行，脚本默认 chunk 25→50，EIO 自然重启时生效。
首类实测 426 s/50 张（=基准 8.4 s/图），ETA ≈ 39.5 h。

用户「我看你GPU memory还是没用满，增大batch size加快速度」→ 已重启为整类 50 张一次前向（说明：GPU 利用率
本就 100%，收益 <1%）。随后问 TF32 与 fp32 区别 →「那直接用tf32吧」：驱动默认开 matmul TF32
（G2_TF32=0 可回 fp32），每个 shard 记录 precision；切换前已完成的 74 个类为 fp32，merge 的 meta 中
按精度计数并注明（v 相对误差 3e-3，γ² 相对偏差 <1e-3，远小于 stage/τ 间差异）。TF32 后 ETA ≈ 20 h 总计。

进度 492/1000 时用户「GPU3也可以用上」→ 加第 4 个 shard（GPU 3，同 claim 机制自动分工）。别人的任务在共享
GPU 1/2，单类耗时 209→228–312 s。

## 结果（2026-09-05 完成；gamma2_stats/{gamma2_all.json, gamma2_labelled.json, gamma2_all.csv, analysis.md,
gamma2_vs_tau.png, perclass_hist_stage3_last.png}，shards/ 1000 个类文件）
1000 类 / 50000 张，74 类 fp32 + 926 类 TF32，总耗时约 16.5 h（4 卡，末段 GPU 3 加入）。

val 全表 vs 7 图旧表（γ²，val/ref 比）：
- stage 0：0.0153→0.0098（τ 0→0.999），ref 0.0100→0.0070，比 1.54→1.40；
- stage 1：0.0193→0.0143，ref 0.0132→0.0106，比 1.46→1.35；
- stage 2：0.0153→0.0208，ref 0.0116→0.0166，比 1.32→1.25；
- stage 3：0.0118→0.1437，ref 0.0094→0.1420，比 1.26（τ=0）→ 0.96（0.888）→ 1.01（0.999）。
即 7 张 demo 图对网络来说比典型 val 图"容易" 25–55%（velocity 误差更小），只有 stage 3 晚期（σ_τ→0，
误差被 1/σ 放大的区域）两者一致。跨图 std 与均值同量级（stage 0 τ=0：0.0085 vs 0.0153）。

per-class 离散：stage 3 τ=0.999 中位 0.1446、CV 0.04（5–95%：0.133–0.152，最高 n01924916 0.158，
最低 n03924679 0.114）；τ=0 中位 0.0114、CV 0.36。早期 τ 的类间差异大（×2），晚期趋同。

精度交叉核对（xcheck_tf32/README.md）：3 个 fp32 类用 TF32 重算，同类同点对比 120 个 (stage,τ) 点：
相对差均值 +6.6e-6、最大 7.2e-5 —— 与预期 (3e-3)²≈1e-5 一致，fp32/TF32 混表可视为同一口径。
（附带修了 G2_SHARDS 相对路径 bug：utils import 会 chdir 到 IP_package，现改为 abspath。）

## 状态
done（2026-09-05）。gamma2_all.json 可直接替换 gamma2_meas_alg4.json（同 table 格式）；是否切换由用户决定
（会改变 (19)/(22) 的 γ² 权重，主要影响 stage 0–2 与 stage 3 早期，+25–55%）。
