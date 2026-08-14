# ImageNet baseline 对比战役(demo15 → 公平性审计 → DAPS 对齐重跑,2026-05~06,已完成)

> 要点摘录;权威文件:`IP_package/baselines/baseline_audit_report.md`、
> `baselines/results/updated_all_results/FINAL_REPORT.md`(2026-06-24,declared citable)、
> `metrics_table_100.md`、`baselines/results/memories/SESSION_HANDOFF*.md`。

## 最终可引用结论(DAPS 对齐 100 图,val 49000-49099)

- **ours ~2.5-3 dB 低于 DAPS PSNR(感知-失真取舍,非 bug)**:最干净对照是 gaussian(算子逐位一致)纯方法差 -2.5~-3.2 dB;best-shift/色彩校正已排除 eval 伪影;且 ours 有类条件+CFG=2.0 信息优势仍落后——论文按"感知向方法,~2 dB PSNR 换 LPIPS 竞争力/更优"框架呈现
- ours 在**两个 inpainting 任务上 SSIM+LPIPS 双赢** DAPS;SR/gaussian LPIPS 持平;**motion 是 DAPS 最大优势(-6~-7.7 dB)——如实呈现,不隐藏**
- DAPS-100 本地复现 = 论文 Table 3 ±0.2-1.1 dB(参考数字已立,勿重推)
- 数字表两套尺子并存**不得混用**:FINAL_REPORT(piq-VGG LPIPS, cca GT)vs metrics_table_100(AlexNet LPIPS, DAPS 用自己的 torchvision GT crop,差 ~1 dB)
- 非 DAPS 最强 baseline:DDRM SR 24.70;fps_smc random 27.26;PSLD/ReSample(FFHQ 人脸先验,OOD)全线垫底——**不得作为 prior-matched 呈现**

## demo15 审计要点(2026-06-22,63-agent 审计,53 断言对抗复核 46 确认)

原始 demo15 表**不公平**:3 套 metric 管线 / 3 把 LPIPS 尺子 / 3 种 GT crop;7/9 方法族自己重造 y;DDNM-gaussian 坏(6.2 dB,已修 DDNM+ ZERO=0.25 → ~23-25);DDNM/DDRM 不能做 motion(SVD 不可分核);DPS/FPS 存 min-max 归一化图;DDNM/DDRM/DiffPIR 跑了 2× 噪声;**我方 demo random-inpainting 曾是反转掩码 bug(70% observed 而非 missing)+ 无噪声 → 35 dB 是伪高,修复后 25.7 dB,旧数字永不引用**

## MMSE / 不确定性 / 数据一致性(demo15,2026-06-29 完成)

- MMSE = 11 个 sampler seed(42-52)逐像素均值;**y 必须跨 seed 固定**(在测量构造后、采样器前重播 seed)
- DC = mean|A(x)−y|,σ=0.05 任务噪声地板 0.040;DAPS 与 ours 最数据一致;MMSE 比单样本 +1.2~3.1 dB
- 减步跑的 DAPS-motion 18.12 / FPS-gaussian 13.30 是**减步伪影,不得当真实数字引用**
- 共享算子库 comparisons/shared_operators/;数据在 method_mmse_summary.json

## 运维事实(重跑时必读 SESSION_HANDOFF_100img)

- baseline runner 只认**方法专属**输出环境变量(DDNM_OUT/DDRM_OUT/DPS_OUT/FPS_OUT + MEAS_ROOT);用通用 OUT_ROOT 曾把 100 图结果写进 15-demo 目录,毁掉 UPD/ddrm 全部与 dps/fps 部分 recon(表 B 数字幸存于 all_baselines_metrics.md)
- DAPS 采样不得包 torch.no_grad;recon clip(0,1) 再进 piq;ToTensor 可能输出 1.0000001 要 clamp
- DAPS 必须对其自身 torchvision GT crop 打分,其余方法对 val100 cca GT
- mask 种子禁用 Python hash()(PYTHONHASHSEED 每进程随机)→ 用 zlib.crc32(已修)
- 100 图战役收尾状态(2026-06-29 handoff 时 DPS/FPS/PSLD/ReSample 过夜未完)→ 查 metrics_table_100.md 各方法 n 数;resample_100 当时 partial(gaussian n=34)
