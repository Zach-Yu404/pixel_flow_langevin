# 矛盾/陈旧登记表(2026-08-14 记忆导入时发现)

> 引用任何旧数字前先查这里。"→"后是裁决(依据文件内证据;标 ⚖️ 的需用户裁决)。

## 数字不可混用
1. **三把 LPIPS 尺子**:piq replace_pooling=True(历史 45 处调用,低 17-49%,任务相关不可换算)/ 官方 AlexNet(论文默认)/ 官方 VGG。→ 对外只报 alex(或对方用的变体);piqT 只做内部连续性
2. **两套 DAPS 打分**:FINAL_REPORT(cca GT+piq-VGG)vs metrics_table_100(DAPS 自身 torchvision GT+alex,差 ~1 dB)。→ 都对,但同一张表内不得混
3. rerun_imageNet 与 debug_IP4 的 val 数字**不逐位可比**(不同 seed-42 抽样);各任务 val/val2/val_ns0 也是不同的 500 图集
4. demo15 三种 GT crop 并存(measurements 的 squish 版/torchvision/cca)→ 每方法对自己重建目标的 GT 打分;新对比一律 cca
5. 减步跑的 baseline 数字(DAPS-motion 18.12、FPS-gaussian 13.30)是伪影,不作为其真实水平

## 已废弃数字(永不引用)
6. demo_runs 修复前 inpainting(~35 dB):反转掩码(70% observed)+ 无噪声。修复后 25.7 dB
7. DDNM-gaussian 6.21 dB:noiseless 路径 bug。DDNM+ 修复后 ~23-25
8. 旧 PSNR(MAX=1 bug):系统性低 6.02 dB;psnr_obs 分母 bug 高 0.7-2 dB
9. memory_blur.md 的旧宇宙禁区(ns=1、hx≥0.3)在新 λ 尺度已被推翻;其数字属旧 config 命名系

## 文件级陈旧(读时绕开)
10. `superresolution_4/WINNERS.md`/顶层 ranked_results:停在 S_hx02,真赢家在 refine_lreg2(M_lr80_L5)且从未过 val
11. Gaussian/Motion 各 val*/validation_summary.md **头部/标题是复制粘贴错误**(Motion 的标题写着 Gaussian);→ 信 metric_logic_audit + configs/*.json + metrics_all_runs.csv;val_ns0 的 audit 是 val2 原样拷贝(ns 值不实)
12. larger_box/val_box complete_configs/*.json 写 tr=0/fd=0,但 run_one.py 运行时强制 tr=1/fd=1(meta.json 为准)
13. IP_package/README demo 节仍写 3 图(实际 15+18);demo_runs metrics_all.csv 陈旧
14. debug5 的"参数调优已穷尽"结论早于 W1(he=1e-3)发现,已过时
15. 旧 handoff 说 MRI prior MISSING → **2026-07 已恢复**,以 CHECKPOINT_RESTORED.txt 为准
16. 所有 2026-06 及之前文档的绝对路径都是旧机器(/home/nvidia/...);`rerun_imageNet/operators.py` 内 `_daps_motion_kernel` **硬编码旧 DAPS 路径,use_daps_kernel=true 在本机会炸,需要重映射**(未修)

## 真正未和解的技术分歧 ⚖️
17. **g_bypass_stage3**:debug_IP 定为根因修复(shipping config =true),debug_IP2 用训练代码证据(每 stage 都 down-up)+ E2 实验反驳。两条 debug 线从未合并;当前所有赢家 config 均 =true——如做理论/论文表述需正面处理
18. **measurement_mode**:debug_IP2 论证 σ_n>0 时应用 'measure',但 debug_IP/IP4 的最终 JSON 都是 'call'+σ_n=0.05(config loader 只 warning)
19. **两种互不相容的有效参数域**:h_x=0.1/λ=50(debug_IP/IP4 线)vs h_x=0.02/λ=0.2(debug_IP2 线,并称对方病态)。现行主线是前者
20. box 的两个"最终最优"并存:exploration 王 F_cfg20_lr150_hxu02(2 图上 LPIPS 0.120,从未 N=500 验证)vs val_box 王(pareto_dual_king/LPIPS_king,N=500 已验证)。→ 引用后者;前者可作候选补验证
21. CelebA random LPIPS_king 的 srs=1e-5 vs 物理一致的 1e-4(PSNR_king 已用 1e-4)⚖️ 改否由用户定(重跑 5 组)
22. STATE.yaml 的 last_seen.claude 落后于 memory_version(onboarding 期写入痕迹)——无害,下次 sync 会刷新
