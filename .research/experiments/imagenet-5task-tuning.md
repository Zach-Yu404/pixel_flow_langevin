# ImageNet-256 五任务调参战役(2026-04~05,已完成)

> 要点摘录;原文件是权威。入口索引见 `.research/context/authoritative-entries.md`。
> 全部基于 debug_IP4 的共享 sampler `ms_sampler_v5.run_ip4`(monkey-patch 各任务算子)。
> 验证协议:5 组不相交 × 100 张 ImageNet val,组内所有 config 见到相同 y/mask/init,rank-sum(PSNR/SSIM/LPIPS/pooled FID)选优。
> ⚠️ 这些 LPIPS 均为 piq replace_pooling=True(比 canonical 低 17-49%,不能对外引用,见 CONSTRAINTS)。

## 各任务验证赢家(N=500)

| 任务 | 最优(rank-sum) | 关键数字 | 感知向备选 |
|---|---|---|---|
| box inpainting 128² | pareto_dual_king(he=0.01,L=15,srs=1e-4,ns=1,tr=1,fd=1) | 20.00/0.793/0.160/FID66 | LPIPS_king(he=1e-3,L=10):LPIPS 0.156, FID 52.4(强信号);balanced_perceptual 被 Pareto 支配,**别选** |
| random inpainting 70% | PSNR_king(he=0.01,L=10,srs=1e-4,**ns=0**,tr=1,fd=0) | 27.51/0.853/0.132/FID37 | LPIPS_king(he=1e-3,L=15,srs=1e-5):0.124/FID36 |
| gaussian deblur σ3 | (2-2 平手,标称 PSNR_king)ns1_lr300_he0.01_hx0.2_srs1e-4 | 23.58/0.540/0.346/FID85 | ns1_lr350_he0.001:LPIPS 0.304/FID59.6;**val_ns0 再跑**:he=0.001/λ350/hx0.2 @ns=0 是全能王 24.34/0.291/FID90 |
| motion deblur i0.5 | (三方平手,标称 PSNR_king)ns1_lr250_he0.01_hx0.2_srs1e-5 | 22.57/0.527/0.356/FID88 | LPIPS_king ns1_lr300_he0.001:0.323/FID56;@ns=0 时 LPIPS_king 胜(23.36/0.308) |
| SR ×4 | SSIM_king(λ350,he=0.01,hx0.2,srs1e-5,ns=1,tr=0) | 21.69/0.542/0.333/FID71 | LPIPS_king(he=1e-3):0.316/FID64.4;另一条 guidance=0 线的 best_lpips:0.3045(superresolution_4/val) |

## 跨任务规律(机制级,见 debug_IP4/exploration/ 与各 audit)

- **h_epsilon 是感知-保真主轴**:1e-3 = 感知/LPIPS 路线(保 eps 熵→生成语义内容),1e-2 = PSNR 路线;≥5e-2 与 1e-4 都是禁区
- **noise_scale 按任务几何分裂**:box=1(多样性)、random=0(双指标严胜)、MRI=0、blur/SR 两条路线(ns=1 buy FID, ns=0 buy PSNR);**config 不得跨任务迁移**
- **terminal_replace**:inpainting 必须 =1(免费投影,最大单杠杆);blur/SR 必须 =0(回注退化,SR@tr=1 = bicubic 基线)
- **motion 伴随算子必须 flip(K)**(非对称核;用对称捷径悄悄损 2-3 dB)
- **warm_restart=True 与 g_bypass_stage3=True 永不关闭**(关 warm_restart:PSNR 14/LPIPS 0.25)
- eps 熵是单调链:低 h_eps 必须从 stage 0 起用;stage 1(64²)是坍缩关键段;T012=[LO,LO,LO,HI] 省 25% 算力等效果
- λ_reg U 型,λ_prox 必须与 λ_reg 同步(锁定约定);λ_x 是死旋钮
- 一次只拉一根激进杠杆;两根以上必然 HF 过冲碎裂
- OAT 消融(146 configs,37.8 GPU-h,`IP_package/Experiments.md`):杠杆排序 λ_reg > h_eps > h_x > S;S≥5 即平台
- 小集 sweep 数字不外推到 N=500(SR: sweep LPIPS 0.22 → val 0.30);sweep 只用于排序

## 早期 debug 链(debug_IP → IP2 → IP3 → debug5 → IP4,机制发现史)

- 原 article 版失败根因:h_x=1e-3+λ=0.01 默认值太保守 + 冷启动;修复集 h_x=0.1/λ=50/warm_restart/g_bypass
- debug_IP2 反证:PixelFlow 训练在每个 stage 都有 down-up(data_in1k.py:58-74),G=I@stage3 让残差更差——**与 g_bypass_stage3=true 从未和解**(shipping config 仍开)
- warm-restart 吞掉一切 in-step 干预(DPS kick/替换/Tikhonov 全等效);掩码算子的 DPS 梯度在洞内恒为 0
- WINNER3(he=1e-3 全局 + h_x=[.1,.1,.1,.7] + tr=1)= 语义内容里程碑("盒子里有鸟")
- 旧 PSNR bug:MAX=1 用于 [-1,1] 数据(系统性低 6.02 dB)、psnr_obs 分母错;旧数字引用需换算
- metric 信任排序:目测 > PSNR_unobs(偏爱平滑填充)> PSNR_all > HF 能量(不可信,被 mask 边界污染)

## 悬而未决 / 陷阱

- superresolution_4 的 WINNERS.md/ranked_results 是**陈旧的**(真赢家 M_lr80_L5 在 refine_lreg2,从未进 val)
- Gaussian/Motion 的 validation_summary.md 头部是复制粘贴错误(标题/参数与实际不符)——**只信 metric_logic_audit + configs/*.json + metrics_all_runs.csv**
- memory_blur.md 是旧机器旧 config 宇宙(S_*/C_*/R_* 命名),其"禁区"(ns=1、hx≥0.3)在新 λ 尺度下已被推翻;只作 legacy 参考
- gaussian val2 只有 2 个独特 config(用户请求重复了一个);第三 config 可补 5 跑
