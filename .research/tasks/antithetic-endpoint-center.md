# antithetic-endpoint-center（2026-08-26，done → 全部撤回）

> **撤回记录**：用户裁决「撤回修改，没有用」。utils.py / alg4_diag.py 逐字还原
> （回归 2,2,2,2 复现 0.1447/0.1143/0.1291/0.1495），antithetic_ablation.py 与
> results/alg4_antithetic_center_ablation/ 已删除。本文件保留全部数字与 Codex
> review 结论作为历史记录（含 0.0875 的结果与其代价/风险），避免重做。

## 用户原始要求（逐字摘录）

「请在当前 Algorithm 4 / clean-endpoint sampler 中实现并测试一个 antithetic
endpoint center 选项，用来减少 Block 2 的 ξ0 通过 x_τ→x̂₁ 被反复继承的问题。
只测试 box inpainting。保留现有 baseline……x_τ⁻ = 2H_τx1_in − x_τ……
x̂₁^anti = (x̂₁⁺+x̂₁⁻)/2……standard 必须逐位复现当前 baseline……oracle：
d± = B x1 ∓ (e−s)x0 下平均须还原 x1_in（dense float64）……」（全规格见对话）
后续指令：「让codex给你review一下」。

## 结果（box/junco，4 种子）

| 臂 | hole(4种子) | obs | PSNR | SSIM | spread | NFE | 纯耗时 |
|---|---|---|---|---|---|---|---|
| standard_sit2/pooled | 0.1499 | 0.0032 | 20.02 | 0.509 | 0.265 | 80 | 19.6s |
| antithetic_sit2/pooled | **0.0962** | 0.0028 | 21.80 | 0.537 | 0.138 | 160 | 38.1s |
| anti_one_shot/pooled | 0.0993 | 0.0031 | 21.59 | 0.523 | 0.158 | 80 | 19.0s |
| standard_sit2/spectral | 0.1411 | 0.0026 | 20.08 | 0.548 | 0.247 | 80 | 31.4s |
| **antithetic_sit2/spectral** | **0.0875** | **0.0022** | **22.29** | **0.581** | 0.115 | 160 | 50.1s |
| anti_one_shot/spectral | 0.1191 | 0.0023 | 20.48 | 0.570 | 0.209 | 80 | 25.0s |

**首次越过 Alg2 参照（0.102）**；四个 antithetic 臂 onset 全部 =None（late-stage
持续恶化消失），stage 2/3 首次变为净增益段。oracle：dense float64 恒等式
1.78e-15（解析：γ²σ[N²+γ²H²]⁻¹Hx0 泄漏项反对称抵消）。同算力（NFE=80）下
anti_one_shot 仍大幅优于 standard。改善位于 endpoint center（Δ_endpoint 增强、
Δ_Block1 注入不变——设计使然）；剩余误差转移至 Block-1 通道（ξ_h 为主）。
spread 减半（0.14/0.12）：非典型 collapse（obs/PSNR/SSIM 同步改善），但低于
Alg2 的 0.356，正式采用前需 §8.6 ker(A) 多样性审计。

## Codex review（对当前工作区代码）

A–E 全 PASS（standard 逐位性、antithetic 数学、随机流纪律、oracle、诊断只读
性——算法级零缺陷）；F/G 指出基础设施 must-fix 共 5 项，已修复 4 项（DONE
延迟到全种子后+原子写、config hash 拒绝混跑、analyze 完整矩阵断言、retime
纯净计时）；第 3 项（one_shot 与 sit2 逐帧共享随机数需索引式 RNG）与
"standard 逐位复现"硬约束冲突，记录为已知局限不改。review 全文存
results/alg4_antithetic_center_ablation/codex_review_transcript.txt。

## 语义（报告已明确）

Block 1/2 仍是各自高斯 conditional 的精确抽样；antithetic 只改代理均值中心
（一个经证明在精确速度下无偏、抵消 ξ₀ 一阶敏感度的更好高斯近似）；
Algorithm 4 本就非原始 posterior 的严格 exact sampler，antithetic 改变近似
质量而非其地位。

## 代码/产物

utils.py（endpoint_center_mode + on_inner in/out 状态拆分）、alg4_diag.py
（**extra_state）、antithetic_ablation.py（oracle/run/analyze/retime）；
results/alg4_antithetic_center_ablation/（report.md 十问全答 + 全 CSV + 每臂
montage/PNG/关键帧 npz）。推荐：**antithetic_sit2 + spectral S**（待多样性
审计与跨图/跨任务验证）。
