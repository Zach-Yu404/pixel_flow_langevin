# x0carry-ablation（2026-08-25，done → 全部退回）

## 用户原始要求（逐字摘录）

「Implement an additional x0 carry option for Algorithm 4. Current baseline
(default) remains unchanged: x0 = xi_0 … endpoint-derived x0 … Option A:
endpoint_raw … x0_hat = N^{-1} z … Option B: endpoint_gamma … (N^2 + gamma^2
H_tau^2) x0_hat = N z … Correctness check … d_exact = B x1_out - (e-s) xi_0 …
endpoint_raw must reconstruct x0_hat == xi_0 …」（完整规格见对话记录）

裁决（逐字）：「都给退回，这个是没用的」

## 负结果（保留此记录以免重做；代码与结果已按用户指令全部删除）

box/junco、S_it=2、per-τ γ²、pooled_junco+spectral 两 S、4 种子：

| carry | pooled hole | spectral hole |
|---|---|---|
| sampled（默认） | **0.1495** | **0.1490** |
| endpoint_raw | 0.2615 | 0.3118 |
| endpoint_gamma | 0.2442 | 0.3014 |

- oracle 通过（d_exact → endpoint_raw 还原 ξ₀ 至 9.5e-7），实现无 bug——差是真差。
- 机制：端点估计按构造欠散布（‖x0‖²/n 在 stage 3 塌到 0.69–0.79），且以相关的
  网络误差替换独立白噪声；经 l.8 的 x_τ=Hx₁+σx0 进入后被 b<1 反卷积固化
  （Block-1 注入/移除不变——损伤不走 Block 1）。γ² 正则只救 6–7%。
  spread 涨 60% 是误差方差不是多样性。额外代价每帧 +1 NFE（80→120）。
- **结论：listing 的 l.16（沿用同一 ξ₀ 坐标）是承重结构，任何确定性 x0 估计都有害。**

## 退回执行记录

utils.py 恢复至改动前逐字原样（l.16、签名、行键、apply_N_inv 全部移除；
回归 2,2,2,2 逐位复现 0.1447/0.1143/0.1291/0.1495）；x0carry_ablation.py 与
results/alg4_x0carry_ablation/ 已删除；临时脚本已清。
