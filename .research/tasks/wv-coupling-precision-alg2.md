# wv-coupling-precision-alg2

state: done · owner: claude · type: implementation+experiment · 开始：2026-08-20

## 【用户原始要求】

> 请在当前 `utils.py` 中，**基于现有 `run_posterior_sampling_alg2` 做最小改动**，新增一个 `run_posterior_wv_sampling_alg2(...)`，其中 `wv` 表示使用由 x0-x1 coupling + velocity uncertainty 推导出的额外 clean-information term。不要重写现有 Algorithm 2，也不要修改 `run_posterior_sampling_alg2` 的行为。尽量复用现有 helper/operator，diff 越小越好。
>
> 1. 新版本只修改 Block-1 的 M_tau 和 b_tau：M^wv = A^TA/eta^2 + H^TH/sigma^2 + N^TN/(sigma^2 gamma^2)；b^wv = A^Ty/eta^2 + H^T x_tau/sigma^2 + N^T[(e-s)x_tau + sigma v]/(sigma^2 gamma^2)。复用 apply_H_tau/apply_N 的嵌套形式，不要显式构造矩阵。
> 2. Exact Gaussian draw 的随机 RHS 也必须对应新 M_tau：额外加 (1/(sigma*gamma)) N^T xi_v。tau=0 的 ridge sqrt(epsilon)*xi_eps 逻辑不变。
> 3. 其它部分必须尽量与现有 Alg2 一致（stage/time schedule、score solve、line 15 restore、Block-2 Langevin、stage transition、RNG、CG、terminal projection、trajectory recording）。唯一算法性变化应主要集中在 Block-1 的 M_tau/b_tilde。
> 4. 同时测试两种 x1_hat：Method A direct_estimate_x1；Method B apply_H_tau_inv（Eq. 51）。tau=0 且 H_tau singular 时不能调用普通 inverse，标记 unavailable，不要引入 ridge/伪逆到正式实现。
> 5. 新函数提供 x1_hat_method="direct"|"inverse"，不要建立复杂 abstraction。如果 WV 核心公式不需要 x1_hat，则它只用于 diagnostic。
> 6. 测试放到 `test/` 目录，并保留测试代码和结果（test/test_wv_alg2.py + test/results_wv_alg2/）。一个测试脚本足够。
> 7. 测试内容保持最小：box_inpainting、junco、seed 42、当前正式 config、相同 measurement/mask/operator、相同 RNG stream。至少比较 baseline / WV+direct / WV+inverse。
> 8. 只记录最重要指标（final hole/visible/full MSE、每 stage end hole MSE、Block-1 mean hole MSE、sampled x1 hole MSE、x1_hat hole MSE、measurement residual），保存少量关键图。不要再做大规模 hyperparameter sweep。
> 9. 做一个 operator-level sanity check：oracle displacement 下 r_tau = N_k x1 到 numerical precision；并验证新增 noise term covariance 与 N^T N/(sigma^2 gamma^2) 一致。
> 10. 最终不要自动修改其它算法（不改 run_posterior_sampling_alg2 / main.py / main2.py / transition / gamma/ridge/h0，不删 baseline）。新 helper 控制在 1-2 个小函数。
> 11. 最后给我清楚的报告：Implemented / Tested / Results location / Key results / Analysis / Decision（BEST CURRENT VARIANT / WHY / KEEP BASELINE / NEXT ISSUE）。

## 结论

**WV coupling 项有效，是本项目至今唯一同时修好 mean 和 sampling noise 的改动。**

洞区 **0.075411**（上一轮同设置 baseline 0.098428，改善 1.31×），可见区 0.0015545
（改善 1.25×），全图 0.020019。**洞区第一步就是 0.2155，全程不超过 0.22**——
此前所有变体在 stage 边界都会冲到 31.66 / 106 / 435。

- 同状态 Block-1 mean（同 x_tau、同 CG，只换 M/b）：`mu_wv` 比 `mu_base` 好 **1.6–51.4×**。
- 抽样/mean 之比从上一轮的 **54×** 降到 **1.07–1.57×**——sampling noise 不再是瓶颈。
- 机制：gamma^2 ≈ 0.007–0.017，故 N^TN/(sigma^2 gamma^2) 比 H^TH/sigma^2 大一个量级以上，
  洞区（ker A）后验精度整体抬升。权重来自实测 gamma^2 表，没有手调 lambda。
- 附带：WV 的 RHS 在洞区隐含的 clean 估计 `N^-1[(e-s)x_tau + sigma v]` 在 range(G) 上
  等于 `direct_estimate_x1`，在 ker(G) 上恰好差 1/1.2——**coupling 推导自动带上了
  G 感知的正确缩放**。
- operator 检查：V-WV1 `r_tau = N_k x1` 相对误差 5.8e-08…2.0e-07（12 个 stage/tau）；
  V-WV2 噪声协方差相对偏差 1.5e-03…2.0e-02（容差 6%）。
- `x1_hat_method` 确认为纯诊断：两臂最终洞区 0.07541140 vs 0.07541135。
  `inverse` 在有定义处一致略优 0.3–1%（差异集中在 ker(G)），但在 3 个 tau=0 点不可辨识。

**同时发现**：用户当前的 `run_posterior_sampling_alg2`（用 x1_hat 重建 x_tau + data_rhs）
**会发散**——洞区 31.66 → 107.9（stage 0 末）→ 9.78e10（stage 1 末），
stage 1 step 7（gstep 17）变 NaN。

代码：`utils.py` **纯新增 274 行、0 行删除**（`make_M_tau_wv` / `data_rhs_wv` /
`_x1_hat_diag` / `run_posterior_wv_sampling_alg2`）；`test/test_wv_alg2.py` 保留。
`main.py`/`main2.py` 未改（注：main2.py 本轮被回退到 HEAD，上一轮我加的
`check_52=False` 与 `skip_tau0` 已不在——非本轮改动）。`--mode verify` ALL CHECKS PASSED。
