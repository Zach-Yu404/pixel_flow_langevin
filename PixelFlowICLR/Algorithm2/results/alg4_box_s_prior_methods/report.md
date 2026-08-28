# Algorithm 4 · prior covariance S 构造方法对比（box inpainting / junco）

> **清理记录（2026-08-25，用户指令）**：对比完成后仅保留 pooled_junco（现行）与
> spectral 两种构造；pooled / centered_trace / channel_diag / two_band_G /
> upper_bound 的代码（utils.py 的 ChannelDiagSOp、TwoBandSOp）、traj 目录与 CSV
> 行已删除。本文档保留全部六臂数字作为决策依据的历史记录。当前
> s_prior_methods.py / 顶层 CSV / 图 只含两个保留臂。

**DEMO_ONLY / LEAKAGE**：所有统计量都测自 demo 图集（6 张非 junco 图；junco 为评测图，已从校准集剔除以避免直接泄漏），不是训练集。任何数值都不得作为论文正式统计。`pooled_junco`（现行配置，junco 单图）与 `pooled_7img` 仅作对照记录，泄漏最大。

固定不变：Eq.(19)、Block 1/2、schedule、S_it=2（均匀）、per-τ γ² 表（gamma2_meas_alg4.json）、seed 42、box mask、sigma_min=1e-8。唯一变量 = S 的构造。所有构造通过 utils.py 的 `SOperator` 接口（`apply_S_inv` / `apply_S_inv_sqrt`）进入同一份 Block-1 代码；标量路径与旧代码逐位一致（回归：2,2,2,2 复现 hole 0.1447/0.1143/0.1291/0.1495）。

## G 的验证（two_band 前置条件）

G = bilinear-down(×2) ∘ nearest-up 数值上是**精确的正交投影**：对称误差 ≤4.9e-9，幂等误差 = 0.0（四个分辨率 32–256），Hutchinson 估计 rank(G)/D ≈ 0.25。two_band_G 合法。

## 每种 S 的公式与实测值（stage 0→3）

| 模式 | 公式 | 实测 |
|---|---|---|
| pooled_junco（现行） | junco 单图 E[x²]−E[x]² | 0.0397 / 0.0472 / 0.0521 / 0.0554 |
| pooled | 6 图全并 E[x²]−E[x]² | 0.2638 / 0.2791 / 0.2898 / 0.2987 |
| centered_trace | Tr[Cov]/D = (1/ND)Σ‖x−μ‖² | 0.2432 / 0.2574 / 0.2673 / 0.2755 |
| channel_diag (R,G,B) | 每通道 centered 方差 | st3: 0.2479 / 0.2597 / 0.3190 |
| two_band_G | s²_low = E‖Gz‖²/rank；s²_high = E‖(I−G)z‖²/(D−rank) | low: 0.891→1.069；high: 0.0273→0.0110（比值 33→97）|
| spectral | S(ω)=E|F z(ω)|²（fft2 ortho，跨图跨通道平均），floor = 1e-8·max(S) | mean(S)=centered_trace（Parseval 校验 ✓）；st3 floored bins=0 |
| upper_bound | s²=1（[-1,1] Popoviciu） | 1.0 |

关键结构事实：**自然图像先验功率的低/高带比是 33–97×**。各向同性 S 把 ker(G)（细节带）先验功率高估 5×（junco 表）到 25×（6 图 pooled）——Block-1 的高频注入被它直接定价。

## Phase 1 · 固定状态 one-step（f32/f35/f37/f38，16 次共享噪声序列的 draw）

x̂₁ 固定（(19) 不含 S），逐臂只换 S 执行一次 Block 1。注入 ‖x₁ᵒᵘᵗ−x̂₁‖²_hole（f32 → f35 → f37 → f38）：

| 臂 | f32 | f35 | f37 | f38 |
|---|---|---|---|---|
| spectral | **0.0177** | **0.0104** | **0.0047** | **0.0017** |
| two_band_G | 0.0510 | 0.0187 | 0.0069 | 0.0021 |
| pooled_junco | 0.0514 | 0.0333 | 0.0100 | 0.0023 |
| centered/channel/pooled | 0.210–0.225 | 0.066–0.067 | 0.0117 | 0.0024 |
| upper_bound | 0.539 | 0.081 | 0.0121 | 0.0024 |

spectral 比现行表再降 2.9–3.2×；Δ_Block1 与 spread 同序。CG 全收敛。完整数据 one_step_metrics.csv。

## Phase 2 · 完整轨迹（S_it=2，seed 42）与 4 种子 spread

| 臂 | hole (seed42) | hole (4 种子均值±std) | obs | spread |
|---|---|---|---|---|
| spectral | **0.1491** | **0.1411 ± 0.0047** | **0.0026** | 0.2468 |
| pooled_junco | 0.1495 | 0.1499 ± 0.0007 | 0.0032 | 0.2646 |
| two_band_G | 0.2905 | 0.2872 ± 0.0064 | 0.0030 | 0.4433 |
| centered_trace | 0.3941 | 0.4102 ± 0.0096 | 0.0032 | 0.5439 |
| pooled | 0.4059 | 0.4199 ± 0.0086 | 0.0032 | 0.5516 |
| upper_bound | 0.4582 | 0.4652 ± 0.0047 | 0.0033 | 0.5822 |

- late-stage 恶化：spectral 与现行表同为温和尾段爬升；四个大各向同性臂在 stage 2 就进入高注入平衡并在 stage 3 爆掉（trajectory_comparison.png）。
- two_band 的 one-step 表现没有传导到轨迹：低带 s²≈1.0 的注入沿 40 帧累积（低带虽受数据约束但 b<1 后同样只出不进）。

## Precision crossover（1/s² vs λ(H_τᵀH_τ/σ_τ²)，ker 方向）

数据精度首次超过先验精度（"噪声被冻结"的起点）：

| 臂 | 首次 crossover |
|---|---|
| upper_bound (1/s²=1) | f26（stage 2, τ=0.666） |
| pooled / centered (1/s²≈3.6) | f28（stage 2, τ=0.888） |
| pooled_junco (1/s²=18) | f36（stage 3, τ=0.666） |
| two_band 高带 (1/s²=91) | f38（stage 3, τ=0.888） |
| spectral（逐频） | 数据占优频率份额：f32 0.6% → f35 6% → f36 13% → f37 31% → f38 79%（近似：全部非 DC 频率按 h_ker 计） |

规律与轨迹一致：**先验越弱（1/s² 越小），crossover 越早，冻进结果的噪声越多**。spectral 的逐频 crossover 从高频向低频渐进推进，没有单一断点——这正是它尾段行为温和的原因。

## 七个问题的回答

1. **pooled vs centered_trace 差多少？** 6 图校准集上差 7–8%（0.264→0.243 等）；两者在采样上几乎不可区分（轨迹差 <3%）。真正的鸿沟在"junco 单图 vs 跨图"：跨图统计因图间均值差异大 5–7 倍。
2. **isotropic S 是否主要在高频过弱？** 恰恰相反——**过强**（先验方差过大=先验约束过弱）。ker(G) 真实功率 0.011，junco 表给 0.055（5×），6 图 pooled 给 0.299（27×）。isotropic 的问题是把低带的大功率摊到了高带。
3. **哪种 S 最有效减少 Block-1 注入？** spectral，全部四个测试帧最低（比现行再降 2.9–3.2×，比 6 图 isotropic 低一个量级）。
4. **MSE 改善是否伴随 spread 明显下降？** 没有。spectral spread 0.247 vs 现行 0.265（−7%），而 4 种子 MSE −6%。大 spread 臂（0.54–0.58）全是 MSE 最差的——那些"多样性"是白噪声不是后验多样性。
5. **改善来自 covariance 还是 collapse？** 来自 covariance。判据：(a) spread 基本不变；(b) 注入下降集中在先验真实功率小 33–97× 的高带，低带注入保留；(c) obs 区改善 19%（collapse 不会改善观测区）；(d) draft §8.2 警告的 collapse 特征（MSE 改善+spread 崩塌）未出现。
6. **调 S 能否解决 late-stage failure？** 不能根治，只能减轻。机制（b∝σ→0 时注入-移除失衡）与 S 无关；spectral 把注入降到更低、把 crossover 推成渐进式，尾段爬升与现行相当（仍存在）。根治仍需 (19) 补 S⁻¹ 或调度层措施。
7. **推荐保留哪种？** **spectral**（每 stage 一条 mean power spectrum + 固定 floor）。理由：无泄漏校准下打平/超过泄漏基线（4 种子均值最优）、注入最低、spread 不塌、obs 最优、实现只是两次 FFT。次选：现行 junco 表仅在"只看 junco"时成立，换图即失效（它的 s² 是 junco 特有的小方差）。

## 文件

- 代码：utils.py（SOperator 家族 + make_M_tau_den 泛化 + Block-1 两行）、s_prior_methods.py（统一入口：verify_g / measure / onestep / trajectory / spread / analyze）
- 统计：s_statistics.json（含 g_check、全部公式与数值）、configs/spectral_power.npz
- 数据：one_step_metrics.csv、trajectory_metrics.csv（合并 6 臂）、spread_metrics.csv、precision_crossover.csv、traj_<arm>/（每臂逐帧 + inner 分解 + 各种子 final npy）
- 图：one_step_comparison.png、trajectory_comparison.png、spread_vs_mse.png
- 配置快照：configs/config_alg4.json、configs/s_prior_config.json

复现：`PYTHONHASHSEED=0 python s_prior_methods.py <verify_g|measure|onestep|trajectory|spread|analyze>`（按此顺序）。
