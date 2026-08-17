# 2026-08-16-alg2-onestep-mse-vs-t

任务：alg2-onestep-mse-vs-t（tasks/ 同名文件）
目的：在实验 1 的 WLS/Model 之外，测 Algorithm 2 one-step 估计器（两次 CG：line 11 score
solve + line 14 clean solve）的 x̂₁ᵏ MSE vs t；顺带测 score 误差与 γ²_meas（note 式 56）。

## 复现信息
| 项 | 值 |
|---|---|
| 代码 | `PixelFlowICLR/Algorithm2/algorithm2.py`（import 复用实验 1 全部 helper；y/operator = demo_runner.build_setup_and_measurement，playground 同代码路径） |
| command | `sbatch PixelFlowICLR/Algorithm2/run_alg2.sbatch`（内含 `PYTHONHASHSEED=0`——demo_runner 掩码种子仍用 hash()，registry #6 残留） |
| config | 各任务 LPIPS_king*.json：operator 块 + sigma_n=0.05（=η）+ kw（与实验 1 相同的 9 个消费参数） |
| job | 18591733（A100 udc-an37-1，5m28s，2026-08-16） |
| 产物 | `PixelFlowICLR/Algorithm2/results/`：alg2_mse.csv（1400 行，35 个 σ<0.01 跳过点 Alg2=NaN）、alg2_<task>.png ×5（全局 t 拼接、log-y、粗均值+细逐图；inpainting 加 obs/miss 面板）、score_gamma2.png、gamma2_meas.json、sanity.json、**alg2_predictions_<task>.mp4 ×5**（实验 2b 最终版，job 18594863，用户指定行序 [GT\|WLS\|Model\|Alg2\|x_t] × 7 图，40 帧/任务，逐图 MSE 标注，σ<0.01 面板标灰，5.4-6.9MB each，gitignored。前两版——9 行合并版 job 18592798、8 行无 x_t 版 job 18594600 中途取消——均已废弃删除） |

## Sanity（全过，先于 GPU 跑）
- 伴随测试（blur/motion 换 autograd 精确伴随后）：worst 4.8e-9（flip(K) 解析伴随在 32² 差 1.8e-3——reflection padding 的伴随不是 reflection padding，已弃用）
- S1 恒等式（解析证明 σ_τB+(e−s)H_τ≡N ⇒ v:=d_exact 时 x̂₀=x₀）：worst rel err 1.1e-7
- S2 判决（stage 3 直测，无噪 y）：random obs RMSE 0.0110 ≪ η=0.05 ✓（stage3 直测通过）；
  resize stage 的 obs 误差偏高（~0.09-0.11，obs≈miss）——偏高原因未定：sanity.json 只有
  RMSE，未做能区分 wiring 与 conditioning 的对照

## 结果（pooled mean MSE，A=Alg2 / W=WLS / M=Model；W、M 与实验 1 逐位同源）

| task | s0 (32px) | s1 | s2 | s3 (256px) |
|---|---|---|---|---|
| box_inpaint | A 7.29 / W 7.74 / M 0.112 | A 3.12 / M 0.041 | A 1.45 / M 0.014 | A 0.035 / M 0.005 |
| random_inpaint | **A 0.011** / M 0.112 | **A 0.019** / M 0.041 | A 0.37 / M 0.014 | A 0.097 / M 0.005 |
| gaussian | **A 0.012** / M 0.112 | A 0.26 / M 0.041 | A 2.82 / M 0.014 | A 0.13 / M 0.005 |
| motion | **A 0.049** / M 0.112 | A 0.17 / M 0.041 | A 0.95 / M 0.014 | A 0.12 / M 0.005 |
| SR ×4 | **A 0.011** / M 0.112 | **A 0.025** / M 0.041 | A 6.17 / M 0.014 | A 0.13 / M 0.005 |

要点（观察，见图；结论表述遵守实验 1 review 的口径纪律）：
1. **Alg2 曲线终于任务分化**（line 14 含 A/y），W/M 仍任务无关（同实验 1）。
2. **早 stage、测量信息强的任务上 Alg2 大幅优于 Model**（random/gaussian/SR/motion 的 s0：
   0.011-0.049 vs 0.112，最高 ~10×）；box 的 s0-s2 差（洞区在 s₀=0 时 H=τe·I 弱先验 +
   测量零约束）。
3. **晚 stage Model 全面更好**（s3 全任务 M≈0.005 < A 0.035-0.13）：one-step line 14 的 b
   只用 (y, x_t)，不含网络 v（按 spec；v 只进 warm start 与 line 11），σ→小时先验项把
   x̂₁ 拉向 x_t 本身。
4. **stage 2 出现 Alg2 的 MSE 反弹**（SR 6.17、gaussian 2.82 反超 s1/s3）。本次未记录
   任何特征值/条件数数据，反弹原因未定（条件数归因需补矩阵无关的谱估计后才可下）；
   胜负计数：SR 全程 alg2<model 118/273，box 11/273。
5. score solve：err 随 stage 升（s0 2.7e-3 → s3 8.7e-2）；γ²_meas ~0.009-0.020;
   γ²=γ²_meas vs γ²=0 的差异（口径：每个 stage/t 网格点对 7 张图求均值，复算脚本
   `Algorithm2/verify_gamma2.py`）：均值曲线最大相对差 10.94%（stage3/τ=0.888：
   0.25466 vs 0.28251），单图最大 19.60%。两组曲线形态与量级一致，但差异达
   ~11%（均值）/~20%（单图），不能称"几乎重合"。

## 备注
- pipeline._daps_motion_kernel 硬编码旧机器 DAPS 路径（registry #16）——实验侧预插当前
  路径解决，原代码未动；playground_runs 未写入。
- v/score/γ² 与任务无关 → 跨任务缓存，网络调用与实验 1 相同（280 次 forward）。


## 追加(2026-08-16 晚):统一版对比(用户改规格,覆盖 results/)

用户裁定:三组唯一变量 = 算法,其余全同;Alg2 的测量**暂用 y=GT(A=I, stage 分辨率),η=0.05**
("y 现在暂时用 gt",无任务退化,任务维度取消)。job 18597943(A100,sacct 验证)。
- 产物(results/ 已按用户要求覆盖,旧分任务 png/mp4/csv 删除,git 历史保留;PDF 保留):
  `unified_predictions.mp4`(40 帧,行序 GT|WLS|Model|Alg2|x_t,4.8MB)、
  `mse_vs_t_unified.png`、`unified_mse.csv`(280 行)
- 数字(pooled mean,已从 unified_mse.csv 复算验证):Alg2 stage0-3 = 2.17e-7 / 1.82e-6 /
  1.32e-5 / 1.72e-4;WLS = 7.74 / 1.08 / 0.13 / 5.3e-3;Model = 0.112 / 0.041 / 0.014 / 5.1e-3;
  y=GT 时测量项主导,Alg2 接近精确复原(stage0/t0 单图 3.6e-13)——这是设计上限探针,
  非真实测量下的算法比较(后者见本文件上方 5 任务结果)。

## 后记（2026-08-17）
- 收尾提交工作区遗留的 `Algorithm2/algorithm2.py` 修复：`make_exact_AT` 的 autograd 伴随在
  strict deterministic 模式下会因 `reflection_pad2d_backward` 无确定性 CUDA 实现而抛错，
  现按仓库 blur/SR 伴随的 warn_only 惯例加守卫并在调用后恢复原模式。
  验证：py_compile ✓；守卫恢复 strict 模式 ✓；伴随恒等式 <Ax,r>=<x,ATr>（atol 1e-5）✓。
  不影响已发布的实验数字（该路径仅在 strict-det 环境下才触发差异）。
