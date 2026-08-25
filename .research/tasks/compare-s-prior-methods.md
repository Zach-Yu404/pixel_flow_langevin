# compare-s-prior-methods（2026-08-25，done）

## 用户原始要求（逐字摘录关键约束；全文规格见对话记录）

「还是全部都是2的，在此基础之上，在当前 Algorithm 4 上比较不同的 prior covariance S
构造方法。先只使用 box inpainting，不修改 Eq. (19)、Block 1、Block 2、schedule、S_it
或其他 sampler 结构；唯一变量是 S 的计算方式。注意：代码接口应统一使用 s2，即方差 s²，
不是标准差 s。」需实现与比较：pooled（当前实现，baseline）/ centered_trace（主候选）/
channel_diag / two_band_G（先数值验证 G 为正交投影，否则说明原因跳过）/ spectral
（FFT 实现 apply_S_inv 与 apply_S_inv_sqrt，只允许防除零 floor，明确记录，不搜索）/
upper_bound（s²=1）。数据口径：正式应用训练/校准集；只有 demo 图时全部标注
DEMO_ONLY/LEAKAGE。第一阶段固定状态 one-step 对照（固定 x_tau/v/x1_hat/xi/CG 初值容差
种子，≥16 共享随机序列的 draw）；第二阶段选 pooled、centered_trace、one-step 最好的
structured、upper_bound 跑完整 trajectory；分析 1/s² 与 λ(H_τᵀH_τ/σ_τ²) 的 crossover；
回答七个问题；输出到 results/alg4_box_s_prior_methods/（s_statistics.json、
one_step_metrics.csv、trajectory_metrics.csv、spread_metrics.csv、
precision_crossover.csv、三张 png、report.md、configs/）；统一 SOperator 接口
（apply_S_inv/apply_S_inv_sqrt/metadata），共用 Block 1 主代码不复制 sampler，不修改
alg2/wv/reg，不引入 g_bypass_stage3/h0/ridge/新搜索参数，清理临时脚本，保留清楚的
运行入口和完整 config。

后续裁决（逐字）：「目前只保留pooled_junco和spectral的，其他的相关结果和代码全部清理干净」
「先确定画的图是没问题的，保存的变量都是对的，描述分别保存的是什么」

## 结论（全文见 results/alg4_box_s_prior_methods/report.md，含六臂历史数字）

- G 数值验证：**精确正交投影**（对称 ≤5e-9、幂等误差 0、rank/D=0.25）→ two_band 合法。
- 结构事实：先验功率低/高带比 33–97×；isotropic S 把 ker(G) 先验功率高估 5×（junco 表）
  至 27×（6 图 pooled）。
- one-step（f32/35/37/38，16 共享 draw）：注入排序 spectral < two_band < pooled_junco
  ≪ centered≈channel≈pooled ≪ upper_bound；spectral 比现行再降 2.9–3.2×。
- 完整轨迹（S_it=2）：spectral **0.1491**（4 种子 0.1411±0.005）≈/超 现行 pooled_junco
  0.1495（0.1499±0.001），obs 0.0026 vs 0.0032；two_band 0.29（低带 s²≈1 沿轨迹累积）；
  isotropic 大方差三臂 0.39–0.46 溃败。
- spread：spectral 0.247 vs 现行 0.265 —— **非 collapse**（改善来自砍掉先验中不存在的
  高带功率；低带多样性保留；obs 同时改善）。
- crossover（1/s² vs h_ker²/σ²）：upper f26 / pooled f28 / junco 表 f36 / two_band 高带
  f38 / spectral 逐频渐进（f32 0.6% → f38 79%）——先验越弱冻结越早，与轨迹完全同序。
- 七问回答与推荐（spectral；正式采用前须换训练集重测谱）见 report.md。
- 用户裁决后清理：只留 pooled_junco + spectral；ChannelDiagSOp/TwoBandSOp 及四臂
  结果已删；顶层 CSV/图已过滤重出；report.md 保留六臂数字作历史依据。
- 产物核验（用户指令）：8 项数值/图目检全过（npy 端到端复算=CSV、Parseval=1.000000、
  delta 恒等式 128/128、谱对称 ≤5e-8、crossover 抽查、图逐张目检）；修复 2 处
  （s_statistics.json 旧键名重生成、spread 图 x 轴刻度）。

## 代码

- utils.py：SOperator / ScalarSOp / SpectralSOp；make_M_tau_den 接受标量或 SOperator
  （返回签名不变，第 5 元仍浮点）；Block-1 仅两行改动；标量路径逐位回归通过。
- s_prior_methods.py：统一入口 measure/onestep/trajectory/spread/analyze（双臂版）。

## 口径警告

DEMO_ONLY/LEAKAGE：spectral 谱来自 6 张非 junco demo 图（无 junco 直接泄漏，但仍是
eval 家族）；pooled_junco 为 junco 单图（最大泄漏）。不得作为论文统计。
