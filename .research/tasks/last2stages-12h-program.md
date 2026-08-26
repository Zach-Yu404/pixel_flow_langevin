# last2stages-12h-program（2026-08-26，done；Claude×Codex(gpt-5.6-sol/ultra) 三轮）

## 用户原始要求（逐字摘录）

「不改变我的posterior sampler，你和codex进行讨论，codex的模型用5.6 sol,
Reasoning ultra，使最后2个阶段可以继续去噪，图片质量变得更好，但是还得是
exact posterior sampler，你们自行探索尝试，相对于现在的结果没有提升或者更差
的都不保留，但是也要思考为什么没提升，或者是否有可取之处，给你喝codex，
12个小时讨论探索，代码和结果要精炼，同时给我写一个报告记录所有关键信息，
无关或中间结果，分析完就可以删掉，不可以占用太多空间」

## 结果

**保留项唯一：`num_langevin=[2,2,1,1]`（config 已生效）**。
spectral 0.1411±0.0047→**0.1317±0.0038**（−6.7%），pooled 0.1499±0.0007→
**0.1382±0.0054**（−7.8%）；NFE 80→60；spread/obs 基本不变；纯调度、每步仍
是原 conditional 精确抽样。剂量-响应单调（2111≫2222>2221>2211），边界在
stage-1/2 交界。终版双回归通过（标量逐位 + 赢家复现）。

否定并回撤（全 4 种子、代码零残留；机理见 report）：Adler/pCN 核族全谱
（ρ<0 反射=网络外推灾难；ρ>0 持续化=实现彩票，seed-42 −16% 假象、跨种子
std ×5；Codex 零旋钮规则 ρ=2^{-1/(2q)} 与实测重合但同败）、深 S_it、
shift τ 集中（obs 反噬）、逐帧 b 阈值（负于 2211）、有色 Γ(ω)（三臂全负，
高误差频段反而关移除通道）。τ 细化=未运行（γ² 查表 fallback 触发
IndexError——08-25 审计隐患首次实际咬人，改网格前必须修）。

**定理级产出（论文素材）**：① 每 sweep 风险增量 R₂−R₁=a²[q−(1−a²)E₀²]
（尾段第二 sweep=纯注入）；② 兼容性缺口：exact-Gibbs 相容要求 C⁻¹J_f=h/σ²，
唯一相容斜率 K*=hs²/(h²s²+σ²) 满足涨落-耗散 K*²σ²+C=(1−a*²)s²，失衡量
𝓑=(h²s²+σ²)(K²−K*²)+b²γ²——(19)/(12) 不自洽的定量化，也解释 antithetic
（K→K*）为何曾到 0.0875；③ 措辞修正：本采样器=**exact updates for frozen
surrogate conditionals**，非全局 exact posterior sampler；④ 实用地板代理
det-0.0797+spread²≈基线均值；合法类无普适下界（0.1317 即证）。

## 产物（精炼后）

results/alg4_last2stages_program/{report.md, program_results.csv,
winner_montage_spectral_sit2211.jpg, winner_final_samples.png}；
references/2026-08-26-codex-last2stages-rounds.md（三轮最终答复存档）。
中间产物（探针/批脚本/日志/失败配置/Γ 谱表）已删。
