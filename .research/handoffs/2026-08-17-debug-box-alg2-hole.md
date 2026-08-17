# Handoff: debug-box-alg2-hole（2026-08-17）

**一句话**：box 洞区不填充 = 论文 Algorithm 2 的结构性缺陷（Block 1 条件分布在 ker(A)
无图像先验，洞区链无收缩）；Tweedie 锚定变体（λ=25）7/7 图修复（洞区 MSE 6.2× 池化改善，
obs 无回退），已实装为 run_posterior_sampling_alg2(anchor=...) 可选参数，默认 0=论文原算法不变。

**证据链**：① h₀∈{0.1,0.5,1.0} 扫描证伪（job 18634904）；② 标量 oracle 仿真：完美 score
下原算法 std 4.8 vs 目标 0.2，锚定 λ=25 精确恢复目标后验；③ Prop.4 核对排除实现偏差；
④ 真模型 anchor 扫描（job 18635053）剂量-响应单调；⑤ 7 图复核（job 18635142）7/7 全胜。

**改动**（全部在 PixelFlowICLR/Algorithm2/）：full_ip_compare.run_posterior_sampling_alg2 增 h0/anchor 参数
（默认行为逐位不变）；debug_box_h0.py（h₀/anchor 双模式 sweep + obs/hole 分离 MSE）；
run_debug_box_h0.sbatch / run_debug_box_confirm.sbatch；results/debug_box_{h0,anchor,confirm}/。

**下一步（待用户）**：是否采纳锚定进论文 Alg2；λ 调度/取值细扫；其余任务（random_inpainting
也有 30% 零空间）是否同样受益。
