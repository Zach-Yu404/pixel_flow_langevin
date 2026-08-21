# alg3-parameter-sweep

state: done · owner: claude · type: experiment · 开始：2026-08-20

## 【用户原始要求】

> trace_arms.py另外删掉

> 我对main2.py也进行了修改，现在对其各个参数进行调整，并测试，和之前的方式进行诊断和分析结果

## 被测改动（用户在 main2.py 中所做）

l.14 由 (51) 的 `utils.apply_H_tau_inv` 换成 `utils.direct_estimate_x1`；
l.15 的 `rebuild_state` 整段注释掉；l.17 的 rhs 换成 `utils.data_rhs_matchx1`。

## 结论

**改动后的 main2.py 已不再是 Algorithm 3。** 注释掉 l.15 的同时把整个 Block 1
（l.12–13 的 x0 Langevin）变成了死代码——l.14/l.17 都不读 x0，l.19 又把它整个覆盖。

两条决定性证据：
1. **h0 是 no-op**：h0 = 0.05 / 0.1 / **5.0** 洞区 MSE 逐位相同（0.8605503439903259），
   h0=0.3 只差第 7 位（浮点噪声量级）。对照：关掉 `recompute_x0` 后 h0 立刻生效
   （0.1→2.909，0.3→3.456）。
2. **x_τ 在 step 内被 l.19 精确冻住**：`‖x_τ^{s+1}−x_τ^{s}‖/‖x_τ^{s}‖ = 5.4e-09…7.9e-08`
   （Algorithm 2 同口径是 0.091…0.451），而 l.13 的 Langevin 确实把 x0 动了 28–33%
   —— 动完就被丢掉。S 次 inner iteration 退化成"用同一个 b 重复抽 S 次"。

扫描结果（box_inpainting/junco/seed 42，对照 alg2 = **0.098428**）：
ref/h0_0.05/h0_5.0 = 0.86055，h0_0.3 = 0.86055，S5 = 0.90027，S20 = 0.86809，
skip_tau0=False = 1.35206，recompute_x0=False = 2.90867，同上×h0=0.3 = 3.45614。
**Alg3 最好一组比 Alg2 差 8.7 倍，无任何参数取值接近。**

## 我做的两处最小改动（正式代码）

1. `check_52` 默认 True → **False**（+ docstring 说明）。**这是硬阻塞**：
   改动后 main2.py 直接 `AssertionError: (52) violated at stage 0 tau=0.111:
   relative residual 7.456e-02`。(52) 只有在 x̂₁ 被定义为 `H⁻¹(x_τ−σx̂₀)` 时才是恒等式。
2. 把硬编码的 τ=0 跳过提成 kwarg `skip_tau0=True`（默认行为不变），
   因为 `apply_H_tau_inv` 已不再被调用、跳过不再有数学必要，需要可测。

产物：`PixelFlowICLR/Algorithm2/results/alg3_sweep/`
（report.md、comparison.csv、sweep.png、各组 traj CSV 与 final PNG）。
临时脚本已删；`main.py`/`utils.py` 未改动；`--mode verify` ALL CHECKS PASSED。
