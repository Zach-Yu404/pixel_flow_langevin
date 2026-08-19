# score-solve-x1hat-72

state: done · owner: claude · type: experiment · 2026-08-19

## 【用户原始要求】（逐字）

> 为了加入prior的信息，我加了这两行，你考虑一下合理性，并测试看hole area是否差生reasonable的结果

（对应其在 main.py 的临时改动 `x_tau = (tau*x1_model + x0_hat * (1-tau)).detach()`，
随后被用户自己替换为 `##Predict x_1_hat^k:` 占位。）

> pdf第20页有个\hat x_1^k的预测，给我补充在/CBIG-Standard-ECE/Zach/MSFlow/PixelFlowICLR/Algorithm2/main.py line 305，不跑后边的内容，我只想看从gt得到的每一步的x_\tau然后预测的\hat x_0^k得到的\hat x_1^k怎么样，都只做一步的预测（这一步的目的是测试section 7.2,测试Score)，在/CBIG-Standard-ECE/Zach/MSFlow/PixelFlowICLR/Algorithm2/test完成实验

## 论文依据（p.20 §7.2）

x̂₁^k = (H_τ^k)^{−1}(x_τ^k − σ_τ^k x̂₀^k)；exact d_τ 下必须精确等于 GT，
v_θ 下应是 x_τ 的合理去噪、小 τ 糊、随 τ→1 锐化、且不比 x_τ 更噪。

## 实现

- `utils.apply_H_tau_inv`：闭式逆（G 为正交投影 ⇒ H_τ 只有两个特征值）；
  恒等式实测 ~1e-15；τ=0 且 ker(G) 非平凡时**拒绝求值**（H_0 = s_k G 奇异）。
- `main.py:305` 按用户指定位置补入 x̂₁/x̂₁_g2，并接进 onestep 的落盘行（新增
  `x1hat_err` / `x1hat_err_g2` 两列），避免悬空变量。
- `test/score_x1hat.py`：§7.2 实验（一步，不跑采样器）。

## 结论

1. **恒等式无一例外成立**：37 个格点 exact 列 x̂₁ vs GT 为 1e-12~1e-16 ⇒
   Prop. 7 代数 + 算子实现 + CG 求解器同时通过。
2. **stage 3 网络列健康**：洞区 0.0086(τ=0) → 0.0006(τ=0.888)，
   对照采样器洞区 ~0.96，好两三个数量级；"never noisier than x_τ" 每个 τ 均满足。
3. **x̂₀ 误差与 x̂₁ 误差反向**：前者随 τ 暴涨(0.012→0.776)，后者单调下降——
   σ_τ→0 抵消。故 γ² 在 stage 3 末期最大之处，恰是它最不影响 x̂₁ 之处。
4. **粗尺度 x̂₁ 是低频色块属正确行为**：x_τ 在 ker(G) 的 SNR = τe_k/σ_τ，
   stage 0 在 τ=0.111 处仅 **2.9%**。无法从 3% 信号恢复内容。
5. **否定式定位**：§7.2 中 x_τ 由完整 GT 构造、无洞可填，mask 仅用于分区；
   结论是 score solve 与 x̂₁ 恢复在洞区位置**健康**，采样器的洞区失败不在这一环。

## 对用户原改动的判断

`x_tau = tau*x1_model + (1-tau)*x0_hat` 与论文的 x̂₁ 公式**不是一回事**：
论文是从 x̂₀ 反解 x₁（用 H_τ 的逆），用户那行是把 x1_model 与 x0_hat 按 (τ, 1−τ) 线性混合
去覆写 x_τ。后者既不等价于 (6) 的插值（那里是 x̂_start/x̂_end 的凸组合，不是 x1 与 x0），
也会把先验直接写进状态而破坏 Block 1 的条件分布语义。已按论文公式实现，未采用该写法。

## 产物

`test/results/score_x1hat/`：`score_x1hat_junco.csv`(37 行) + 4 张 stage 面板图；
结论写入 `test/results/SUMMARY.md` §5。
