# prior-injection-b-tau

state: done · owner: claude · type: experiment · 开始：2026-08-20

## 【用户原始要求】

> 我已经修改了 Algorithm 2(utils.py)：现在用 network-derived clean estimate (\hat x_{1,\theta}) 直接向 (b_\tau) 注入 prior information，而不是把当前 (x_\tau) 做 self-consistent decomposition 后再塞回去。
>
> 请做一次最小但有区分度的验证。不要继续改算法，先判断这个修改是否真的解决 hole 区没有 semantic content 的问题。
>
> 1. 测试设置
> 用当前正式 inverse-problem config：固定 1 张 representative image；固定 seed；与修改前完全相同的 measurement / mask / stage / time grid；同时跑修改前 baseline 和修改后的 Algorithm 2；不做大规模 sweep。
>
> 2. 最重要：验证新的 (b_\tau) 是否真的带入了新的 prior information
> 对几个代表性的 ((k,\tau))，尤其是 hole 开始恶化的位置，保存 \hat x_{1,\theta} 以及 Block-1 conditional mean。比较 Baseline mu_old = M^-1[A^T y/eta^2 + H^T x_tau/sigma^2] 与 Modified mu_new = M^-1[A^T y/eta^2 + H^T H \hat x_1/sigma^2]。重点看 hole 区：x1_hat_model / mu_old / mu_new / GT，并记录 hole MSE。如果修改正确，期望看到 mu_new,hole 明显比 mu_old,hole 更接近 \hat x_1 / GT，而不是仍然只是 noise。
>
> 3. 完整 end-to-end trajectory
> 然后各跑一次完整 baseline / modified sampler。沿 global time 只记录：total MSE / visible MSE / hole MSE / \hat x_1 hole MSE / Block-1 mean hole MSE / sampled x_1 hole MSE。保存少量关键时间点图片：GT | x1_hat_model | Block1 mean | sampled x1 | error map。重点回答：新 prior 是否真的进入 hole？Block-1 mean 是否已经有 semantic structure？如果 mean 好但 sampled x_1 又变成 noise，是否是 Gaussian sampling covariance 把 prior 淹没？后续 Block 2 是否保留、改善还是破坏这部分 semantic information？修改后的 final hole MSE 相比 baseline 改善多少？visible-region data consistency 是否基本不变？
>
> 4. 一个关键分解：mean vs sampling noise
> 同一个 ((k,\tau)) 比较 mu = M^-1 b 和实际 sample x1 = M^-1(b+zeta)，分别计算 hole MSE。如果 mean 很好 / sample 很差，说明新的 prior injection 已经成功，下一瓶颈是 Block-1 covariance / sampling noise。如果 mean 本身仍然差，说明 \hat x_1 或新的 b_tau 注入方式仍没有提供足够有效的 clean prior。
>
> 5. 最后不要继续自动调参
> 根据结果只做判断：SUCCESS / PARTIAL / FAIL。不要自动加入 ridge、lambda、clipping 或其它 heuristic。
>
> 6. 实验结束后清理
> 所有 instrumentation 都保持 minimal。测试完成后删除临时 debug/test 代码，只保留 results/.../baseline/ results/.../modified/ comparison.csv report.md。正式 main.py/main2.py/utils.py 不要因为本次测试变复杂。
>
> 最后只告诉我：Tested / Results / Where / Analysis / Conclusion。

## 被测改动（用户在 utils.py 中所做，未提交）

- 新增 `data_rhs_matchx1`：`b = A^T y/eta^2 + H_tau^T H_tau x1_hat / sigma_tau^2`
- 新增 `direct_estimate_x1(x_s, x_e, s_k, e_k) = ((1-s)x_e - (1-e)x_s)/(e-s)`
- 采样器 l.11 之后由 v 反解端点得到 `x1_hat`，l.13 的 rhs 改用 `data_rhs_matchx1`
- 不消耗随机数 → 与 baseline 的 xi 流逐位对齐

## 结论

**SUCCESS**。box_inpainting/junco/seed 42：洞区 MSE 1.0469 → 0.0984（10.6×），
可见区 0.003622 → 0.001947（1.86×），全图 0.26443 → 0.026067。
同状态 rhs 对照（同 x_tau、同 M_tau，只换右端项）下 Block-1 mean 的洞区误差降低 19–93 倍。
mean/noise 分解精确闭合，噪声项与 baseline 逐点相同（协方差与 b 无关）——
每一步的抽样几乎是纯噪声，链条靠 mean 逐步收敛。

**下一个问题**：语义在 stage 2 末已建立（连贯背景，x1_hat 洞区 0.0654），
过 stage 2→3 后退化成周期性点阵（最终图洞区是灰底+点阵，不是可用重建）。

**已确认的次要偏差**：`direct_estimate_x1` 在 stage 1/2 上把 ker(G) 分量放大 1.2000 倍
（= e(1-s)/(e-s)），stage 0 和 stage 3 精确。

**性质改变（事实记录）**：`b_new` 丢掉 `sigma_tau H_tau^T x0`，Block-1 不再是论文模型
给定 x_tau 的精确 Gibbs 条件分布；改动后的采样器是 plug-and-play/proximal 变体。

产物：`PixelFlowICLR/Algorithm2/results/prior_injection/`
（`report.md`、`comparison.csv`、`trajectories.png`、`baseline/`、`modified/` 各 11 张关键时间点图 + final.png）。
仪表全在临时脚本内完成，已删除；`utils.py`/`main.py`/`main2.py` 未因本次测试改动。
