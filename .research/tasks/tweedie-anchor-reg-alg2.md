# tweedie-anchor-reg-alg2

state: done · owner: claude · type: implementation+experiment · 开始：2026-08-20

## 【用户原始要求】

> 请在当前 `utils.py` 里，**直接替代现有 `run_posterior_reg_sampling_alg2` 的实现**，改成一个基于 **Tweedie pseudo-observation / nullspace regularization** 的 Algorithm 2 版本。目标不是改原 `run_posterior_sampling_alg2`，只修改 `run_posterior_reg_sampling_alg2`。
>
> 核心思想：x1_model = x1 + eps_a, eps_a ~ N(0, lambda^-1 I)，把 network 的 Tweedie clean estimate 当作一次 Gaussian pseudo-observation。M_tau = M_tau^(0) + lambda P；b_tau = b_tau^(0) + lambda P x1_model。P 优先使用 measurement nullspace / hole projection；inpainting 用 P = I - A^T A，在当前 mask convention 下直接用现有 hole mask/operator 实现，不显式构造矩阵。这样 observed region 不加 anchor、hole region 才加入 model prior，避免原来 lambda I 把观测区域也拉向 model prediction。如果 operator 不是简单 inpainting、无法可靠构造 P_ker(A)，允许 fallback 到 P=I，但必须在 report 中明确记录。
>
> 1. x1_model 使用当前已有 velocity，不要增加额外 NFE，默认用 direct_estimate_x1（不要修改它）。
> 2. Gaussian sampling noise 必须同步修改：加入 sqrt(lambda) P xi_a。tau=0 的 ridge epsilon I 与 sqrt(epsilon) xi_eps 逻辑保持原样。
> 3. 保留一个明确参数 anchor_lambda=0.0；anchor_lambda=0 时尽可能严格退化回原 Algorithm 2 路径。不要增加其它 regularization 超参数。
> 4. Lambda sweep：至少 0/5/10/25/50/100，保持 image/seed/mask/measurement/config/h0/CG/transition/line 15/stage schedule 完全相同。不要做大规模 grid search。
> 5. RNG 必须配对：即使 anchor_lambda==0 也生成 xi_anchor，只是乘以 0，使所有 lambda 是 paired-noise comparison。
> 6. 测试代码和结果都放在 test/（test_reg_alg2.py + results_reg_alg2/），保留不删除。
> 7. 每个 lambda 至少记录 final hole/visible/full MSE、每 stage-end hole MSE、x1_model hole MSE、Block-1 posterior mean hole MSE、Block-1 sampled x1 hole MSE、hole sample variance / RMS noise、measurement residual。重点判断 lambda 是否主要通过压低 hole posterior variance 起作用。
> 8. 理论 sanity check：Var = 1/(m+lambda)；特别检查 lambda=25 => std≈0.2 是否与实际一致。scalar/projection-level 即可。
> 9. 只选 lambda=25 额外比较 isotropic (P=I) 与 nullspace，验证 nullspace 是否保留填洞能力同时减少 observed-region bias。
> 10. 不允许同时改其它东西（run_posterior_sampling_alg2 / run_posterior_wv_sampling_alg2 / line 15 / Block 2 / h0 / transition / ridge schedule / CG / network / direct_estimate_x1）。
> 11. 最终 report 写在 test/results_reg_alg2/report.md，只回答 Implementation / Lambda sweep / Stage-end results / Isotropic vs nullspace / Analysis / Decision。

## 【用户后续要求（逐字）】

> 你之前是怎么加的lambda，不过当时我让你放弃这种方式了，现在重新给我写到这个reg方法，进行测试，现在写的版本不对

原版定位为 commit `74a0c39`（`full_ip_compare.py: sample_alg2(anchor=lam)`），
在 `8ff8091` 被删除：`diag = ridge + anchor; M += diag*I`，
`b += anchor*x1_model + sqrt(anchor)*randn_like_cpu(x1)`，
`x1_model = direct_estimate_x1(x_tau - tau*v, x_tau + (1-tau)*v, sk, ek)`。
当时记录：junco 洞区 0.969 → 0.086（λ=25，11.3×），7 图 pooled 1.097 → 0.178。

已把它恢复为 `run_posterior_reg_sampling_alg2` 的**默认路径**（`anchor_P="identity"`），
nullspace 版保留为可选。两处有意保留的后续修复：τ=0 的 `sqrt(epsilon) xi_eps`；
`xi_a` 在每个 λ（含 0）都抽以修掉 RNG 错位。

## 结论

**KEEP。** 各向同性（原形式）anchor 把洞区从 λ=0 的 **1.0062** 降到 λ=100 的 **0.1023**（9.8×），
**且 visible（0.0036286 → 0.0028924，−20%）与 measurement residual（0.0021282 → 0.0020962）
也随 λ 单调改善**。拐点 λ≈25–50，λ=100 三项皆最优。

- **注**：`run_posterior_reg_sampling_alg2` 此前不存在于 `utils.py`，是新建而非替换。
- **P 的真实性质**：精确正交投影（P²−P、自伴均为 0.00e+00），但因 `A_k = A∘U` 且 U 是
  插值上采样，"属于 ker(A_k)"**只在 stage 3 精确**；stage 0–2 泄漏 16%–42%。
- **因果方向是"先压方差、再救均值"**：λ 0→5 均值改善 30×（2.377→0.079）、方差只改善 2.4×；
  λ 25→100 均值几乎不动（−7%）、方差减半（−48%）。
  末端（gstep 38）方差在所有 λ 下都只有 ~0.046——**末端方差从来不是瓶颈**，
  瓶颈是早中期方差造成的状态退化级联（λ=0 时网络自身的 x1_model 退化到 0.99983）。
- **Var = 1/(m+λ) 成立**：λ=25 预测 std 0.19，实测 0.1836 / 0.1710。
- **nullspace vs isotropic：原来的各向同性形式在三个指标上都不差**
  （λ=25：hole 0.115043 vs 0.116042；visible 0.0034351 vs 0.0036451；
  meas-resid 0.0021161 vs 0.0021319）。
  原因是 inpainting 的 y 无加性噪声，观测区误差来自采样器自身噪声而非测量。
  **不可外推到 blur/SR**（那里 y 有噪声，且无 hole mask 会 fallback 到 P=I）。

代码：`utils.py` **纯新增 502 行、0 行删除**；`test/test_reg_alg2.py` 保留。
`--mode verify` ALL CHECKS PASSED。
