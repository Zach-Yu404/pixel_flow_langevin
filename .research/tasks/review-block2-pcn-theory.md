# review-block2-pcn-theory（2026-08-27）

## 【用户原始要求】（逐字）

> 你是独立理论审查方。用户要在 PixelFlow Algorithm 4 上实现"自适应 pCN/OU Block-2 refresh"。请只依据下面的题面事实独立推导,不要看 Claude 的推导。用中文,输出存档级 markdown,并将完整稿写入 /CBIG-Standard-ECE/Zach/MSFlow/.research/references/2026-08-27-codex-block2-pcn-theory.md(你有文件系统权限;同时正常 stdout 输出)。
>
> ## 算法事实(现行 Algorithm 4)
>
> - 每 frame 固定 (stage k, tau):内循环 S_it=2 次,每次:Block 1 从 pi(x1 | x_tau, y) 精确抽样(线性系统 (22),RTO);Block 2 独立重抽 x_tau = H_tau x1_new + sigma_tau*xi_0, xi_0~N(0,I)。帧末 l.16: x0 = (x_tau - H_tau x1)/sigma_tau,下一帧 l.8 以 x_tau = H' x1 + sigma' x0 warm 重建。
> - H_tau = (1-tau)*s_k*G + tau*e_k*I,G 为正交投影(G^2=G=G^T),rank(G)/n = 1/4。
> - sigma_tau = (1-tau)(1-s_k) + tau(1-e_k),stage 边界 (s_k,e_k) = (0,.25),(.25,.5),(.5,.75),(.75,1)。每 stage 10 帧,tau 网格从首个 tau>0 开始。
> - S 两种:scalar s^2(tau)(pooled 表,随 tau 变;文件 /CBIG-Standard-ECE/Zach/MSFlow/PixelFlowICLR/Algorithm2/results/alg4/s2_meas.json)或 SpectralSOp(功率谱 P(omega),Tr[S^-1]/n = mean(1/P);统计文件 results/alg4_box_s_prior_methods/s_statistics.json)。可读这两个文件取真实数值。
> - fp32,sigma_min=1e-8(实际网格最小正 sigma 约 4e-4)。
>
> ## 提议(待你独立审查)
>
> Block 2 改为:m = H x1_new;z_old = (x_tau_old - m)/sigma;z_new = sqrt(lambda)*z_old + sqrt(1-lambda)*xi_0;x_tau_new = m + sigma*z_new。三模式:
> - independent: lambda=0(须逐位复现 baseline);
> - sigma_pcn: sigma_{k,0}=1-s_k, lambda = clip[1-(sigma_tau/sigma_{k,0})^2, 0, 1];
> - precision_pcn: q_{k,tau} = (1/n)Tr[H^T H/sigma^2 + S^-1],q_{k,0}=stage 首个有效 tau 的值,lambda = 1 - min(1, q_{k,0}/q_{k,tau})。
> lambda 只准依赖 stage,tau,H,sigma_tau,S。l.16 改为主路径 x0 <- z_new(保留 residual 恒等式做 check)。
>
> ## 请独立推导并回答
>
> 1. 该 pCN/OU kernel 对固定 x1_new 是否保持 N(H x1_new, sigma^2 I) invariant?是否可逆(细致平衡)?作为 Gibbs 子核替换独立条件抽样后,联合链的不变分布是否不变?它与独立 conditional draw 的本质区别?
> 2. 两种 lambda 定义是否 well-defined(值域、端点、clip 是否必要)?sigma_pcn 的 fresh-noise scale sigma*sqrt(1-lambda) 是否恒等于 sigma^2/sigma_{k,0}?
> 3. **precision_pcn 数值审查(重点)**:用上述闭式(Tr[H^T H]/n = a^2+(2ab+b^2)/4, a=tau*e_k, b=(1-tau)*s_k;Tr[S^-1]/n 用真实表)计算 4 个 stage x 10 帧的 q_{k,tau} 与 lambda 表(pooled 与 spectral 各一),判断:q 是否基本单调?有无异常(非单调、跳变、除零、stage 0 tau->0 的退化)?lambda 表是否合理?若不满足预期,按题面要求应停用 precision_pcn 并保留诊断。
> 4. 为什么不能直接把物理 sigma_tau 换成 w*sigma_tau?
> 5. 为什么 lambda 不能依赖当前 state(图像/MSE/resid/v_theta)?依赖 (stage,tau,H,sigma,S) 为什么可以?
> 6. l.16 直接 carry z_new 与原 x0=(x_tau-Hx1)/sigma 是否同一坐标?数值上何时更优?
> 7. 预期效果的理论边界:平稳态下不变核能否改善终态边缘分布?本算法链是否平稳?若有改善应归因于什么?预期 autocorrelation/ESS 代价?
> 8. 实现的验证清单:你认为 baseline 逐位、dense invariance、stationary chain 三类验证各需什么最低配置才有说服力?
>
> 有分歧处请给推导或反例,不要只给结论。

## 任务判定与证据边界

- 类型：独立理论 review；`execution_allowed: false`，不改采样器实现。
- 只使用上述题面事实、题面指定的数值文件，以及为识别指定文件缺项而读取的同目录原始数值资产；不读取 Claude 的推导、计划、共识稿或实现结论。
- 交付：中文存档级 Markdown 写入 `.research/references/2026-08-27-codex-block2-pcn-theory.md`，并将同一完整稿输出到 stdout。

## 当前状态

- 状态：done。
- 完整稿：`references/2026-08-27-codex-block2-pcn-theory.md`。
- 文件 SHA-256：`c32d18a7f1ac6baab280590af8cb268dcf0ed7390f2e2e70eb784003b6098270`。
- 验证：Markdown 0–9 节齐全；pooled/spectral 两张条件表共 80 行，全部由题定闭式重新
  计算核对（最大 lambda/rho 舍入误差 < `5e-10`）。
- 结构化 plan：按协议三次执行
  `.research/bin/research-peer plan --agent codex --task .research/tasks/review-block2-pcn-theory.md`，
  均在建立隔离 worktree 时因 Ceph `Remote I/O error` 失败；另一次临时 clone 因对象复制
  长时间无进展而终止。未读取 Claude plan，亦未伪造 research-peer artifact。
