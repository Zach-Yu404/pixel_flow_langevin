# algorithm2-line15-diagnosis

state: done · owner: claude · type: experiment · 开始：2026-08-20

## 【用户原始要求】

> 请对我当前的 **Algorithm 2** 做一次最小但有效的诊断。重点只查真正可能导致 hole 区最终退化的问题，不要扩展成复杂 benchmark。
>
> ## 1. Transition 只做代码对比，不单独设计实验
> 当前 stage transition 直接和 `/CBIG-Standard-ECE/Zach/MSFlow/PixelFlow/ms_posterior_sampling_article_version_final.py` 中的原始实现逐行对比。重点确认：`x1` 的 upsample 是否一致；upsample mode 是否一致；新 stage 的 `x0` 是否都是 fresh noise；noise distribution / correlation 是否一致；stage index / `g_bypass` 是否一致；stage 起点 `x_tau` 的构造是否一致。如果完全一致，就不要继续怀疑 transition，也不要为它写额外实验。
>
> ## 2. 重点检查 Algorithm 2 的 line 15
> 当前流程 x_tau^old = H_tau x1^old + sigma_tau x0^old，Block 1 得到新的 x1^new，然后执行 x0 <- (x_tau^old - H_tau x1^new)/sigma_tau。这会严格保证 H_tau x1^new + sigma_tau x0 = x_tau^old。请验证：现在我已经修改了 b_tau，让 network-derived \hat x_1 直接作为 prior information 进入 Block 1 后，继续执行 line 15 是否会把 Block-1 新得到的 semantic information 从实际 x_tau state 中抵消掉。只做一个最小实验。在相同 image / seed / config 下比较：Baseline 保持当前 line 15；Test 临时不使用 line 15 去强制恢复旧 x_tau，让新的 x1 真正进入下一 state，使用当前已有 x0 构造 x_tau^new = H_tau x1^new + sigma_tau x0。不要同时改 Block 2、h_0、ridge、CG 或其它任何东西。比较：hole MSE trajectory；final hole MSE；visible MSE；Block-1 mean；实际 x_1；新 x_1 是否真正改变了下一次 network 看到的 x_tau。尤其直接记录 |x_tau^{after Block1} - x_tau^{before Block1}|。如果 baseline 几乎为 0，而去掉 restore 后 hole 明显改善，则说明 line 15 和当前 modified Block-1 语义不兼容。如果没有明显改善，则保留 line 15，不再继续修改它。
>
> ## 3. 检查 `direct_estimate_x1`，但只判断是否值得改
> 当前 \hat x_1 = ((1-s_k)\hat x_end - (1-e_k)\hat x_start)/(e_k-s_k)。我们已经知道 stage 1/2 中 G != I，因此这个公式在 ker(G) 上存在系统缩放误差。不要直接修改。只检查：stage 1/2 的 \hat x_1 hole / range(G) / ker(G) error；这个误差相对于 Block-1 mean 和最终 hole error 是否足够大。如果它不是当前主要误差源，就保持不动。如果它明显限制 reconstruction，再单独给出一个最小的、保留 G 的修正方案，但先不要自动实施。
>
> ## 4. 不再测试这些已确认问题
> 不要继续测试：G^2=G；self-adjoint；`H_tau_inv` algebra；ridge sweep；CG vs closed-form；tau=0 singular guard；大规模 Block-2 sweep。
>
> ## 5. 最终决策
> 实验完成后，只分成 MUST FIX / KEEP / OPTIONAL。不要因为"理论上可能更好"就自动改代码。
>
> ## 6. Debug 和修改必须 minimal
> 一个问题能改几行就不要改几十行。不要新建复杂 diagnostic framework、重构 sampler、增加新超参数、同时修改多个因素、加大量永久 debug branch。测试尽量放在一个临时脚本或极少量 instrumentation 中。
>
> ## 7. 测试完成后彻底清理
> 最终保留 results/algorithm2_diagnosis/{report.md, comparison.csv, 必要的关键图片}，删除临时 test script / temporary debug branch / 临时 plotting code / 无关中间 tensor/results / 一次性 log / 为本次测试加入的 instrumentation。确认正式的 main.py / main2.py / utils.py 以及 Algorithm 2 实现恢复简洁。如果某个 correctness fix 最终被证明必须保留，只留下这个最小修改。
>
> ## 8. 最后只给我这些内容
> Tested / Results location / Key results / Analysis（观察到什么 → 排除了什么 → 支持什么原因）/ Decision（MUST FIX / KEEP / OPTIONAL）/ Cleanup（删除了哪些临时代码；删除了哪些无关结果；最终正式代码还保留了哪些修改）。

## 结论

**MUST FIX: 无。KEEP: line 15、direct_estimate_x1、stage transition 全部原样。**

1. **transition**：upsample / mode / stage index / g_bypass / x_tau 构造式与参考实现
   （`PixelFlow/IP_package/ms_posterior_sampling_article_version_final.py:586-634`，
   用户给的 `PixelFlow/` 那份是 439 行前缀、不含采样器）逐行一致。
   不一致的只有 x0：参考跨 stage 携带 latent_tau 并用 `sample_block_noise` renoise
   （实测 `‖G·n‖/‖n‖=4.9e-04`，噪声只在 ker(G)），Algorithm 2 用 fresh iid x0
   （一半能量在 range(G)）——这是论文 l.20 明文规定的偏离，docstring 已记录，非 bug。
2. **line 15**：`‖x_τ^after − x_τ^before‖/‖x_τ‖` 在 baseline 是 1.6e-08…9.6e-08
   （精确抵消，用户假设的机制成立），但去掉 restore 后最终洞区 0.0984 → **3.1247（差 31.7×）**，
   stage 3 内单调上升。原因：Block-1 抽样是噪声主导的，l.15 抵消掉的绝大部分是噪声不是语义；
   去掉它会把噪声灌进网络输入，x1_hat 洞区 0.081→0.625→1.336→3.123 正反馈恶化。
3. **direct_estimate_x1**：与参考实现逐字相同。stage 1/2 洞区误差 86–90% 在 range(G)，
   ker(G) 只占 10–14%，精确 `/1.2` 修正只值 0.19–0.74%；stage 3（产出最终图）ker(G) 为空、
   修正无效。不是瓶颈。

产物：`PixelFlowICLR/Algorithm2/results/algorithm2_diagnosis/`
（`report.md`、`comparison.csv`、`line15.png`、`l15/`、`no_l15/` 各 6 张图）。
临时脚本已删；正式 `main.py`/`main2.py`/`utils.py` 未因本次诊断改动
（`utils.py` 的唯一改动仍是用户自己的 prior-injection 编辑，17+/1−）。
