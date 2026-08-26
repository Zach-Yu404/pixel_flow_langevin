# eq19-endpoint-prior（2026-08-26，done）

## 用户原始要求（逐字摘录）
「请基于当前 Algorithm 4 测试新的 prior-aware Eq. (19′)…… μ_τ = N⁻¹[(1−s)x̂_e
−(1−e)x̂_s]……[N²+γ²H²+γ²σ²S⁻¹]x̂₁ = N[(e−s)x_τ+σv]+γ²Hx_τ+γ²σ²S⁻¹μ_τ……
endpoint_mode: original/endpoint_prior……检查现有 endpoint_prior_shrink……
如果重复删除……」（全规格见对话）

## 结果：结构性零效应
4 种子：pooled 0.1499→0.1492（±0.0006）、spectral 0.1401→0.1406（±0.0046），
spread 不变。机理（inner 级 counterfactual 对照钉死）：原 (19) 解 =
(N²+γ²H²)⁻¹(N²μ+γ²Hx_τ)，μ 权重全程主导 ⟹ hat≈μ（逐帧差 ≤0.001）；
(19′)-μ 只是把 μ 侧权重 +γ²σ²S⁻¹（≈6%）——锚在与解几乎相同、且同携冻结
污染的对象上。对照 m=0 零锚变体（batch-7 有效：0.1227/0.1290）：**收缩项
是否有效完全取决于锚点是否独立于 x_τ**。
验证全过：恒等式 8.9e-16、μ(exact v)=x₁ 1.8e-15、original 逐位复现修复
预条件后基线（spectral 0.1487；0.1490 为截断时代旧值）、4 臂求解审计零违例。
endpoint_prior_shrink 已删（单开关原则）；要 12–14% 收益需恢复零锚为
endpoint_mode="prior_zero"——用户裁决。
产物：results/alg4_eq19_endpoint_prior/（report+6 CSV+trajectory.png+各臂
montage/关键帧张量），39MB。
