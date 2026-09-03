# 2026-09-03-fixed-spectral-class-S

任务：default-spectral-class-s
状态：已采纳（用户直接指令）

## 【共识】
目标：Algorithm 4 的先验协方差 S 固定为 s_stats 的 per-class 谱统计（spectral_class），
不再是 config 选项。

确认方案：
- main4.S_STATS 硬编码 s_stats/spectral_power_labelled.npz + LOC_synset_mapping.txt；
  default_s2_fn(K) → _SpectralS2（每 stage 一个 utils.SpectralSOp，按图片 class_idx 绑定）。
- config_alg4.json 删除 "S_prior" 段；CONFIG_SCHEMA 同步。make_s2_fn 的四种标量
  prescription 只作 --mode diversity 的消融臂保留。
- 用户原话：「现在默认使用spectral_class」→「我的意思是，这个不用作为一个选项，默认使用
  spectral_class的s_stats作为指定选项」。

是否需要实现：是（已实现）

## 依据
S4 测试（s_stats/test/s4_per_task.md，1440 行）：spectral_class 在 4 种 val-set S 中全面最优
（MMSE10 24.86 dB vs spectral_all 24.72；标量 val-set S 带噪单样本塌到 14.5 dB），且与旧的
14 图 spectral（含测试图，校准泄漏）持平或更好（no_noise 25.46 vs 25.26，LPIPS 更低）。
排除：pooled_all/pooled_class（噪声路径灾难）、旧 junco 表（单样本 18.8 dB）、
旧 14 图 spectral（泄漏）。

## 参与
【Claude｜实现+验证】；Codex 未参与（用户此前明确不需要该协议环节）。
