# finalize-alg4-config（2026-09-05，running）

## 用户原始要求（逐字）
「现在gamma2我用gamma2_all的，现在给我一个finalize的版本和我check一下」

## 做法
1. config_alg4.json paths.gamma2_table → {HERE}/gamma2_stats/gamma2_all.json（decisions/2026-09-05-gamma2-all-default.md）。
2. 三视角独立审计（workflow wf_14a6b8cc：代码/配置一致性、新机复现、结果簿记）。采纳：
   - **删除 tasks_setup.random_inpainting.kw.num_langevin=15**（full_ip 下会覆盖 [2,2,1,1]；所有证据实验都强制
     [2,2,1,1]，而首次参考跑正用 15 → 停掉重跑）；
   - results/alg4（8-24 旧 S/旧 γ²/num_langevin 10 的 full_ip 输出）改名 results/alg4_pre_final_0824，最终参考跑
     直接写 config 默认的 results/alg4（resume 逻辑会跳过已存在格，故必须先移走）；diagnose.compare_to 随之指向最终跑；
   - main4 docstring sigma_min 0.39 → 注明 config 1e-8；utils 注释 γ² 表来源改 gamma2_all；full_ip 循环每格打印
     已绑定类的 Tr(S_k)/D；gamma2_stats/.gitignore（claims/log）；CONSTRAINTS 中 γ² 表指向更新；
   - 记录：gamma2_stats/test 是唯一与最终配置完全一致的实验（其 gamma2_all/seed42 box 行与参考跑逐格一致到 4 位）；
     S4（s_stats/test）用的是旧 7 图 γ² 表（差 ≤0.03 dB，近似参考）；all_img_tests 用旧 S 臂，仅作历史参照。
   - 新机复现硬依赖（未入库）：PixelFlow/IP_package（demo_runner/pipeline/demo 图/模型 config+权重）、DAPS clone、
     ImageNet val + LOC 文件（main4.S_STATS 绝对路径）、s_stats npz（须重建）、conda env pixelflow、PYTHONHASHSEED=0。
3. PixelFlowICLR/Algorithm2/FINAL.md：固定选择、config、统计量重建、运行命令、参考结果表（待跑完填入）。

## 状态
running：最终参考跑 `PYTHONHASHSEED=0 python main4.py`（默认 config，out=results/alg4）GPU 0 进行中（~15 min + EIO）。
