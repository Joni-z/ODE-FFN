# Synthetic ODEFFN：跨分布验证与 K=1 消融

用户授权：2026-10-06，继续完成 Ring-Gaussian / Two Moons 的现有三模型对照，并增加 K=1 对照。所有 run 只使用 seed 0。

## 要回答的问题

1. Gaussian pilot 的现象能否在多峰分布和弯曲分布上重现？原版是否仍保持有限的项幅度，谱约束是否改善拟合或生成？
2. 删除二阶项是否改变结果？如果 K=1 与 K=2 相近，就缺乏证据把效果归因于二阶结构。
3. 限制 P 的收益是否依赖 K？分别比较 K=1 与 K=2 下的原版 / 约束版差异，观察约束是否主要影响二阶项。

## 实验矩阵

| 阶数 | 数据集 | 模型 | 新 run 数 | 配置 |
|---|---|---|---:|---|
| K=2 | Ring-Gaussian、Two Moons | FFN、原版 ODEFFN、谱上限 ODEFFN | 6 | `configs/synthetic/train_v1.json` |
| K=1 | 各向异性 Gaussian、Ring-Gaussian、Two Moons | 原版 ODEFFN、谱上限 ODEFFN | 6 | `configs/synthetic/train_k1_v1.json` |

Gaussian K=2 与 FFN 的三个 pilot 结果直接保留使用。总共 15 个配置，新增 12 个 run。普通 FFN 不存在 K，不重复训练。

K=1 每个 ODELayer 输出 `x + s(x) P_eff x`；K=2 额外包含 `0.5 s(x)^2 P_eff^2 x`。ODEFFN 两个 ODELayer、中间 SiLU、外部网络均保持不变。K=1 与 K=2 的参数数量及 seed 0 的初始权重完全一致。原版和谱上限各自作阶数比较，不修改 s(x) 的设定。

新配置仅改变 experiment_id、模型列表（不重复 FFN）与 orders=1。数据文件、5000 步、batch 512、优化器、学习率、评估频率及 Heun 32 / NFE 64 保持与 K=2 一致。新输出放在 `../runs/synthetic/odeffn_synthetic_k1_v1`，K=2 输出仍在 `../runs/synthetic/odeffn_synthetic_v1`。

## 事先确定的比较口径

- 固定 5000 步的最终结果是主比较点，不按每个模型的最佳 checkpoint 挑选结果。完整曲线用于检查收敛过程；同时报告 3000–5000 步评估点的平均误差作为诊断，不作为独立重复实验或置信区间。
- Gaussian 与 Ring-Gaussian：对已知解析速度场的 validation_oracle_mse 衡量拟合误差；同时报告 SW2。Two Moons 没有本项目实现的解析 oracle，只报告固定配对 MSE 与 SW2，不能把配对 MSE 当作纯拟合误差。
- 报告实际 P 的最大奇异值、谱形状、稳定秩以及 s(x)、一阶项、二阶项的实际隐藏输入 RMS。K=1 的二阶项指标记为空，而非伪造为测得的零。约束版区分 P_raw 与 P_eff。
- 单个 seed 只支持对当前 run 的观察，不推断统计显著性。SW2 的独立参考样本距离是有限样本量级的参照，不是严格下界或置信区间。
- 总体训练不发散，并不证明所有状态下的稳定性。P 范数较大也不能单独证明不稳定；还需看实际项、门控与误差轨迹。
- 从日志记录 GPU 型号。循环耗时包含评估和诊断开销；不同型号 GPU 的墙钟时间不直接用于模型效率排名。

## 调度

每个任务申请一张 `a100|l40s|h100|h200` 中可用的 GPU，4 CPU、16 GB RAM、20 分钟时限，每个数组最多 3 个并行任务。先提交 Ring-Gaussian / Two Moons 的 K=2 数组，K=1 数组原计划依赖其成功完成（afterok）；2026-10-09 恢复连接时前批已经全部正常完成，核验后直接提交 K=1，使这批新实验最多同时使用 3 张 GPU。

两个脚本分别为 `scripts/train_synthetic_remaining.slurm`（index 3–8）和 `scripts/train_synthetic_k1.slurm`（index 0–5）。不覆盖旧结果、不改动已完成 pilot 的训练配置。

## 作业记录

- 2026-10-06：K=2 跨数据集数组已提交，job ID `19283509`，index 3–8。随后 SSH 认证失效，K=1 尚未提交；已同步的脚本等待认证恢复。K=1 的六项配置映射、数据哈希、初始权重一致性、公式和梯度及两个模型 dry-run 已通过。

汇总入口：`python3 scripts/summarize_synthetic.py --json-output ../runs/synthetic/ablation_summary.json`。汇总程序只读训练结果，报告完成状态、固定最终误差、后期平均误差、完整曲线和 P 的初始 / 最后谱与项幅度。全部结束后可加 `--require-complete`，缺少任一 run 的完成标记、5000 步指标或最终 checkpoint 时返回失败。

- 2026-10-09：SSH 恢复；19283509 的六个任务均 COMPLETED/0:0。九个 K=2/FFN run 的 COMPLETE 标记与 5000 步 checkpoint 均存在。K=1 六个输出目录尚不存在，已正式提交数组作业 `19469259`（index 0–5，只用 seed 0）。前批已完成，无需继续设置 afterok。

- 2026-10-09 最新核验：K=1 的 Gaussian / Ring8 四个任务正常完成；Two Moons 两个任务仍因 QOSGrpGRES 排队，预计开始时间 N/A。当前共 13/15 个 run 完成。汇总 JSON 已缓存到本地 `analysis-2026-10-09/summary.json`，对应远端 `../runs/synthetic/ablation_summary_2026-10-09.json`。绘图脚本 `scripts/plot_synthetic_ablation.py` 已同步并要求 15/15 完成后才生成图；完整比较尚待最后两项结果。

- 2026-10-13 最终核验：15/15 run 均完成 5000 步。K=1 数组 19469259 六项均 COMPLETED/0:0；最后两项已于 Slurm 记录的 2026-10-09 完成。全部训练共享数据 manifest 与 synthetic 源码哈希；配置、seed 0、实际 GPU 已逐项核对。完整分析与三组 PNG/PDF 图见远端 docs/analysis-2026-10-13/analysis-report.md。没有提交新增实验。
