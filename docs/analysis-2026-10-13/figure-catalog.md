# 图目录与解读

所有图使用 summary.json 中 15 项已完成 run。PNG 与 PDF 已实际生成；代码为项目根目录中的 scripts/plot_synthetic_ablation.py。图没有推断误差条；每配置 n=1、seed 0。

## figure-01-final-generation

Purpose：比较固定终点生成效果，观察阶数和谱上限是否有一致优势。变量：各数据集 5000 步 SW2，越低越好。斜线填充为 K=1；横线是两份独立目标样本之间的参考距离，不是严格噪声下界或 CI。

Caption requirements：标注 seed 0、5000 步、2048 样本、128 投影、Heun 32 / NFE 64，无独立重复和推断误差条，各 panel Y 轴不同。

Observation：Gaussian 原版 K1 最低，Ring8 FFN 最低，Moons 原版 K2 最低。

Interpretation：不能给 K2 或谱上限一个跨分布的普遍排名。

Checklist：核对每根柱与精确数表；不把三个任务视为重复实验；不把少量距离差声明为显著。

## figure-02-training-dynamics

Purpose：检查终点排名是否代表训练过程，并对比拟合与生成指标。上排 Gaussian/Ring8 是解析 oracle MSE，Moons 是配对 MSE；下排是每 1000 步 SW2。Y 轴为 log。

Caption requirements：固定评估样本与采样源；线型区分 K；Moons 指标定义不同；评估点不是独立重复，不提供误差条。

Observation：Ring8 K1 的 oracle MSE 明显更早下降，却没有得到更低的最终 SW2；Gaussian K2 的终点优势不能简单解读成全程收敛优势。

Interpretation：下一步应定位误差在时间和采样过程中的作用，而非只比较最终 loss。

Checklist：不挑选每模型最佳 checkpoint；保留初始到 5000 的完整曲线；不直接跨 panel 比较不同指标数值大小。

## figure-03-operator-dynamics

Purpose：检查 P 的谱增长和实际二阶项幅度，区分干预是否生效与生成效果是否改善。上排是六层中最大的 P_eff 奇异值；下排是 K2 二阶项/输入 RMS 的六层均值。

Caption requirements：K1 没有二阶项，不伪造零值；说明测的是实际隐藏输入；六层均值不是统计置信估计；P_eff 是前向实际使用的算子。

Observation：原版 P 的最大奇异值增长，约束版维持在 1；约束版二阶项均值较低。

Interpretation：当前谱上限是有效的内部幅度干预，尚未被证明是生成收益或完整动力稳定性的原因。

Checklist：不把 P 范数增长称为发散；不把 s 的影响忽略；不把固定 P 的整体缩放称为奇异值均匀化。
