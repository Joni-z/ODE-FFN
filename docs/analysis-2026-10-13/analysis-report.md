# Synthetic ODEFFN 完整分析（2026-10-13）

15/15 个 run 已完成 5000 步，全部只用 seed 0。K=1 数组 19469259 六项均 COMPLETED/0:0，最后两项在 Slurm 记录的 2026-10-09 已结束；今天恢复 SSH 后完成整体核验。所有完成标记、最终指标与 checkpoint 齐全。

## 问题与比较口径

检验同参数量的两种 ODEFFN：二阶项是否提供收益，P 最大奇异值上限是否改善学习或生成。普通 FFN 是共享外部网络的对照；K=1 / K=2 各自有原版与谱上限版。5000 步终点为事先确定的主比较；后期均值是 3000、3500、4000、4500、5000 五个评估点的描述性均值，不是五次独立实验。全部采用相同数据、seed、采样源规则及 Heun 32 / NFE 64。

这是向量场 toy 模型，不能直接推断图像 DiT 的收益，也不能证明这种结构专门适配 flow-matching objective。

## 最终生成距离

SW2 越低越好。每格是一项完整训练的观测，不带跨 seed 的误差估计。

| 模型 | Gaussian SW2 | Ring8 SW2 | Two Moons SW2 |
|---|---:|---:|---:|
| FFN | 0.086822443 | 0.16312511 | 0.044774991 |
| ODE K=1 | 0.061060548 | 0.20082602 | 0.031507067 |
| Cap K=1 | 0.063206486 | 0.19695909 | 0.029898569 |
| ODE K=2 | 0.067436814 | 0.1848201 | 0.027083287 |
| Cap K=2 | 0.065567866 | 0.18892877 | 0.029728327 |

Gaussian 的独立参考样本距离为 0.0667713，Ring8 为 0.0915702，Two Moons 为 0.0210348。这些数值只表示本次有限样本参照的量级，不是严格下界或置信区间。

## 拟合与训练曲线

| 模型 | Gaussian 最终 oracle MSE | Gaussian 后期平均 | Ring8 最终 oracle MSE | Ring8 后期平均 |
|---|---:|---:|---:|---:|
| FFN | 0.0062054941 | 0.0064670959 | 0.027150708 | 0.052095407 |
| ODE K=1 | 0.0040504655 | 0.0055633547 | 0.020474799 | 0.032150607 |
| Cap K=1 | 0.0041104179 | 0.0060700894 | 0.02056009 | 0.033063892 |
| ODE K=2 | 0.0042422675 | 0.0065406126 | 0.028707141 | 0.098456588 |
| Cap K=2 | 0.0041576854 | 0.0065999775 | 0.022893521 | 0.05868032 |

Two Moons 没有本项目实现的解析速度 oracle，配对 MSE 含随机配对方差，不能与前两数据集的 oracle MSE 混为一个指标。

1. Gaussian：K=1 原版终点 SW2 最低；K=1 的最终和后期平均 oracle MSE 也不逊于 K=2。最初只看 K=2 最后一项得出的 ODE 优势，应弱化：K=2 的后期平均 oracle MSE 并未优于 FFN。
2. Ring8：普通 FFN 的最终 SW2 最低。K=1 的两个 ODE 模型拟合更快、最终 oracle MSE 更低，但生成 SW2 比对应 K=2 更高。K=2 的谱上限版最终 oracle MSE 优于 FFN，却有更高的 SW2。平均速度拟合误差和生成终点分布不是同一个结论。
3. Two Moons：所有 ODE 模型最终 SW2 低于 FFN，原版 K=2 的 SW2 最低；约束版 K=1/K=2 差异仅约 0.00017。配对 MSE 曲线整体非常接近，不能据此宣布显著差异。

## P 学成了什么

原版 P 的最大奇异值在训练中增加；所有原版的最终最大值约为 1.86–2.43。K=1 并不必然得到更小的 P：Gaussian 和 Two Moons 上其最大值反而高于 K=2，说明不能仅凭 P 的范数给训练质量排序。没有任务触发训练中的非有限 loss / gradient 检查失败。

K=2 的二阶项 / 输入 RMS，六个 ODELayer 的最终均值从原版的 Gaussian 0.255、Ring8 0.297、Moons 0.193，降到约束版的 0.177、0.167、0.135。谱上限确实抑制了实际二阶项，但没有转化成一致的生成收益。约束版的 s(x) 平均值又普遍增大，门控与 P 会共同调整。

当前约束为 `P_eff = P_raw * min(1, 1 / sigma_max(P_raw))`，对某个固定 P 是整体缩放，保留相对奇异值及稳定秩；它不直接实现“奇异值均匀化”。不同训练后谱形状的差异来自优化路径变化。

从数学上，`||P_eff||_2 <= 1` 且 `s(x) <= 1` 时，二阶项的向量范数相对输入至多 0.5；这只是点态幅度控制，不能直接当作整层 Lipschitz 或完整生成轨迹稳定性的保证，因为 `ds/dx` 和外部网络同样影响 Jacobian。

## 下一步决策

优先解释 Ring8 的拟合 / 生成分歧：按外部 FM 时间 t 分段评估 oracle 误差，并用相同 checkpoint 比较不同采样 NFE，区分误差出现的时间、沿轨迹放大与数值求解误差。此处只是下一步设计，没有提交额外训练或评估作业。

继续研究约束时，应明确要控制的是内部项幅度、完整 Jacobian 还是生成终点误差；当前结果不足以把“原版发散 → 约束修复”当作论文故事。

## Claim Candidates

- Claim：二阶项的收益依赖分布与评价指标。
  - Source evidence：summary.json 的 15 项终点，figure-01 与 figure-02。
  - Allowed wording：当前 seed 0、5000 步设置下，Gaussian 的 K=1 更好；Ring8 的 K=1 拟合更好而 K=2 生成更好；Two Moons 原版 K=2 生成更好。
  - Forbidden stronger wording：K=2 普遍更好，或二阶项完全没有用。
  - Uncertainty：每配置仅一次训练；不同 GPU；完整非线性网络的作用没有被逐项归因。
  - Next check：时间分段误差与固定 checkpoint 的 NFE 敏感性。
  - Decision：keep（保留为受设置限制的观察）。

- Claim：谱上限抑制了内部二阶项，但未产生一致的生成收益。
  - Source evidence：figure-03 与各数据集原版/约束版的最终 SW2。
  - Allowed wording：这些 run 中 P_eff 的最大奇异值保持约 1，二阶项平均幅度降低；最终生成效果有升有降。
  - Forbidden stronger wording：谱约束保证整体稳定，或谱约束对生成模型无效。
  - Uncertainty：只测试上限 1；门控发生了适应；未测完整 Jacobian。
  - Next check：先定义控制量，再决定约束形式。
  - Decision：keep。

- Claim：速度拟合误差改善不必然改善当前采样设置下的终点分布。
  - Source evidence：Ring8 的 oracle MSE 与 SW2 对应比较。
  - Allowed wording：同一采样协议下，这些 Ring8 run 的两项指标排序不同。
  - Forbidden stronger wording：已经证明分歧来自求解器误差，或已经证明 flow matching objective 不合适。
  - Uncertainty：缺少按 t 分段误差、不同 NFE、轨迹 Jacobian 及跨 seed 验证。
  - Next check：区分上述机制。
  - Decision：keep。

## 图与原始证据

- [最终生成距离](figures/figure-01-final-generation.png)
- [完整训练曲线](figures/figure-02-training-dynamics.png)
- [P 与二阶项轨迹](figures/figure-03-operator-dynamics.png)
- [精确数值与统计限制](stats-appendix.md)
- [图的用途、解读与限制](figure-catalog.md)
- summary.json：完整指标与曲线，保留机器精度。
- protocol.json：每个 run 的数据/代码/配置哈希、参数数、实际 GPU 与数组任务号。
