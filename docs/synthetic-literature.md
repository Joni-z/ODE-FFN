# Synthetic 机制验证：论文与官方实现核对

核对日期：2026-09-23。范围是“什么 toy 数据能回答当前 ODEFFN/P 约束问题”，不是全面文献综述或 SOTA 排名。直接核对原论文 HTML、正式发表页面和作者维护代码；没有把第三方教程作为论文实验依据。

## 有直接参考价值的先例

1. **Gaussian Mixture Flow Matching Models（GMFlow，ICML 2025）**
   - 论文 §4.1 用二维 Checkerboard，采用相同的 5 层 MLP，只改变不同方法所需的输出通道，观察样本直方图与采样步数变化。
   - 本文支持：简单二维数据仍被近期高水平工作用于分析机制。它研究的是速度分布参数化与采样器，不是 ODEFFN 的 P 谱约束，不能直接援引其结论证明本项目设计有效。
   - [正式发表页](https://proceedings.mlr.press/v267/chen25cl.html)；[论文 §4.1](https://arxiv.org/html/2504.05304#S4.SS1)。

2. **Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport（OT-CFM，TMLR 2024）**
   - 论文 Table 2 的四个任务是：标准 Gaussian → 8 Gaussians、Moons → 8 Gaussians、标准 Gaussian → Moons、标准 Gaussian → S-curve。
   - 报告 W2 squared、normalized path energy、运行时间，并使用五个种子。
   - 官方 TorchCFM README 的可视化另有 8 Gaussians → Two Moons 和单 Gaussian → Two Moons；注意它与论文表格的源/目标方向并不完全相同。
   - 借鉴分布家族与轨迹诊断，不在本轮同时改变 coupling/objective，以免混淆 FFN 架构的作用。
   - [论文实验部分](https://arxiv.org/html/2302.00482#S5)；[官方代码](https://github.com/atong01/conditional-flow-matching)。

3. **Meta 官方 flow_matching 的 2D 示例**
   - Notebook 明确使用 Checkerboard、线性调度和 MLP 速度场，模型接收二维坐标和时间。
   - 它是可运行的官方示例，不是新的独立论文或架构对比结果。
   - 官方示例对棋盘坐标除以 0.45；本项目使用未额外缩放的 [-2,2) 棋盘，因此属于同分布家族，不是参数级复现。
   - [官方 notebook](https://github.com/facebookresearch/flow_matching/blob/main/examples/2d_flow_matching.ipynb)。

## 对本项目的具体选择（我们的判断）

第一轮只需三个任务，所有源分布固定为标准 Gaussian，所有模型固定相同的 independent linear FM objective：

1. **各向异性 Gaussian**：解析最优速度场已知，控制不同方向的收缩/扩张；判断约束是否降低拟合误差或过度限制表达能力。这是本项目针对 P 的诊断设计，不声称照搬上面论文的具体 Gaussian 参数。
2. **8-Gaussian ring**：与 OT-CFM 的常见任务家族对应；在保留精确速度场参考的同时加入多模态和空间变化。
3. **带噪 Two Moons**：观察弯曲几何、轨迹、采样误差；本实现没有速度场 oracle，使用分布指标及配对验证 loss，且后者不等于对最优场的误差。

Checkerboard 留作第二轮压力测试。16D Gaussian 与各向异性 4-GMM 已准备但暂不扩展训练矩阵。精确的 noise/variance/scale 以 `distribution.json` 为准，不因为名称相同就宣称复现论文。

二维输入也可以经过投影进入较宽的隐藏空间，因而内部 P 不必是 2×2；这让我们能在易观察的数据上研究 64/128 维 P 的谱。模型宽度、参数量、有效计算与训练预算应对齐或明确报告。

Gaussian 数据的解析速度场是整个模型输出的参考，不是隐藏层 P 的唯一真值；隐藏表示可以改变基底，不能要求学出的 P 直接等于数据空间的线性速度矩阵。

仅观察 P 的谱与 loss 的相关性不等于因果证明。需要同架构受约束/无约束的干预对照，并同时记录步长网络、实际 Taylor 项、层 Jacobian 或更新放大量。这里的“内部隐藏特征演化”也不自动等于“外部数据生成流”。

## 调研工具说明

本次使用 research-lookup 技能组织证据核对；Parallel CLI 在本机不可用，实际检索使用可用的网页检索工具并回到一手来源验证。没有使用付费 Research API 或安装新工具。

技能出处（工具说明，不属于上述方法论文）：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. [arXiv:2609.00065](https://doi.org/10.48550/arXiv.2609.00065)。作者与年份已核对当前 arXiv 页面。
