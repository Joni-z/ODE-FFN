# ODE-FFN：Report2 教授汇报备忘

Zhizhe Zhang · 准备于 2026 年 11 月 9 日

依据：[已定稿 Report2](../../../reports/report2/report2.pdf)。这是口头汇报准备，
报告中的实验、数字和结论保持原样。默认按五分钟准备；教授打断时，直接进入下方问答。

## 一分钟开场

> 我这一阶段想弄清楚，原版 ODEFFN 里哪些动力学结构值得保留。首先检查了内部时钟、
> Taylor 阶数和 P 的谱，发现更高阶或更平的谱没有稳定改善生成。然后沿着 Muon 和
> 输入依赖的 PD(h) 做了探索，得到一个有界的 ISRU-D 特征响应。它在 Ring8 的场拟合上
> 有优势，但普通 Free ISRU 和 SwiGLU 说明，这个收益还不能归到 ODE 结构本身。
> 最新的发现是：继续训练后，场误差和去中心后的残差分布更好，各模式的样本占比却更偏。
> 我把这个现象联系到模式边界上的净流量。下一步想先验证这个量能否预测模式分配，
> 再决定 P 或特征响应应该受到什么约束。

English opening:

> I studied which parts of the original ODE-inspired FFN provide a useful bias
> for flow matching. I tested the internal clock, Taylor order and P spectra,
> then explored Muon and input-dependent diagonal operators. A bounded ISRU-D
> response improves field fitting on Ring8, while ordinary controls give
> comparable sampling performance. The latest checkpoint analysis shows a
> specific tradeoff: field fitting and centered residuals improve, but mode
> probabilities drift. I derived a boundary-flux response for this imbalance.
> My proposed next step is to calibrate that response before choosing a new
> constraint on the feature transformation or its updates.

## 五分钟怎么讲

| 时间 | 看报告哪里 | 只讲这一件事 |
|---|---|---|
| 0:00–0:40 | 第1页开头、原版公式 | 目标是为 flow matching 找有用的 FFN 归纳偏置；内部特征时间和全局生成时间不同。 |
| 0:40–1:40 | 第1页 Figure1 | Muon 的更新谱变宽，但 P 的谱范数仍能增长；只看 P 还不足以选择约束。 |
| 1:40–3:00 | 第2页 cubic/PD 和 Table1 | 从 cubic 精确响应得到 ISRU，构造 ISRU-D；拿 Free ISRU 和强 SwiGLU 对照判断收益。 |
| 3:00–4:15 | 第3页 Figure2 和轨迹案例 | 残差分布改善与模式人数失衡同时发生；早中期路径影响后续模式分配。 |
| 4:15–5:00 | 第3页末尾 | 用冻结模型校准边界净流量，再讨论一个针对功能响应的约束。请教授判断这个设计目标是否值得优先。 |

完整梯度读出和条件方向实验，是第2页支持研究逻辑的两个例子。教授问到时再展开：
前者两种 backbone 都受益；后者在薄目标上强，增加平面外方差后采样优势消失。

## 三组数字卡片

### 1. Muon 改变了什么？第1页

Product16，原版 scalar K2，20k；AdamW 矩阵学习率 .001，Muon .02，六个 P 的均值。

| 指标 | AdamW | Muon |
|---|---:|---:|
| 更新 stable rank | 3.08 | 32.74 |
| P 的最大奇异值 | 3.37 | 19.17 |
| Field MSE | 0.03343 | 0.02385 |

说法：Muon 作用在矩阵更新上；训练后的 P 并不因此保持奇异值均匀或范数很小。
普通 SwiGLU192 的 MSE 是 0.01567。范数匹配的 momentum 控制也没有复现全部收益。

### 2. 当前候选如何？第2页

Ring8，20k，Heun128；SW2 是八个共同采样 bank 的均值。

| 模型 | Field MSE | SW2 |
|---|---:|---:|
| ISRU-D / AdamW .0003 | 0.00977 | 0.17110 |
| Free ISRU / AdamW .0003 | 0.01000 | 0.17930 |
| SwiGLU64 / AdamW .0003 | 0.01372 | 0.19453 |
| SwiGLU64 / Muon .02 | 0.01128 | 0.17145 |
| SwiGLU192 / Muon .02 | 0.01123 | 0.18399 |

说法：ISRU-D 保留了 Ring8 拟合收益，采样与 selected SwiGLU64 接近。
Product16 的八-bank SW2 是 ISRU-D 0.08900、Free ISRU 0.08596、selected SwiGLU192
0.08350，普通模型更好。Ring8 的结果不能替代这个任务差异。

### 3. 最新发现具体是什么？第3页

ISRU-D，固定训练轨迹的两个 checkpoint；pool 使用 32,768 个保存的生成样本。

| 指标 | 15k | 20k |
|---|---:|---:|
| Oracle field MSE | 0.01026 | 0.00977 |
| pooled native SW2 | 0.11727 | 0.13261 |
| 减去所分配模式中心后的 SW2 | 0.01768 | 0.01502 |
| 模式扇区占比相对 1/8 的 L1 | 0.03827 | 0.05450 |

直觉说法：减去各个团中心后，剩余点云更接近目标；但八个团的样本人数更不均匀。
Free ISRU 和 SwiGLU 也有这个方向的变化。各团协方差并非全部改善，
所以把图中间那一项叫“合并的去中心残差分布”，不要泛称每个团内部都拟合更好了。

案例：ISRU-D20k 的同源 oracle 终点 MSE 是 0.04934。前75%使用 oracle、最后25%使用
学习场时，是 0.00075；前75%学习、最后25%换 oracle 时，是 0.05009。
这说明这组源样本的模式去向主要在早中期形成。

## 教授可能追问什么

### 1. 你到底训练的是什么模型？和 JiT 有什么关系？

当前机制实验使用 width64、三个残差块的时间条件向量 MLP。输入是带噪坐标和 t，
输出速度。训练目标是独立噪声—数据配对的速度标签；评估时利用已知分布计算解析
oracle 速度。原版 ODEFFN、ISRU-D 和普通 FFN 在这个 backbone 中替换比较。
Ring8 是八个等权二维高斯团；Product16 是16个双峰因子的乘积分布再旋转。
这些目标分布可解析计算理想速度，方便分开检查场误差和采样误差。生成分布用 SW2
比较，即沿共同方向投影后比较一维分布；FID 则留给图像实验。
JiT 是最终应用，图像 baseline 还没有达到用来判断 FFN 效果的质量要求。

### 2. 原版到底哪里是 ODE？内部时间和 t 是不是一回事？

```text
单个原版层：h + s(h) P h + 0.5 s(h)^2 P^2 h
整个原版 FFN：T2(SiLU(T1(h)))
```

单个层冻结入口处算出的 s(h)，近似线性系统 dy/dr=P y 的局部演化。
同一 P 在该层的 Taylor 项里共享，两个层各有自己的 P。r/s 是特征空间的内部时长；
t 是外部噪声到数据的生成时间。研究问题是这种参数化是否有用，二者目前没有强制对应。

### 3. P 大了，级数是不是就发散？

带阶乘的矩阵指数 Taylor 级数对任意有限 P 都收敛。需要检查的是有限阶近似、方向
放大和输入门控的导数。K1 的完整 Jacobian 已经多出一项：

```text
J = I + s(h)P + (P h) [grad s(h)]^T
```

P 的谱是其中一部分。测试过的谱限制没有提供一个普遍改善生成的方案。

### 4. ISRU-D 与普通 ISRU 的差别在哪里？为什么还研究 ODE？

```text
q(h) = h / sqrt(1 + 1.2 h^2)       # 逐元素
ISRU-D 的一个 K1 层：F(h) = h + P q(h)
ISRU-D FFN：F2(SiLU(F1(h)))
普通 Free ISRU：A q(B h)
```

q 是 dy/dr=-y^3 在 r=.6 的精确响应。这给了一个明确的构造来源；完整 ISRU-D
并不是整个耦合非线性 ODE 的精确解。两个网络还改变了残差位置、非线性次数和初始化。
有界的是 P 作用的 q；这不等于整个残差层或网络是收缩映射。
下一项归因控制要匹配这些因素，再判断约束是否值得保留。

回答“为什么继续”的核心：现在要检验一个具体动力学目标——模式概率分配——能否
指导 FFN 的功能响应。ODE 参数化是否更适合实现它，需要和普通参数化正面比较。

### 5. MSE 变好为什么生成变差？是不是你已经找到了原因？

已测到的是误差组成的变化：整体场误差下降，去中心残差分布改善，但模式人数更偏。
整体 MSE 不单独决定边界上把样本推向哪个模式的法向误差。
我们有模式质量变化的连续性方程，并推导了 oracle 附近的一阶响应；
这个预测量能否准确解释六个已训练场，是下一项验证。

### 6. 你说早期决定模式，这能解释整体 SW2 吗？

替换实验针对预先指定的共同 source bank，比较同一噪声对应的 oracle 终点。
那个 bank 在20k时的 SW2 反而改善，整体八-bank均值则变差。
因此它定位的是路径和模式去向差异。报告同时保留全部 pool/leave-one-bank-out 结果，
整体分布比较另用共同 bank 的 SW2 和模式人数。两种问题分别回答。

### 7. 边界净流量和“跨过边界的次数”有什么不同？

用两个相邻房间想象：向左走10人、向右走10人，跨越20次，两个房间人数仍然不变。
模式人数变化由进出相减后的净流量决定。Ring8 的理想场沿模式边界切向运动；
学习场边界上的净法向流量会改变模式占比。

要记住的数学链条：

```text
continuity equation -> 每个扇区的概率质量变化
Ring8 对称性 -> oracle 没有穿越扇区边界的法向速度
oracle 附近的小误差 -> B_k = -∫dt ∫边界 p_t (v-u)·n_k dS
改变一个 P -> D_P v[ΔP] = J_down ΔP q(h)
```

第一条质量守恒关系是精确的；B_k 是有光滑性条件的一阶预测。它连接到实际特征和
下游网络，再连接到 P 更新。不能从单独的 P 奇异值推出这个量。

## 最后希望和教授决定什么

建议的第一个有限步骤：用现有六个冻结场计算 B_k，做弱 oracle/model 插值，
比较预测和实际模式占比变化。先回答“这个量预测得准不准”。

- 预测准确：再设计一个改变边界响应的 FFN/更新控制，与普通 FFN 和匹配幅度更新比较。
- 预测不准：检查非线性密度响应和数值校准，暂不把一阶推导变成训练约束。

可以问教授：

> 我想把模式概率的误差作为一个可测量的设计目标，先校准边界流量，再回到 Pq(h)
> 和参数更新上做控制。您认为这个目标值得优先，还是应先把 ISRU-D 与 Free ISRU 的
> 初始化、残差位置完全匹配，再判断是否有结构收益？

## 证据入口

- [定稿状态和哈希](../../../reports/report2/finalization.json)
- [报告的数字绑定](../../../reports/report2/data/transport-evidence.json)
- [完整共同-bank比较](../../autoresearch/professor-20261103/cycle14-results-complete/analysis-report.md)
- [共同轨迹、形状和模式分配](../../autoresearch/professor-20261103/cycle15-results/analysis-report.md)
- [统计与数值口径](../../autoresearch/professor-20261103/cycle15-results/stats-appendix.md)
- [完整推导与假设](../../autoresearch/professor-20261103/mode-mass-derivation.md)

准备时先练一分钟开场，再看第1、2、3页各讲一个结论，最后演练上面的七个追问。
