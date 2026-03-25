# JiT Codebase 中 12 种 FFN Block 设计总览

本文档对 `src/ffn_blocks.py` 中已经实现的所有 FFN block 做统一整理，目标是方便组会或向教授汇报时直接说明：

- 每个 FFN 在 JiT block 里扮演什么角色
- 与 baseline MLP / SwiGLU 相比到底改了什么
- 时间条件 `t` 或条件向量 `c` 是如何进入 FFN 的
- 数学形式上发生了什么变化
- 各方案的直觉、优点、风险和适用位置

---

# 1. 总体背景：这些 FFN 在哪里使用

本项目不是标准分类 Transformer，而是一个 JiT-like 的 class-conditional image generation model。  
在每个 `JiTBlock` 中，attention 和 FFN 都由共享条件向量 `c` 调制。

block 主体可以写成：

\[
x' = x + g_{\mathrm{msa}}(c)\cdot \mathrm{Attn}(\mathrm{AdaLN}_1(x,c))
\]

\[
y = x' + g_{\mathrm{mlp}}(c)\cdot \mathrm{FFN}(\mathrm{AdaLN}_2(x',c), c)
\]

其中：

- `AdaLN` 表示由条件向量调制的归一化
- `FFN(\cdot, c)` 是本文讨论的对象
- 外层 residual 和 gate 在 JiT block 中统一处理

所以本文所有 FFN 的输出都应理解为：

- 它们不是直接输出最终图像
- 它们输出的是一个 token-wise feature update
- 最终由 JiT block 外层 residual 加回主干

---

# 2. 条件与记号约定

为了统一描述，下面固定以下记号。

输入 token feature：

\[
x \in \mathbb{R}^{B\times N\times D}
\]

其中：

- \(B\): batch size
- \(N\): token 数
- \(D\): hidden dimension

JiT block 传入的条件向量：

\[
c \in \mathbb{R}^{B\times D}
\]

在当前代码里，`c = t_{\mathrm{emb}} + y_{\mathrm{emb}}`，即：

- diffusion / flow 时间嵌入
- 类别嵌入

二者相加后的共享条件向量。

很多 block 会先把 `c` 转成某种时间/步长标量：

\[
s = \sigma(z / \tau)\cdot \mathrm{scale} + \mathrm{shift}
\]

这里：

- \(\tau\) 控制温度
- `scale` 和 `shift` 把步长限制在一个稳定范围

这样做的核心目的是：

- 不让 FFN 内部步长失控
- 让动力学强度有明确上界和下界

---

# 3. 一个自然的分组方式

这 12 个 FFN 可以按激进程度分成 4 层：

## 第一层：经典 baseline

1. `mlp`
2. `swiglu`

## 第二层：把 FFN 改造成动力系统

3. `ode`
4. `ode_swiglu`
5. `mh_ode_swiglu`
6. `headwise_ode_value_glu`
7. `lowrank_state_ode`
8. `tied_flow`

## 第三层：显式的导航/细化分工

9. `nav_refine`
10. `time_split`

## 第四层：更贴近 flow / denoising 语义

11. `clean_target`
12. `time_moe`

下面按照这个顺序逐一展开。

---

# 4. Baseline 类

## 4.1 `mlp`: 标准 Transformer FFN

### 结构

代码中对应两层全连接加 GELU：

\[
h = \mathrm{GELU}(W_1x + b_1)
\]

\[
\tilde h = \mathrm{Dropout}(h)
\]

\[
y = W_2\tilde h + b_2
\]

### 设计含义

这是最普通的 token-wise feature mixer：

- 不使用时间条件
- 不使用状态依赖步长
- 不显式区分 coarse / refine
- 不包含 ODE 结构

### 优点

- 最稳定
- 最容易训练
- 是所有新结构的对照组

### 局限

对于 flow matching / rectified flow 来说，它默认：

- 所有时间段使用同一个静态映射
- 早期高噪阶段和后期低噪阶段没有机制上的分工

也就是说，它更像通用 feature mixer，而不是时间相关动力学模块。

---

## 4.2 `swiglu`: GLU 化 baseline

### 结构

SwiGLU 先产生 gate 和 value 两路：

\[
[u,v] = W_{12}x
\]

\[
g = \mathrm{SiLU}(u)
\]

\[
h = g \odot v
\]

\[
y = W_3(\mathrm{Dropout}(h))
\]

### 与 MLP 的差别

MLP 是：

\[
y = W_2\phi(W_1x)
\]

SwiGLU 是：

\[
y = W_3\big(\mathrm{SiLU}(W_a x)\odot W_b x\big)
\]

因此它引入了：

- 显式门控
- 更强的乘性非线性
- 更好的表示能力

### 设计动机

SwiGLU 在大模型里常常比普通 GELU MLP 更强，因为：

- gate 可以实现动态通道选择
- value 分支负责真正的内容承载
- gate/value 解耦使表达更灵活

### 仍然存在的问题

虽然比 MLP 强，但它依旧是静态 FFN：

- 不显式看时间
- 不显式建模步长
- 不把 FFN 看成一个小动力系统

所以在本项目里，它是强 baseline，但仍不是 flow-aware 设计。

---

# 5. ODE 化 FFN：把静态映射变成动力学更新

## 5.1 `ODELayer`: 所有 ODE 系列的基本原语

很多后续 block 都依赖一个内部模块 `ODELayer`。因此先单独说明它。

### 结构

`ODELayer` 先从输入或条件中生成一个标量/向量步长：

\[
s = \sigma(\mathrm{TimeMLP}(x \ \text{or}\ c)/\tau)\cdot \mathrm{scale} + \mathrm{shift}
\]

然后对一个线性动力系统做有限阶展开：

\[
z_0 = x
\]

\[
z_{k+1} = \frac{s}{k+1}P z_k
\]

最后求和：

\[
\mathrm{ODELayer}(x,s) = \sum_{k=0}^{K} z_k
\]

其中：

- \(P\) 是可学习线性算子
- \(K=\text{orders}\)

### 数学直觉

如果把连续系统写成：

\[
\frac{dz}{dt} = Pz
\]

其解析解是：

\[
z(t) = e^{tP}x
\]

而矩阵指数的泰勒展开为：

\[
e^{tP}x = x + tPx + \frac{t^2}{2!}P^2x + \cdots
\]

代码中的 `ODELayer` 本质上是在做这种低阶近似动力学传播，只是用递推近似而不是显式求矩阵幂。

### 设计意义

相比普通线性层，`ODELayer` 的重点不在“混合一下特征”，而在：

- 给特征定义一个随时间变化的传播强度
- 让 FFN 看起来像一个有限步的连续动力系统近似

---

## 5.2 `ode`: 纯 ODE 残差 FFN

代码注册名 `ode` 对应的是 `ODEOnlyFFN`。

### 结构

先做 ODE 传播：

\[
\hat x = \mathrm{ODELayer}(x, c)
\]

取其相对输入的增量：

\[
\Delta = \hat x - x
\]

如果启用归一化，则把增量 RMS 对齐到输入量级：

\[
\Delta \leftarrow \Delta \cdot \frac{\mathrm{RMS}(x)}{\mathrm{RMS}(\Delta)}
\]

然后乘以一个全局可学习门控：

\[
\gamma = \sigma(\theta)
\]

\[
y = \gamma\cdot \Delta
\]

### 设计动机

这个版本最激进地抛弃了传统 FFN 的两层投影结构，直接假设：

- FFN 的核心职责是产生一个动力学更新
- 而不是先升维再降维的 feature transform

### 好处

- 结构简单
- 完全动力学化
- 容易观察 FFN update 的幅值和步长行为

### 风险

- 缺少传统 GLU / MLP 的高表达非线性
- 如果 ODE 更新空间过于线性，表达能力可能不足
- 对超参数 `orders`, `tau`, `scale`, `shift` 比较敏感

---

## 5.3 `ode_swiglu`: SwiGLU 主体 + ODE 残差校正

### 结构

它把标准 SwiGLU 输出当作 base path：

\[
y_{\mathrm{base}} = \mathrm{SwiGLU}(x)
\]

同时用 ODELayer 产生一个动力学修正：

\[
\Delta = \mathrm{ODELayer}(x,c) - x
\]

可选归一化：

\[
\Delta \leftarrow \Delta \cdot \frac{\mathrm{RMS}(x)}{\mathrm{RMS}(\Delta)}
\]

再用标量门控融合：

\[
\gamma = \sigma(\theta)
\]

\[
y = y_{\mathrm{base}} + \gamma \Delta
\]

### 设计含义

这是一个折中方案：

- 保留成熟的 SwiGLU 表达能力
- 再往里加一个动力学 correction term

从结构上说，它不要求 ODE 完全替代 FFN，只要求 ODE 负责：

- 提供额外的时间相关修正方向
- 在保持 baseline 稳定性的前提下增加 flow-aware 成分

### 适合怎样的叙述

如果汇报时想强调稳健创新而非完全重构，这是很好的过渡模型。

---

## 5.4 `mh_ode_swiglu`: multi-head 动力学版 SwiGLU

### 核心变化

前面的 `ode_swiglu` 仍然是整体共享一个动力学结构，而 `mh_ode_swiglu` 进一步把动力学拆到 head 维度。

### 内部原语 `MultiHeadODELinear`

先把输入映射到多头表示：

\[
z = W_{\mathrm{in}}x,\qquad z\in\mathbb{R}^{B\times N\times H\times d_h}
\]

然后对每个 head 使用自己的动力学矩阵 \(A_h\)：

\[
u^{(0)}_h = z_h
\]

\[
u^{(k)}_h = \frac{s_h}{k}A_h u^{(k-1)}_h
\]

\[
\mathrm{MHODE}(x) = \sum_{k=0}^{K} u^{(k)}
\]

最后拼回原维度。

### 完整 FFN 公式

先从 `x` 或 `c` 生成时间步长：

\[
s = \sigma(\mathrm{TimeMLP}(x \ \text{or}\ c)/\tau)\cdot \mathrm{scale} + \mathrm{shift}
\]

然后分别构造 gate 和 value：

\[
g = \mathrm{MHODE}_1(x, s)
\]

\[
v = \mathrm{MHODE}_2(x, s)
\]

再做 GLU：

\[
h = \mathrm{SiLU}(g)\odot v
\]

\[
y = W_o(\mathrm{Dropout}(h))
\]

### 设计直觉

普通 ODE 化还是所有通道共享一个动力学视角，而 multi-head 化后：

- 不同子空间可学习不同演化规律
- 某些 head 偏 coarse transport
- 某些 head 偏局部修正

它本质上在尝试把 attention 中多头分工的思想移植到 FFN 动力学里。

### 额外特点

代码里专门做了参数预算对齐：

- 会搜索 `hidden_dim_eff`
- 尽量让新结构的参数量不超过 baseline SwiGLU

这对做 fair comparison 很重要。

---

## 5.5 `headwise_ode_value_glu`: 只让 value 分支 ODE 化

### 设计思想

这个版本比 `mh_ode_swiglu` 更克制：

- gate 分支保持普通线性投影
- value 分支才使用 multi-head ODE dynamics

### 数学形式

先生成 head-wise 步长：

\[
s = \sigma\left(\frac{W_s x + U_s c}{\tau}\right)\cdot \mathrm{scale} + \mathrm{shift}
\]

其中 \(s\in\mathbb{R}^{B\times N\times H}\)。

然后：

\[
g = \mathrm{SiLU}(W_g x)
\]

\[
v = \mathrm{MHODEValue}(x,s)
\]

\[
y = W_o(\mathrm{Dropout}(g\odot v))
\]

### 直觉

为什么只改 value？

- gate 更适合做选择器
- value 更适合承载真正的动态演化内容

因此这个设计在归因上更清楚：

- gate 决定哪些通道激活
- ODE value 决定激活后的信息如何传播

### 相比 `mh_ode_swiglu` 的差异

- 参数更集中在 value dynamics 上
- 结构更容易解释
- 对哪个分支更该承担动力学角色给出更明确的结构假设

---

## 5.6 `lowrank_state_ode`: 低秩状态依赖动力系统

### 核心思想

前面的 ODE 结构一般使用固定矩阵或按 head 分块的矩阵；这里进一步改成：

- 对角项由输入决定
- 低秩项由输入决定
- 整个动力学算子是 state-dependent 的

也就是把线性动力系统：

\[
\frac{dz}{dt} = Az
\]

升级为：

\[
\frac{dz}{dt} = A(x)z
\]

### 结构

先把输入映射到隐空间：

\[
z_0 = W_{\mathrm{in}}x
\]

从输入中生成：

- 对角系数 `diag`
- 低秩状态 `rank_state`
- 步长 `step`
- gate

核心算子是：

\[
A(x)z = \mathrm{diag}(d(x))\odot z + U\Big((V^\top z)\odot r(x)\Big)
\]

其中：

- \(d(x)\) 是输入相关的对角项
- \(r(x)\) 是输入相关的 rank 控制向量
- \(U,V\) 是可学习低秩基

### 高阶展开

然后像 ODE 一样做有限阶更新：

\[
u^{(0)} = z_0
\]

\[
u^{(k)} = \frac{s}{k}A(x)u^{(k-1)}
\]

\[
z = \sum_{k=0}^{K}u^{(k)}
\]

最后再乘门控并投回：

\[
y = W_o(\sigma(W_g x)\odot z)
\]

### 设计价值

这是一个非常重要的设计点：它不再假设 FFN dynamics 是固定线性系统，而是假设：

- 不同样本
- 不同 token
- 不同时间

都可能对应不同的局部线性化动力学。

### 优点

- 比固定 ODE 灵活得多
- 低秩结构又能控制参数量
- 兼具状态依赖性和可训练稳定性

### 风险

- 状态依赖矩阵可能更难稳定
- 如果 rank 太小，表达受限；太大则预算不友好

---

## 5.7 `tied_flow`: 共享向量场的多步流动

### 核心思想

这个版本不再强调矩阵指数近似，而是更像在 FFN 内部做一个小型离散 flow integration。

它定义一个共享向量场：

\[
f(z) = W_o\big(\mathrm{SiLU}(W_1 z)\odot W_2 z\big)
\]

然后用若干步显式 Euler 式更新：

\[
z_{k+1} = z_k + \alpha_k f(z_k)
\]

初值：

\[
z_0 = x
\]

最终输出：

\[
y = z_T
\]

### 步长设计

步长是可学习向量：

\[
\alpha = \sigma\left(\frac{\theta + Uc}{\tau}\right)\cdot \mathrm{scale} + \mathrm{shift}
\]

这里每一步的步长都不同，但向量场 \(f\) 在所有步骤共享参数，因此叫 tied flow。

### 设计动机

和普通单步 FFN 相比，它等价于让 FFN 做：

- 少量步的内部积分
- 而不是一次性静态映射

这在叙事上很贴近生成模型：

- 输入是当前状态
- FFN 在 block 内部做若干小步推进

### 优点

- 很容易解释
- 和 flow matching 的离散 ODE 视角天然契合
- 参数共享使其不会因多步而线性爆炸

### 风险

- 多步展开增加计算量
- 若步长过大，可能出现过冲

---

# 6. 导航/细化分工类：显式改变 FFN 的角色

## 6.1 `nav_refine`: 导航分支 + 细化分支的双路融合

### 核心思想

这个结构明确假设：

- 一条分支负责 navigation
- 一条分支负责 refinement

而不是让单个 FFN 自己隐式学出两种行为。

### 分支定义

#### Navigation 分支

导航分支使用 `HeadwiseODEResidualBranch`，本质上是：

\[
s_{\mathrm{nav}} = \sigma\left(\frac{W_s x + U_s c}{\tau}\right)\cdot \mathrm{scale} + \mathrm{shift}
\]

\[
g_{\mathrm{nav}} = \mathrm{SiLU}(W_g x)
\]

\[
v_{\mathrm{nav}} = \mathrm{MHODE}(x, s_{\mathrm{nav}})
\]

\[
n = W_o(g_{\mathrm{nav}}\odot v_{\mathrm{nav}})
\]

它偏向方向推进、结构迁移。

#### Refinement 分支

细化分支直接使用较轻量的 SwiGLU：

\[
r = \mathrm{SwiGLU}(x)
\]

它偏向局部修正、纹理修补、精细残差。

### 混合权重

从条件向量 `c` 生成两个门控：

\[
\alpha,\beta = \sigma(\mathrm{MLP}(c))
\]

输出：

\[
y = \alpha n + \beta r
\]

### 设计动机

相比单路 FFN，这个设计强制引入功能分工：

- `nav` 学大方向
- `ref` 学后期精修

这很适合用来解释 flow 任务里的阶段差异。

### 额外意义

因为 `nav` 分支本身还是 head-wise ODE，所以这里其实是：

- 结构分工
- 动力学分工

两层同时存在。

---

## 6.2 `time_split`: 显式的 early/late 路由 FFN

### 核心思想

与 `nav_refine` 类似，但 `time_split` 的叙事更直接：

- navigation path 处理更早的、粗粒度阶段
- refinement path 处理更晚的、细粒度阶段
- 用一个随输入和条件变化的 gate 在两者间切换

### Navigation path

先做 token 平均池化得到全局上下文：

\[
\bar x = \mathrm{MeanPool}(x)
\]

然后：

\[
h_{\mathrm{nav}} = W_{\mathrm{in}}x + W_{\mathrm{ctx}}\bar x
\]

\[
n = W_{\mathrm{out}}(\mathrm{SiLU}(h_{\mathrm{nav}}))
\]

这个设计有两个重要点：

- 它显式注入全局上下文
- 它更像 coarse semantic transport

### Refinement path

细化分支是一个普通的 SwiGLU：

\[
[u,v] = W_{12}x
\]

\[
r = W_3(\mathrm{SiLU}(u)\odot v)
\]

### 路由门控

路由不是只依赖时间，而是依赖 pooled feature 和条件：

\[
g = \sigma(\mathrm{MLP}(\bar x) + Uc + b)
\]

最终：

\[
y = g\cdot n + (1-g)\cdot r
\]

### 设计价值

相比 `nav_refine`，这里更强调：

- 真正的时间路由
- global context 决定 coarse branch 的参与度

因此很适合汇报时讲成显式阶段性行为的代表。

### 关键直觉

普通 FFN 默认认为一套参数能同时服务所有噪声水平；`time_split` 则明确假设：

- 早期应该更依赖全局导航
- 后期应该更依赖局部细化

---

# 7. 更贴近 flow / denoising 语义的设计

## 7.1 `clean_target`: 让 FFN 预测 clean feature

### 核心思想

这是整个代码库里最语义重定义的 FFN 之一。

普通 FFN 的思路是学：

\[
y = F(x,c)
\]

或者在 residual 语境下学：

\[
\Delta = F(x,c)
\]

但 `clean_target` 认为，对 flow matching 来说，更自然的是：

- 当前输入 \(x\) 像 noisy feature
- FFN 先预测一个更干净的 latent target \(\hat x_0\)
- 再把这个 target 转成速度式更新

### 第一步：预测 clean target

先做一个可受条件调制的 SwiGLU 式预测器：

\[
[u,v] = W_{12}x + Uc
\]

\[
\hat x_0 = W_3(\mathrm{SiLU}(u)\odot v)
\]

### 第二步：估计时间分数

内部再从 `x` 或 `c` 推一个时间比例：

\[
t_{\mathrm{frac}} = \sigma(\mathrm{TimeMLP}(x \ \text{or}\ c)/\tau)
\]

### 第三步：转成 velocity-like update

\[
d = \max(1 - t_{\mathrm{frac}}, \mathrm{min\_denom})
\]

\[
u_{\mathrm{upd}} = \frac{\hat x_0 - x}{d}
\]

这一步和 flow / diffusion 里

\[
v \approx \frac{x_0 - x_t}{1-t}
\]

的形式是对齐的。

### 第四步：控制更新幅度

为防止后期 \(1-t\) 太小导致更新爆炸，代码里加入 RMS 限幅：

\[
\mathrm{RMS}(u_{\mathrm{upd}}) \le \lambda \cdot \mathrm{RMS}(x)
\]

具体实现是：

\[
u_{\mathrm{upd}} \leftarrow u_{\mathrm{upd}}\cdot \min\left(1,\frac{\lambda \mathrm{RMS}(x)}{\mathrm{RMS}(u_{\mathrm{upd}})}\right)
\]

### 第五步：再乘一个强度门

\[
\alpha = \sigma(W_\alpha \bar x + U_\alpha c + b_\alpha)
\]

\[
y = \alpha \cdot u_{\mathrm{upd}}
\]

### 为什么这个设计重要

这不是在换一个更 fancy 的激活函数，而是在重新定义 FFN 的任务：

- 普通 FFN：做 feature transform
- `clean_target`：做 block-level denoiser / target predictor

这和整个生成目标的语义更一致。

### 汇报时推荐强调的点

如果想讲为什么简单 ODE 还不够，这里可以说：

前面的 ODE 系列是在改变 update 的动力学形式，而 `clean_target` 是在改变 FFN 要预测的对象本身。

---

## 7.2 `time_moe`: 时间路由的轻量专家混合

### 核心思想

如果单个 FFN 很难同时覆盖：

- 早期粗糙迁移
- 中期平衡建模
- 后期精修残差

那就直接显式构造多个专家。

这里的 `time_moe` 不是传统大规模 sparse MoE，而是一个小型三专家结构。

### 三个专家

#### Expert 1: coarse expert

先做全局池化：

\[
\bar x = \mathrm{MeanPool}(x)
\]

然后：

\[
e_0 = W_o^{(0)}\big(\mathrm{SiLU}(W_x^{(0)}x + W_c^{(0)}\bar x)\big)
\]

这个专家偏向 coarse/global processing。

#### Expert 2: balanced expert

这是普通的 SwiGLU expert：

\[
e_1 = \mathrm{SwiGLU}_{\mathrm{mid}}(x)
\]

用于承担中间区间、较均衡的 feature mixing。

#### Expert 3: refined expert

这个专家先减去全局均值再处理：

\[
e_2 = \mathrm{SwiGLU}_{\mathrm{ref}}(x - \bar x)
\]

它更偏向高频残差、局部修补。

### Router

router 从 pooled feature 和条件向量产生 3 个权重：

\[
\ell = \mathrm{MLP}(\bar x) + Uc + b
\]

\[
w = \mathrm{softmax}(\ell)
\]

其中：

\[
w_0 + w_1 + w_2 = 1
\]

### 输出

\[
y = w_0 e_0 + w_1 e_1 + w_2 e_2
\]

### 设计意义

这个结构把 FFN 从单一函数升级成条件化函数族。

相比 `time_split` 的双路门控，`time_moe` 进一步允许：

- 不止 early/late 两段
- 还能有一个中间平衡区间

### 适合怎样的汇报表达

可以把它讲成：

当我们怀疑单个 FFN 无法同时承担所有时间阶段的最优行为时，最自然的下一步就是让模型显式学习一个时间条件下的专家组合。

---

# 8. 各种 FFN 的核心差别总结

## 8.1 从静态映射到动力系统

最基础的分界线是：

- `mlp`, `swiglu`：静态 feature transform
- `ode` 及其变体：把 FFN 看成时间相关更新器

关键变化是从：

\[
y = F(x)
\]

变成：

\[
y = F(x, s(x,c))
\]

甚至：

\[
y = \sum_k \frac{s^k}{k!}A^k(x)
\]

或：

\[
z_{k+1} = z_k + \alpha_k f(z_k)
\]

---

## 8.2 从单路处理到显式功能分工

第二条重要分界线是：

- `ode`, `ode_swiglu`, `mh_ode_swiglu`, `headwise_ode_value_glu`, `lowrank_state_ode`, `tied_flow`
  仍然主要是单路结构
- `nav_refine`, `time_split`, `time_moe`
  则显式把 FFN 内部拆成功能分支或专家

这对应一个更强的结构假设：

早期和后期的最优子函数并不相同，因此不应完全共享。

---

## 8.3 从学 update 到学 clean target

`clean_target` 最特殊，因为它的目标形式变成：

\[
\hat x_0 = F(x,c), \qquad
y \propto \frac{\hat x_0 - x}{1-t}
\]

也就是说它不是简单在设计 update 公式，而是在改变 FFN 预测对象的语义。

---

# 9. 如果做汇报，推荐的讲述顺序

可以按下面的顺序汇报，逻辑最顺：

## 第一部分：baseline

1. `mlp`
2. `swiglu`

要点：

- 作为标准 Transformer FFN
- 强但不看时间

## 第二部分：第一轮动力学化

3. `ode`
4. `ode_swiglu`

要点：

- 先把 FFN 看成小型 ODE update
- 再与强 baseline 结合

## 第三部分：提高动力学结构的颗粒度

5. `mh_ode_swiglu`
6. `headwise_ode_value_glu`
7. `lowrank_state_ode`
8. `tied_flow`

要点：

- head-wise dynamics
- state-dependent dynamics
- multi-step shared flow

## 第四部分：显式分工与条件路由

9. `nav_refine`
10. `time_split`
11. `time_moe`

要点：

- 不再相信单个 FFN 能覆盖所有 regime
- 通过 branch / expert 直接做结构分工

## 第五部分：语义重定义

12. `clean_target`

要点：

- FFN 不再学普通 residual
- 改成学 clean latent target，再转成 velocity-like update

---

# 10. 一页式结论

如果只用一页总结，可以直接说：

1. 本项目的核心不是换激活函数，而是把 JiT block 里的 FFN 从静态特征变换器，逐步改造成与 flow matching 更一致的动力学模块。
2. 设计路径大致是四步：`MLP/SwiGLU baseline` -> `ODE 化` -> `显式导航/细化分工` -> `clean target / time-routed experts`。
3. 其中最值得强调的三个方向分别是：
   - `mh_ode_swiglu / lowrank_state_ode`：说明 FFN 内部可以有更细粒度、更状态依赖的动力学
   - `time_split / time_moe`：说明 FFN 不应强行用一套子函数覆盖所有时间阶段
   - `clean_target`：说明 FFN 的预测对象本身也可以与 flow 目标对齐，而不只是修改 update 形式

---

# 11. 当前代码中的 12 个 FFN 名称清单

实际 registry 中的名字如下：

- `mlp`
- `swiglu`
- `ode`
- `ode_swiglu`
- `mh_ode_swiglu`
- `headwise_ode_value_glu`
- `lowrank_state_ode`
- `tied_flow`
- `nav_refine`
- `time_split`
- `clean_target`
- `time_moe`

---

# 12. 汇报时的简短推荐话术

如果教授只给你几分钟，你可以这样概括：

> 我们做的不是简单替换一个 FFN 激活，而是在 JiT 的每个 block 里系统性探索 FFN 的角色重定义。最基础的是 MLP 和 SwiGLU；接下来把 FFN ODE 化，使其变成时间相关的小动力系统；再进一步引入 head-wise、low-rank、tied-flow 等更细粒度动力学；最后通过 time-split、time-MoE、clean-target 等设计，让 FFN 显式承担导航、细化、专家路由或 clean target 预测的角色，从而更贴近 flow matching 任务本身。
