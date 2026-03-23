# 激进版 FFN 设计方案（for Flow Matching / Rectified Flow / JiT-like Transformer）

下面给出 3 个比普通 SwiGLU / ODE-SwiGLU 更激进的 FFN 方案。  
核心目标不是“小修小补”，而是**强行把 FFN 变成和 flow 目标更匹配的动力学模块**，从而避免“训练几十 k step 后全部并轨”的问题。

---

# 方案一：双路时间分工 FFN（Time-Split Dual-Path FFN）

## 核心思想
把一个 FFN 拆成两条并行路径：

- **Navigation path**：负责小 \(t\) / 早期阶段，做粗粒度、全局性的语义推进
- **Refinement path**：负责大 \(t\) / 后期阶段，做局部细节修补和残差校正

最后用一个随时间变化的门控函数融合两条路。

---

## 数学形式

设输入为 \(x \in \mathbb{R}^{B \times N \times D}\)，时间为 \(t \in [0,1]\)。

### 两条分支
\[
h_{\text{nav}} = F_{\text{nav}}(x)
\]
\[
h_{\text{ref}} = F_{\text{ref}}(x)
\]

### 时间门控
\[
g(t) = \sigma(a t + b)
\]

更激进一点也可以写成：
\[
g(x,t) = \sigma(\text{MLP}([t, \mathrm{Pool}(x)]))
\]

### 输出
\[
y = g(t)\cdot h_{\text{nav}} + (1-g(t))\cdot h_{\text{ref}}
\]

---

## 为什么这个方案可能比 baseline 强
普通 FFN 的问题是：  
**所有时间段都共用同一套映射**，但 flow matching 在不同时间段的任务其实不同。

- 早期：更像“往正确语义方向推”
- 后期：更像“做局部细节修复”

单路 FFN 容易学成一个 compromise。  
双路 FFN 则强制不同参数负责不同阶段，减少 gradient conflict。

---

## 结构设计建议

### 1. Navigation branch
建议偏“低频 / 全局 / 强 bottleneck”

\[
F_{\text{nav}}(x) = W_o \phi(W_2 \phi(W_1 x))
\]

特点：
- hidden dim 可以压小
- 可以额外引入 pooled token summary
- 目标是增强 coarse semantic transport

一个更 aggressive 的版本：
\[
c = \mathrm{MeanPool}(x)
\]
\[
h_{\text{nav}} = W_o \phi(W_x x + W_c c)
\]

这样导航分支不是纯 token-wise，而是显式读 global context。

---

### 2. Refinement branch
建议偏“小步修正 / 高频 / residual correction”

\[
\Delta_{\text{ref}} = W_o(\phi(W_2 \phi(W_1 x)))
\]
\[
h_{\text{ref}} = x + \alpha(t)\Delta_{\text{ref}}
\]

其中
\[
\alpha(t) = \sigma(c t + d)
\]

特点：
- 更像局部修补器
- 后期允许更大比例使用该分支
- 可以在这一路做更强 norm control

---

## 进一步激进化
不要让门控只是依赖 \(t\)，而是依赖输入状态：

\[
g(x,t) = \sigma(W_2 \phi(W_1[\mathrm{MeanPool}(x), t]))
\]

甚至做 token-wise gate：

\[
g_i(x,t) = \sigma(W_2 \phi(W_1[x_i, t]))
\]

这样每个 token 在不同时间都可以选不同路径。

---

## 参数量控制
如果担心参数爆炸，可以这样做：

- 两条路都比 baseline 的 hidden dim 小
- 例如 baseline hidden 为 \(H\)，则两条路各用 \(H/2\) 或 \(0.6H\)
- 或者共享输入投影，后面再分叉

例如：
\[
u = W_{\text{in}}x
\]
\[
h_{\text{nav}} = W_{\text{nav}} \phi(u)
\]
\[
h_{\text{ref}} = W_{\text{ref}} \phi(u)
\]

这样参数不会直接翻倍。

---

## 预期优点
- 不同时间段专门化
- 比单路 FFN 更容易形成阶段性行为
- 解释性强，paper narrative 很顺
- 很适合做可视化：画 \(g(t)\) 随时间变化曲线

---

## 潜在风险
- 如果 gate 学塌了，可能永远偏向某一路
- 两条路如果设计得太像，最后还是会退化
- 需要额外监控 gate 的分布和 usage entropy

---

## 推荐实验
1. 固定 gate：人为设定 early 用 nav，late 用 ref
2. 学习 gate：只依赖 \(t\)
3. 更强版本：依赖 \(x,t\)
4. 看每个 timestep bucket 上的 loss / FID / update norm

---

# 方案二：Clean-Target FFN（让 FFN 预测“干净特征”而不是普通残差）

## 核心思想
普通 FFN 本质上是在学一个 feature transform：

\[
y = x + \Delta(x,t)
\]

但对 flow matching 来说，更自然的想法是：

> 让 block 内部也像一个小 denoiser，直接预测“当前 noisy feature 对应的 clean feature”

然后再把它转成一个 velocity-like update。

---

## 数学形式

设 block 输入为 \(x_t\)。

### 第一步：预测 clean feature
\[
\hat{x}_0 = F_{\theta}(x_t,t)
\]

### 第二步：转成 block 内 velocity
\[
v_{\text{block}} = \frac{\hat{x}_0 - x_t}{1 - t + \epsilon}
\]

### 第三步：做小步更新
\[
y = x_t + \alpha \, v_{\text{block}}
\]

其中 \(\alpha\) 可以是：
- 固定标量
- 可学习标量
- 依赖 \(t\) 的门控

---

## 和普通 ODE residual 的区别
普通 ODE-like FFN 是：
\[
y = x + \Delta t \cdot f(x,t)
\]

而这里是：
\[
y = x + \alpha \cdot \frac{\hat{x}_0 - x}{1-t+\epsilon}
\]

也就是说，FFN 不直接学“下一步该往哪走”，  
而是学“当前 noisy feature 的 clean target 长什么样”。

这和 flow / diffusion 里 x-pred 的想法更一致。

---

## 直觉
对于高噪声输入，直接学 noisy-to-noisy 的复杂映射可能很难；  
但如果网络先形成一个“干净目标”的隐式估计，再从当前状态朝那个目标靠近，会更稳定。

你可以把它理解成：

- 普通 FFN：学一个静态 feature mixer
- Clean-Target FFN：学一个 block-level denoiser

---

## 结构设计建议

### 版本 A：最简单版
\[
\hat{x}_0 = W_o(\phi(W_2\phi(W_1[x_t, e_t])))
\]
其中 \(e_t\) 是 time embedding。

然后：
\[
y = x_t + \alpha \frac{\hat{x}_0 - x_t}{1-t+\epsilon}
\]

---

### 版本 B：加一个 confidence gate
因为不是所有时间都适合强烈往 clean target 靠，可以加：

\[
c(x,t) = \sigma(W_c \phi([x_t, e_t]))
\]

\[
y = x_t + c(x,t)\odot \frac{\hat{x}_0 - x_t}{1-t+\epsilon}
\]

这样不同 token / channel 可以有不同 update 强度。

---

### 版本 C：和 baseline 分开
为了避免太激进导致完全不稳，可以保留 baseline 支路：

\[
y = \text{BaseFFN}(x_t) + \lambda(t)\cdot \frac{\hat{x}_0 - x_t}{1-t+\epsilon}
\]

但如果你想“大胆设计”，更推荐直接让 clean-target 成为主体。

---

## 为什么这个方案有潜力
因为它不是在“给 baseline FFN 加一个更花的激活”，  
而是在**改 FFN 的训练语义**：

- baseline：做 feature transform
- 这个方案：做 target-oriented denoising

它更接近 objective-aware architecture。

---

## 参数预算处理
为了和 baseline 对齐，可以：
- 让 clean-target predictor 的 hidden dim 稍微缩小
- 或者共享一部分投影层
- 或者把输出拆成 low-rank form：
\[
\hat{x}_0 = U(Vx)
\]
减少参数量

---

## 潜在风险
- 分母 \(1-t+\epsilon\) 在 \(t \to 1\) 时会变敏感，需要 clamp
- 如果 \(\hat{x}_0\) 学得不好，block update 可能噪声很大
- 需要配合 RMS norm / residual scaling

---

## 稳定性技巧
建议加以下几个：

### 1. 限制 update norm
\[
u = \frac{\hat{x}_0 - x_t}{1-t+\epsilon}
\]
\[
u \leftarrow u \cdot \min\left(1, \frac{\tau \cdot \mathrm{RMS}(x_t)}{\mathrm{RMS}(u)+\epsilon}\right)
\]

### 2. 给 \(\alpha\) 做可学习初始化
初始化成很小，比如 0.1 左右

### 3. late timestep clamp
\[
1-t+\epsilon \ge \delta
\]
避免极端爆炸

---

## 推荐实验
1. 纯 Clean-Target FFN
2. Clean-Target + norm control
3. Clean-Target + confidence gate
4. 和普通 ODE residual 对比不同 timestep bucket loss

---

# 方案三：混合专家式 FFN（MoE-style Time-Routed FFN，但不是传统大 MoE）

## 核心思想
既然单个 FFN 学不出明显阶段分工，那就更激进：

> 不是两条路，而是直接做 3~4 个“小专家”，每个专家负责不同时间段 / 不同 feature regime。

这不是传统大规模 token-level sparse MoE，  
而是一个**轻量时间路由专家 FFN**。

---

## 数学形式

设有 \(K\) 个专家：
\[
E_1(x), E_2(x), \dots, E_K(x)
\]

门控网络根据 \(t\) 或 \((x,t)\) 输出权重：
\[
\pi(x,t) = \mathrm{softmax}(G(x,t))
\]

输出：
\[
y = \sum_{k=1}^{K}\pi_k(x,t) E_k(x)
\]

---

## 为什么它比双路更激进
双路只是在“早期 vs 后期”上分工。  
多专家则可以逼网络学出更细的行为模式，例如：

- Expert 1：极高噪声，做强 semantic transport
- Expert 2：中噪声，做结构整理
- Expert 3：低噪声，做细节修补
- Expert 4：特殊 token 模式，做异常修正

这相当于把 FFN 从“单一映射”升级为“条件化函数族”。

---

## 推荐的小规模版本
不要直接做 8 个专家，那太重。  
先做 \(K=3\)：

- \(E_1\)：coarse/global expert
- \(E_2\)：balanced expert
- \(E_3\)：refinement/high-frequency expert

---

## 专家结构可以故意做得不一样

### Expert 1：低秩 / 全局型
\[
E_1(x)=W_o\phi(W_2\phi(W_1x + W_c\mathrm{Pool}(x)))
\]

### Expert 2：标准型
\[
E_2(x)=W_o\phi(W_2\phi(W_1x))
\]

### Expert 3：细节残差型
\[
E_3(x)=x+\beta \cdot W_o\phi(W_2\phi(W_1x))
\]

这样专家之间不是“参数不同但结构一样”，  
而是**归纳偏置本身不同**。

---

## 门控怎么做

### 最简单
只看时间：
\[
\pi(t)=\mathrm{softmax}(W_2\phi(W_1 t))
\]

### 更强
看输入和时间：
\[
\pi(x,t)=\mathrm{softmax}(W_2\phi(W_1[\mathrm{Pool}(x), t]))
\]

### 更激进
token-wise expert routing：
\[
\pi_i(x,t)=\mathrm{softmax}(W_2\phi(W_1[x_i,t]))
\]

这样每个 token 走不同专家，但实现会更重。

---

## 怎么防止 collapse
MoE 最大的问题是 gate collapse：所有样本只用一个专家。

可以加一个简单负载均衡正则：

设 batch 内平均专家使用率为
\[
\bar{\pi}_k = \frac{1}{B}\sum_{b=1}^B \pi_k(x_b,t_b)
\]

加入正则：
\[
\mathcal{L}_{\text{balance}} = \sum_{k=1}^K \bar{\pi}_k \log \bar{\pi}_k
\]

或者鼓励接近均匀：
\[
\mathcal{L}_{\text{balance}} = \sum_{k=1}^K \left(\bar{\pi}_k - \frac{1}{K}\right)^2
\]

---

## 为什么这个方案可能真正拉开差距
因为它不再假设“一个 FFN 可以统治所有时间段和所有 feature 状态”。  
它让模型显式学到：

- 什么时候该粗推
- 什么时候该稳修
- 什么时候该用特殊策略

如果你图里现在所有方法都后期并轨，很可能就是因为函数空间还太单一。  
轻量 MoE 直接扩大了函数族的多样性。

---

## 参数预算怎么控
为了不比 baseline 大太多，可以：

### 方法 1：每个专家都很小
如果 baseline hidden dim 是 \(H\)，3 个专家各自只用 \(H/3\) 或更小。

### 方法 2：共享输入投影
\[
u = W_{\text{shared}} x
\]
\[
E_k(x)=W_{o,k}\phi(W_{k}u)
\]

### 方法 3：只激活 top-1 / top-2
如果要进一步控算力，可以 sparse routing：

\[
y = \sum_{k \in \mathrm{TopK}(\pi)} \pi_k E_k(x)
\]

但第一版建议先 dense mix，简单稳定。

---

## 潜在风险
- 实现复杂度最高
- gate 容易 collapse
- 如果专家差异设计不够大，会白费
- 分析难度也更高

---

## 推荐实验
1. 3 experts, dense mix, time-only gate
2. 3 experts, dense mix, input-aware gate
3. 加 balance regularization
4. 可视化不同 timestep 的 expert usage

---

# 三个方案的对比总结

| 方案 | 核心激进点 | 优点 | 风险 | 推荐程度 |
|---|---|---|---|---|
| 双路时间分工 FFN | 强制 early/late 分工 | 结构清楚，最容易实现和讲故事 | gate 可能塌 | 很推荐 |
| Clean-Target FFN | FFN 变成 block-level denoiser | 和 flow objective 最一致 | 稳定性要小心 | 很推荐 |
| 轻量 MoE 时间专家 FFN | 让多个专家负责不同 regime | 最有机会真正跳出 baseline 函数族 | 最复杂 | 推荐做第二阶段冲击实验 |

---

# 我最建议的推进顺序

## 第一优先级：双路时间分工 FFN
原因：
- 结构清晰
- 最容易实现
- 最容易验证“是不是时间分工真的有用”
- 最容易写进论文故事里

## 第二优先级：Clean-Target FFN
原因：
- 最 objective-aware
- 和你整个项目主线非常一致
- 如果有效，会很有新意

## 第三优先级：MoE 时间专家 FFN
原因：
- 最激进
- 最可能真拉开差距
- 但训练和分析成本最高

---

# 我个人建议的最强组合版本

其实这三个方案还能融合：

## Time-Split Clean-Target MoE FFN
例如：
- 先做 2 路：nav / ref
- 每一路内部不是普通 MLP，而是 clean-target predictor
- 或者用 3 个小专家，其中一个专门做 clean-target correction

比如：
\[
\hat{x}_{0,k} = E_k(x_t,t)
\]
\[
v_k = \frac{\hat{x}_{0,k} - x_t}{1-t+\epsilon}
\]
\[
y = x_t + \alpha \sum_{k=1}^K \pi_k(x,t) v_k
\]

这已经不是“FFN 调味料”了，  
而是一个真正的 **flow-aware dynamical FFN block**。

---

# 建议你接下来真正要看的指标
不要只看最终 FID，还要看这些：

## 1. 按 timestep bucket 的训练损失
例如分成：
- \(t \in [0, 0.2]\)
- \(t \in (0.2, 0.5]\)
- \(t \in (0.5, 0.8]\)
- \(t \in (0.8, 1.0]\)

## 2. 每层 FFN update norm 随时间的变化
看新 FFN 是否真的产生阶段性行为。

## 3. gate / expert usage 分布
看是不是：
- early 更偏 nav / coarse expert
- late 更偏 ref / detail expert

## 4. 0~100k step 的 FID AUC
因为你现在的问题不是最终完全不行，而是后面都并轨。  
那就该看“早期谁更快、整体谁更省训练”。

---

# 一句话结论
如果你现在的所有 FFN 变种在几十 k step 后都差不多，  
说明问题不是“激活函数还不够 fancy”，而是**FFN 的角色定义本身没有变**。

真正大胆的方向应该是：

1. **按时间分工**
2. **按 clean target 建模**
3. **按 regime 用多个专家**

这样才有机会从“baseline 的局部扰动”变成“不同的动力学机制”。

---
