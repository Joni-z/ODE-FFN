# ODELayer 完成实验的证据分析

## 比较对象与证据完整性

主问题：原版 ODELayer 的输入门控、Taylor 阶数是否带来生成收益，收益在计算量代理匹配后是否仍存在。主终点为预定 checkpoint 的 SW2@Heun32；方向为越低越好。协议见 [protocol](../protocol.md)，完整数值见 [stats appendix](stats-appendix.md)。

30 条训练轨迹、54 个预定 checkpoint 诊断齐全，三个数组任务和报告任务均 COMPLETED/0:0。54 个 checkpoint、29 个冻结代码/配置文件及实际远端15个数据文件的哈希通过审计。六项 baseline 前5000步权重比较最大差均为0。它们是技术复跑，不增加训练seed数。

存在一项元数据更正：冻结快照注明的是本地数据manifest哈希，实际训练provenance使用远端manifest。三数据集12个NPZ的26个数组逐项相同；分布采样参数相同，压缩归档字节、生成器文件清单以及约2e-16的Ring8均值统计不同。实际训练/评估数据没有混用。原记录保留，更正证据见 [completion audit](../completion-audit.json)、[data content audit](../data-content-audit.json)。

## 主要观察

1. 原版K2在同5000步时SW2低于SwiGLU：Gaussian 0.067437 vs 0.074611；Ring8 0.184820 vs 0.208018；Moons 0.027083 vs 0.030210。对应K2计算量的SwiGLU6572步为0.065757、0.150846、0.062414，方向变为前两组SwiGLU更低、Moons ODE更低。不能用同step比较证明预算优势。
2. K4→K8的SW2为Gaussian 0.070309→0.070221、Ring8 0.175176→0.174829、Moons 0.027682→0.027664；全网络前向线性MAC从154304增至252608（约+63.7%）。三个数据集的相对分数变化幅度均不到0.2%，这是该seed的描述，不是统计等价检验。
3. K8的平均指数作用相对偏差为Gaussian 1.33e-7、Ring8 1.82e-6、Moons 6.16e-8。它与矩阵指数很接近，但Gaussian K1仍更低SW2，Moons K2仍更低SW2；Ring8高阶更低SW2。数据不支持“越像精确隐藏ODE，生成就越好”的跨任务单调关系。
4. time gate的SW2差值（time-input）：Gaussian K1 -0.004288、K2 -0.003182；Ring8 K1 -0.009907、K2 +0.003552；Moons K1 +0.002572、K2 -0.000456。消去对hidden的门控导数项，并不保证训练后的完整层Jacobian更小。例如Moons K2从2.334升至2.509；Ring8 K2从3.014升至3.141。P和门控会重新适应，且输入分布已改变。
5. 原版输入门控K1/2/4/8的六层平均sigma_max(P)从共同初始化1.109增至约1.715–1.862，稳定秩从17.37降至约8.31–9.40。说明最大方向增强、奇异值能量相对更集中。不能称为代数秩坍塌、也不能据此认定训练失稳。K4/K8的P训练曲线几乎重合，与终点结果接近相吻合。
6. 固定评估样本下baseline分数随训练明显波动。Gaussian SwiGLU从5000步0.074611变为5036步0.057765；Ring8 SiLU从5000步0.163125变为5024步0.225790。评估源和投影固定，差异不是每次重抽评估样本导致；但目前未定位到优化器、门控或数据配对的具体原因。预定终点仍保留，不事后挑最好checkpoint。
7. Heun32增加到256没有改变本轮原版K1/2/4/8在各数据集上的主要阶数排序。256步不是理论真值，细小差异仍不作精确等价或收敛证明。

## Claim Candidates

- Claim：本设置中高于K4的阶数收益有限。
  - Source evidence：`contrasts.csv`的K8-minus-K4、机制图、MAC表。
  - Allowed wording：seed0三分布上K4与K8终点相近，而K8计算量代理更大。
  - Forbidden stronger wording：K4普遍最优；K8统计上无效；矩阵指数没有价值。
  - Uncertainty：单seed、局部探针、仅2D任务。
  - Next check：若继续，固定同一checkpoint只改变求值阶数，区分求值截断和训练后P适应；明确这是部署时干预。
  - Decision：keep，限于当前设置。

- Claim：输入依赖门控没有统一生成优势，移除门控导数也不保证完整Jacobian下降。
  - Source evidence：gate-source图、stats-appendix、各checkpoint诊断。
  - Allowed wording：time/input对照的质量变化方向依赖K和数据集，训练后P有补偿变化。
  - Forbidden stronger wording：门控是问题根源；time gate已解决稳定性。
  - Uncertainty：门控输入分布和初始输出同时改变，不是纯导数项干预。
  - Next check：若用户选择门控方向，单独加入固定s=0.6或每层可学习常数s作为可解释对照，预注册比较；不同时改变P。
  - Decision：revise，弱化原先单纯压门控的推测。

- Claim：尚未建立一致的ODE计算预算优势。
  - Source evidence：order-and-budget图、training-and-nfe图。
  - Allowed wording：同step收益在部分MAC匹配对照中消失，且终点敏感。
  - Forbidden stronger wording：ODE已证明优于SwiGLU；SwiGLU已普遍淘汰ODE。
  - Uncertainty：MAC仅代理、终点波动、单seed。
  - Next check：优先分析已有轨迹，未来预先定义固定尾段摘要/验证集选点规则，不能用本轮测试集挑最好点。
  - Decision：weaken。

- Claim：训练中的P呈现更强的主方向和更集中的奇异值能量。
  - Source evidence：p-training-dynamics图、p-trajectory-summary.csv。
  - Allowed wording：该现象在原版四种K的三组seed0训练中均观察到。
  - Forbidden stronger wording：谱集中导致性能差；强制奇异值平均能改善生成。
  - Uncertainty：未施加对应因果干预，稳定秩不是代数秩；六层不是六个独立模型。
  - Next check：受控P谱实验明确区分整体尺度、奇异值分布与非正规性；不一次混合三种约束。
  - Decision：keep为描述性观察，因果解释保留。

## 推断范围

只有seed0，故不报告跨seed标准差、置信区间或显著性检验。三个数据集不是随机抽样的任务总体；不以“几胜几负”估计普适成功率。高阶模型、门控模型的参数量匹配不等于计算量或函数类相同。玩具任务是直接velocity prediction，无attention，不等价于JiT图像实验。
