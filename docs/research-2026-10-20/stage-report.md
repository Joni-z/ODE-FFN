# ODELayer 项目阶段性报告

完整结果解读、P训练轨迹和下一步选择见 [阶段性分析报告](2026-10-20--odelayer--r00--ablation-report.md)。本文件保留自动生成的完整指标表；指数偏差的精确小数见分析报告与统计附录。

状态：实验与诊断齐全；以下为描述性分析，方向由用户决定。训练 30/30；预定checkpoint诊断 54/54。

## 研究问题与原始设计
项目考察：把普通FFN替换为教授提出的ODELayer组合，是否更适合flow matching学习的速度场。原版单层 F(h)=T_K(s(h)P)h，s=0.2+0.8*sigmoid(g(h)/10)，T_K(A)=I+A+...+A^K/K!；完整FFN为两段ODELayer，中间SiLU。P为自由、无bias的方阵。隐藏演化时间s与外部生成时间t不同，P不等于图像速度场的Jacobian。
本阶段使用2D向量FM，直接预测velocity；JiT图像主干以SwiGLU为baseline、直接预测clean image后换算velocity。toy模型没有attention，当前证据不支持直接声称图像收益。

## 数学依据及其边界
冻结s时，指数exp(sP)是dy/du=Py的精确演化。指数级数对任意有限P收敛；本项目研究的是有限阶结构与学习，而不是通过限制eigenvalue保证级数存在。
令A=T_K(sP)，b=P*T_(K-1)(sP)*h，则完整导数J_F=A+b*grad(s)^T。限制P没有自动控制门控项。P=S+R中S对称、R反对称；真实连续系统的长度增长由S决定，但低阶Taylor不会自动继承性质。例如纯二维旋转、s*w=1：精确解增益1，K1为1.4142，K2为1.1180。数值核验见上一阶段math-verification.json。
这些是标准数学恒等式及本项目的核验，不是新定理。更接近exp也未必更好：网络可能利用有限多项式的表达能力。实际FM终点误差还依赖速度拟合误差的时间位置、完整网络的误差传播和外部solver误差。当前没有证明内部P约束能直接控制完整生成场。

## 既有18run证据（seed0，5000步）

|模型|Gaussian SW2|Ring8 SW2|Moons SW2|
|---|---:|---:|---:|
|SiLU|0.086822|0.163125|0.044775|
|SwiGLU|0.074611|0.208018|0.030210|
|ODE K1|0.061061|0.200826|0.031507|
|cap K1|0.063206|0.196959|0.029899|
|ODE K2|0.067437|0.184820|0.027083|
|cap K2|0.065568|0.188929|0.029728|

以上SW2来自固定Heun32。原版K2本次在三个数据集优于SwiGLU，但Ring8上SiLU更好；K2比SwiGLU更贵，且只有一个训练seed。谱cap减小内部层Jacobian但无一致生成收益。增大Heun到256未消除主要排序；不能把差距主要归因于32步不足。
Ring8已有分时诊断：SwiGLU在t=0–0.1的oracle MSE约0.0250，SiLU约0.0141；t=0.7–0.8时分别0.0256和0.0690。早期偏差与模态分配的联系仅为假设，尚未通过轨迹干预或mode mass证明。Gaussian上原版K1比K2更偏离指数，但生成更好，不支持“越像精确ODE越好”的简单叙事。完整出处为上一阶段results.md及diagnostics/*.json。

## 新实验协议
第一波：gate来源input/time × K1/K2，加SiLU和SwiGLU同步baseline，三个数据集共18run。第二波：原版input gate的K4/K8及两种baseline长轨迹，共12run。没有叠加P约束、新loss、归一化、optimizer或数据改变。所有训练seed0。配置和预算节点在训练前冻结；见protocol.md与configs/synthetic/stage_20261020/manifest.json。
每个数据集的两波实验在同一GPU顺序运行；不同数据集可用不同GPU。输入与时间gate同参数、同初始权重，但gate输入分布、初始s及梯度路径不同，不能把所有性能差异归因于单独移除了rank1项。K8仍是截断多项式，不称精确指数。
计算预算使用整个网络的前向线性MAC代理，含time embedding、FiLM等，不含非线性、反向、优化器与评估成本，因此不声称严格相同GPU时长。额外checkpoint已测试不改变训练权重/RNG轨迹。

## 新实验：同5000步对照

|模型|Gaussian SW2|Ring8 SW2|Moons SW2|
|---|---:|---:|---:|
|SiLU|0.086822|0.163125|0.044775|
|SwiGLU|0.074611|0.208018|0.030210|
|ODE K1 input|0.061061|0.200826|0.031507|
|ODE K1 time|0.056772|0.190919|0.034079|
|ODE K2 input|0.067437|0.184820|0.027083|
|ODE K2 time|0.064254|0.188372|0.026628|
|ODE K4 input|0.070309|0.175176|0.027682|
|ODE K8 input|0.070221|0.174829|0.027664|

## 配对门控差值

差值=time gate减input gate，负数代表time gate的SW2更低；没有跨seed置信区间。

|数据|K|SW2差值|input gate项范数|time gate项范数|
|---|---:|---:|---:|---:|
|Gaussian|1|-0.004288|0.439030|0.000000|
|Gaussian|2|-0.003182|0.545801|0.000000|
|Ring8|1|-0.009907|0.815905|0.000000|
|Ring8|2|0.003552|1.121218|0.000000|
|Moons|1|0.002572|0.496511|0.000000|
|Moons|2|-0.000456|0.534999|0.000000|

## 计算预算对照

|数据|K|ODE步数|ODE SW2|SiLU步数|SiLU SW2|SwiGLU步数|SwiGLU SW2|
|---|---:|---:|---:|---:|---:|---:|---:|
|Gaussian|1|5000|0.061061|5024|0.105231|5036|0.057765|
|Gaussian|2|5000|0.067437|6557|0.081459|6572|0.065757|
|Gaussian|4|5000|0.070309|9621|0.082986|9644|0.056350|
|Gaussian|8|5000|0.070221|15751|0.097199|15788|0.054928|
|Ring8|1|5000|0.200826|5024|0.225790|5036|0.208023|
|Ring8|2|5000|0.184820|6557|0.185766|6572|0.150846|
|Ring8|4|5000|0.175176|9621|0.291947|9644|0.149529|
|Ring8|8|5000|0.174829|15751|0.187657|15788|0.198176|
|Moons|1|5000|0.031507|5024|0.041995|5036|0.024115|
|Moons|2|5000|0.027083|6557|0.038795|6572|0.062414|
|Moons|4|5000|0.027682|9621|0.042684|9644|0.035437|
|Moons|8|5000|0.027664|15751|0.018287|15788|0.046784|

## 新实验机制证据

|数据|模型|指数作用相对偏差|完整层Jacobian范数|Heun128对256端点RMS|
|---|---|---:|---:|---:|
|Gaussian|ODE K1 input|0.149183|1.994067|0.000568|
|Gaussian|ODE K1 time|0.135388|1.878796|0.000613|
|Gaussian|ODE K2 input|0.031594|2.342409|0.000531|
|Gaussian|ODE K2 time|0.029553|2.250759|0.000547|
|Gaussian|ODE K4 input|0.000812|2.464667|0.000499|
|Gaussian|ODE K8 input|0.000000|2.461983|0.000498|
|Ring8|ODE K1 input|0.200669|2.371580|0.002120|
|Ring8|ODE K1 time|0.212868|2.229024|0.001614|
|Ring8|ODE K2 input|0.053102|3.014349|0.000990|
|Ring8|ODE K2 time|0.064517|3.140525|0.001159|
|Ring8|ODE K4 input|0.002553|3.421417|0.001105|
|Ring8|ODE K8 input|0.000002|3.428861|0.001086|
|Moons|ODE K1 input|0.123621|2.015508|0.000812|
|Moons|ODE K1 time|0.140833|2.072251|0.000913|
|Moons|ODE K2 input|0.023854|2.334310|0.000988|
|Moons|ODE K2 time|0.029340|2.509107|0.001453|
|Moons|ODE K4 input|0.000508|2.440810|0.001745|
|Moons|ODE K8 input|0.000000|2.445270|0.002022|

这些Jacobian是固定probe的局部描述，非全局Lipschitz界；六层和16个输入不算独立训练重复。模型训练后P和gate会共适应，阶数与指数偏差之间的关系不等于性能因果机制。完整时间段误差、P轨迹与NFE扫描保存在stage-summary.json及各checkpoint诊断文件。

## 技术复跑与数据完整性
- Gaussian baseline前5000步权重最大差：silu=0, swiglu=0。这些是技术复跑，不增加独立seed数。
- Ring8 baseline前5000步权重最大差：silu=0, swiglu=0。这些是技术复跑，不增加独立seed数。
- Moons baseline前5000步权重最大差：silu=0, swiglu=0。这些是技术复跑，不增加独立seed数。

## 图

![门控比较](figures/gate-source.png)
固定5000步的门控对照与实际门控修正项，baseline为水平参考线。
![阶数与计算预算](figures/order-and-budget.png)
ODE固定5000步，对照baseline使用预定MAC匹配步数；没有挑选最佳checkpoint。
![内部机制](figures/operator-mechanisms.png)
指数作用偏差和局部层Jacobian分开呈现；不暗示二者对生成质量有单调因果关系。

## 决策依据与尚未解决的问题
本报告供用户选择继续方向，不自动决定保留哪一结构。判断时应同时读质量、拟合、内部机制和预算：time gate即使降低J也可能损失表达力；高K即使更接近exp也可能不值额外计算；ODE若只在同step优于baseline、在匹配预算后优势消失，应明确降低对架构优势的判断。
只有一个seed，当前是机制探索，不能作统计显著性或普适性结论。既有测试点曾多次用于阶段性观察与假设形成，不是全新的独立确认集。不同分布的SW2不直接平均。Moons无解析oracle，pair MSE不冒充真实场误差。Heun256仅为数值参考；若128→256差异仍大，不宣称收敛。尚未证明隐藏P性质传递到完整速度网络，也尚无图像迁移证据。
待用户决策：①输入依赖是否值得保留；②高阶是否具有预算优势；③是否值得另做P约束/结构化生成元；④何时迁移回JiT。任何新方向应单独立对照，不把多个trick合并后再解释。

## 证据与复现入口
- protocol.md：提前确定的比较、预算、固定条件与决策规则。
- stage-summary.json：所有训练曲线、进度与诊断；diagnostics/：逐checkpoint结果及哈希。
- job-*.json / prefix-check-*.json：执行顺序、GPU、jobID和baseline技术复跑核验。
- source-manifest.json / submission.json（如已生成）：发布的代码哈希和真实Slurm作业号。
- 数学背景：[Lipschitz regularity in Flow Matching and Diffusion Models (2026预印本)](https://arxiv.org/html/2604.06065v1)；baseline定义：[JiT官方代码](https://github.com/LTH14/JiT/blob/main/model_jit.py)。这些文献不证明本项目方法有效。

报告生成完整性标记：complete=True。未齐全时所有新结论保持待定。
