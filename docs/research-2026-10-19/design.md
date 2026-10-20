# ODELayer 下一阶段设计与推导

本文件记录数学推导和待验证设计。代数恒等式不等于模型有效性的证明；内部隐藏流与外部图像生成 ODE 是两个不同对象。训练当前仍为 synthetic v-prediction；JiT 使用 x-prediction，迁移时必须保留这个差异。

## 1. 现状与 baseline

已有 15 run：三个数据集 × [SiLU FFN, ODE K1, cap K1, ODE K2, cap K2]。全部 seed0，5000步。cap1 确实降低二阶项，但没有一致生成改善；原版没有观测到发散。Ring8 的平均速度误差与生成距离排序不同。

正式加入 SwiGLU：out = W3[SiLU(W1 h + b1) * (W2 h + b2)] + b3。按 JiT 官方 hidden=int(nominal_hidden*2/3) 约定，nominal128，实际85；总参数81792，ODE82056，差0.322%。外部 width64/depth3/FiLM/LayerNorm/residual、数据、AdamW3e-4、batch512、5000step、seed0、Heun32保持原协议。只是移植 FFN 运算，没有声称复现完整 JiT 的初始化、RMSNorm或图像架构。不同架构使用相同 seed 不意味着所有共享形状权重逐项相同，因为构造消耗随机数不同；K1/K2和旧代码回归则逐项验证一致。

参数匹配不等于计算量匹配。单 FFN 每样本近似矩阵乘法 MAC：SiLU=2*d*m=16384；SwiGLU=3*d*85=16320；ODE（两层）=2*(d*g+g+K*d*d)，K1=16512、K2=24704。还未计激活、归一化、谱 cap 的 SVD。因此未来必须给质量/算力曲线，不能只看相同步数。四种GPU混跑的 wall time 不能直接作为架构速度比较。

## 2. 原版到底表示什么

写 T_K(A)=I+A+A²/2!+...+A^K/K!。

s(h)=0.2+0.8*sigmoid(g(h)/10)
F(h)=T_K(s(h)P)h。

每个输入只算一次 s，并在级数里冻结。K→无穷时 F(h)=exp(s(h)P)h，对应固定线性系统 dy/du=P y 从 h 出发演化 s(h) 的时间。它不等于在演化过程中不断重算 s(y) 的系统。s 与生成时间 t 无必然相等关系。

对有限矩阵，指数级数处处收敛，不需要 eigenvalue<1。真正要研究的是有限阶误差、变换的条件数、对输入扰动的敏感度，以及它们是否对学习有用。所有层独立 P，也没有自动形成整个 Transformer 的同一条连续轨迹。

## 3. P 的正确分解：形变与旋转

P = S + Omega
S=(P+P.T)/2
Omega=(P-P.T)/2

对真实线性 ODE：

d||y||²/du = 2*y.T*S*y
mu(P)=lambda_max(S)
||exp(sP)||₂ <= exp(s*mu(P)), s>=0。

Omega 在该瞬时能量式中不产生长度变化；S 决定扩张/收缩。sigma_max(P) 同时惩罚两部分，是保守代理。P 的 eigenvalue 实部不能代替 mu(P)，因为一般 P 非正规。

这不是说旋转对数值误差完全无影响。固定步长显式方法对高频旋转仍可能很差；控制真实连续解与控制离散映射是两个问题。

“奇异值拉平”也不是自然终点。目标分布各向异性时不同方向本来需要不同变化率。更适合的目标是限制不必要的最大增益，同时保留有用的各向异性，而不是令所有奇异值相等。

## 4. 门控项是原版数学里不可省略的一部分

令 A=T_K(sP)，b=P*T_(K-1)(sP)*h，则：

J_F(h) = A + b * grad(s(h)).T

第二项最多 rank1，但不保证小。它的谱范数恰好是 ||b||*||grad(s)||。

因此：
||J_F|| <= ||A|| + ||b||*||grad(s)||。

s 的 sigmoid 范围有界，不代表梯度一定小：
||grad(s)|| <= (0.8/(4*10))*||grad(g)|| = 0.02*||grad(g)||。

g 的线性层没有约束，训练仍可放大梯度。不能用 stop_gradient(s) 宣称解决真实 Jacobian：那只是改变训练梯度，并没改变前向函数的数学敏感性。

对精确指数还有一个可解释恒等式：

det J_F = exp(s*trace(P)) * [1 + grad(s).T*P*h]。

证明：指数对 s 求导为 P exp(sP)，再用矩阵行列式引理。说明即使 exp(sP) 本身可逆，输入相关的演化时间仍能让完整映射局部奇异。括号非零只是局部条件，不是全局可逆性的保证。对有限多项式若 A 可逆，也有 det J=det A*(1+grad(s).T*A^-1*b)。

这组公式已通过 float64 autograd 数值验证（K1/2/3 Jacobian，精确指数 determinant）；文件 math-verification.json。它们是本次推导，不是新颖性声明。

## 5. Taylor 截断与稳定性不兼容的最小反例

P=[[0,-w],[w,0]]，固定 s，记 a=s*w。

真实 exp(sP) 是旋转，所有奇异值为1。
K1 的增益=sqrt(1+a²)。
K2 的增益=sqrt(1+a⁴/4)。

a=1 时分别为1、1.4142、1.1180。这说明“把 P 变成反对称 → 原版 K2 就保持长度”是错误的。测试已验证。

一般截断误差可用 r=s||P|| 粗界：
||exp(sP)-T_K(sP)|| <= exp(r)*r^(K+1)/(K+1)!。

这个界往往很松，真正诊断应该同时测算子误差和实际 hidden input 上的作用误差。后者只说明访问到的输入，不是全局保证。即使 Taylor 与 exp 差异很大，也不能自动称为训练的“误差”：如果网络把多项式当可学习非线性使用，精确 exp 可能更差。必须用同一个 P 家族做 exact-vs-polynomial 消融。

## 6. 为什么不应强迫处处收缩

我们的 independent Gaussian coupling：x_t=(1-t)x0+t*x1，x0标准高斯，x1均值m、方差diag(v)。

C_i(t)=(1-t)²+t²*v_i
v_star(x,t)=m + diag(a_i(t))*(x-t*m)
a_i(t)=[t*v_i-(1-t)]/C_i(t)。

当前 v=(.09,4)，t=.5 时真实场 Jacobian 两个特征值约 -1.6697 与 +1.2。一个方向收缩，一个扩张，而且随 t 变号。禁止所有正增长会排除真实目标场。对隐藏 P 而言，它与真实场不一一对应，因此这只是防止错误外推的例子，不是证明 P 必须取同样数值。

更一般地，令 m_t(z)=E[x1|x_t=z]，0<=t<1，适当可微/可积条件下：
v_star=(m_t(z)-z)/(1-t)
J_m = t/(1-t)² * Cov(x1|z)
J_v = t/(1-t)³ * Cov(x1|z) - I/(1-t)。

这说明当前独立标准高斯插值的 oracle 场 Jacobian 为对称矩阵，但不等于负定。非高斯源、非独立coupling、其他schedule不能直接套用该式；非线性隐藏表示也不要求每层 P 对称。该关系给我们一个 toy oracle 对照，而不是强迫隐藏层没有旋转。

## 7. 与 FM objective 真正相接的地方

用同一初始噪声耦合真实轨迹 X 和学习轨迹 Y，记误差 delta=Y-X。若学习速度场满足 one-sided Lipschitz 上界 L(t)，则：

d||delta||/dt <= L(t)*||delta|| + ||v_hat(X,t)-v_star(X,t)||。

积分得：终点路径误差 <= 各时刻速度误差乘以后续 exp(integral L) 权重的积分；实际数值采样再加 solver 误差。在足够正则、积分有限条件下取 L2 并使用该 coupling 可给 W2 上界。这是标准稳定性框架，不是本项目新定理。经验局部 Jacobian 最大值不能当作全局 L。

关键研究链条应为：

P / s / 内部求解方式
  -> 实际 ODELayer Jacobian
  -> 完整网络的速度 Jacobian 与局部拟合误差
  -> 外部生成轨迹与采样误差
  -> 最终分布质量。

中间两根箭头尚未证明。norm、FiLM、SiLU、外部残差和所有层都会参与链式求导。不能直接把 mu(P) 填进外部场误差定理。

JiT 的 clean prediction D_theta 转成 v=(D_theta-x)/(1-t)，所以 J_v=(J_D-I)/(1-t)。因此 x-pred 的稳定目标并非简单让 J_D 越小越好；时间接近终点的比例与实现 clamp 也必须考虑。toy v-pred 的结论不能直接套到 JiT。

## 8. 优先候选：可以控制增益的隐藏流

建议先研究同一个生成元家族的精确解与截断解，并显式分离增益和旋转，而不是一下增加 K、强制负特征值和拉平奇异值。

一个可扩展候选（尚未实现/训练）：
P=Q*blockdiag(B1,...,Br)*Q.T，Q为正交矩阵，
Bi=[[a_i,-w_i],[w_i,a_i]]。

在每个二维子空间中：
exp(s*Bi)=exp(s*a_i)*rotation(s*w_i)。

因此用两次 Q 投影和逐对缩放/旋转计算 F，无需对每个 token 做 dense matrix_exp。a_i 控制增益，w_i 控制旋转。允许 a_i 正负，给可解释范围，例如用 bounded 参数化和逐步验证的上界；不先规定最佳上界。Q 可从可训练 Householder 乘积实现；需比较不同正交参数化的成本。该家族为正规矩阵，主动舍弃非正规生成元；这是表达能力限制，必须有同族 Taylor 对照，不能把其结果全归于精确求解。

若先用 scalar s(t) 或每层常数步长，固定 t 时对 hidden h 没有 gate rank-one 项；输入门控 s(h,t) 是后续表达能力消融。若保留输入门控，可约束/惩罚测得的 ||b||*||grad(s)||，但涉及二阶训练梯度，先评估开销。也可对 g 的权重做有保证的 Lipschitz 参数化，仍需要 hidden norm 控制。

这个候选的核心不是“指数矩阵很新”：正交指数/Cayley和动力系统网络已有大量工作。潜在贡献只能来自可扩展的生成任务架构、内部稳定性到外部向量场的可验证关系、以及实际计算预算下的收益。当前尚不具备新颖性或性能主张。

## 9. 最小实验顺序与停止条件

A. 本轮：补 SwiGLU；在六模型/三数据集的固定5000步 checkpoint 做分时误差、NFE扫描、局部 J_F 及 J_v、实际输入截断误差。seed0不扩展。

B. 读诊断后只选一个主要变量。优先同一个结构化 P 下 exact vs K2（同参数、近似一致初始化）；再同一 exact 流下常数/时间门控 vs 输入门控。保留 SwiGLU 和原版 ODE K1/K2。

C. 检验计算与表达权衡：同 step 之外匹配总MAC/同设备训练时间；同一个NFE下 ODE block 可能更贵。数据预处理、参数量、loss、学习率、采样source和投影方向全部记录。不要因为一个指标好就挑checkpoint。

D. 若两三种toy目标的机制证据成立，再迁移到小规模 unconditional image JiT：原装 SwiGLU 与新 ODE，共享 attention、RMSNorm、x-pred objective、噪声schedule、EMA、采样器。先验证正确性/质量成本，不直接上昂贵 ImageNet 大任务。

若强基线胜出、精确求解无收益或局部稳定改善与终点无关，应据此否定对应叙事；仍可以转向 tied polynomial 的表达/优化分析。复杂数学应服务于可证伪问题，不能作为装饰。

补充边界：FFN 是速度/图像预测网络的一部分，FM 并不要求每个内部 FFN 可逆。因此 determinant 公式用于识别门控可能引入的局部折叠，不应把强制全局可逆当作未经论证的必要条件。正交 Q 的训练、构造和存储也有成本；Householder 数量及自由度会影响表达能力/参数匹配，二维闭式更新并不自动意味着总训练更快。

## 10. 阅读依据与文献定位

优先阅读 [Stéphanovitch 2026：Lipschitz regularity in Flow Matching and Diffusion Models](https://arxiv.org/html/2604.06065v1) 的 section3.2.1：它把学习误差的传播与数值误差分开，在明确正则性条件下给出界。这为研究外部速度场提供框架，不能证明我们的内部 ODE block 有效。

baseline 以 [JiT 官方实现](https://github.com/LTH14/JiT/blob/main/model_jit.py) 为准。动力系统网络早有工作，例如 [Meunier 等，ICML2022](https://proceedings.mlr.press/v162/meunier22a.html)，所以“稳定 ODE 层”本身不足以构成新贡献。

本轮也筛查了 2026 的 strain/vorticity 和 Lagrangian FM 预印本；其中前者的离散 Euler 范数界存在纯旋转反例，已在 notes.md 记录。不能因为论文新、术语贴题就把其结论当作设计保证。尚未做穷尽性新颖性审查。

## 11. 候选结构能证明到哪一层

对于正交 Q 和二维 Bi，上述精确指数在冻结 s>=0 时：

sigma_i(F_linear)=exp(s*a_i)（每对两次）
||F_linear||=max_i exp(s*a_i)
cond₂(F_linear)=exp(s*(max_i a_i-min_i a_i))。

所以可以分别限制最大增益与增益跨度，同时保留不相同的 a_i；无需通过压小 w_i 来控制真实流的增益。若 s=s(t)，这些就是固定 t 时隐藏映射的精确 Jacobian 奇异值；若 s=s(h,t)，必须补上前述 rank-one 项。

原版完整 ODEFFN=F2(SiLU(F1(h)))。若两层在所考虑区域的 Jacobian 范数上界是 M1、M2，那么该 FFN 的界是 M2*L_SiLU*M1。SiLU 的全局导数绝对值最大约1.1，不能当成1-Lipschitz。再加入外部 residual h+c*FFN(Norm(h)调制) 时，还要加 identity，并乘上 norm 与 time modulation 的 Jacobian 上界。多层乘积可能很松；能证明单层精确增益，也并不保证整个网络收缩。

因此更可行的理论贡献分两级：先给出局部层的精确公式/条件数和可验证上界；再在明确 bounded-normalization/modulation、区域与采样假设下推完整网络的稳定性命题。若完整界过松，没有解释实验现象，就不把它包装成实际生成保证。

## 12. 还需要排除的替代解释

- 若新层也同等改善普通监督回归，它可能是通用表达/优化收益，不能称为 FM 专属适配。若要声称 objective-specific，需补匹配的非 FM 回归任务或至少与采样误差/时间结构相关的机制证据。
- 可构造已知线性生成元/旋转的单层实验核对数学，但这种 teacher 明确偏向指数层，只能做 sanity check，不能替代现有 Gaussian/Ring8/Moons 的生成对照。
- 更高 K 与更低 solver NFE 不是同一算力维度。要同时报告内部矩阵应用次数与外部网络调用次数。
- 只知道 oracle field 的 Jacobian 对称，不代表最优 latent P 对称：输入输出投影、非线性、残差组合会改变几何。不能将数据空间定理不加条件地投射到每个隐藏矩阵。

另一个需要诊断后排查的共用因素：toy TimeEmbedding 使用16个频率，覆盖1到1000。它允许很快的外部时间变化，不能只用空间 Jacobian 推断采样难度；material derivative 含 d(v)/dt + J_v*v。Heun128/256 都不是理论真值，若二者差异仍大，需要更细步数或可靠自适应参考，而不能把Heun256当完全收敛。若低NFE问题在所有模型中普遍存在，应单独控制时间embedding频率，避免把共用backbone的数值困难误归因于 P。这个控制会改变协议，必须另立实验，不能混进本轮 baseline。

## 13. 完整诊断后的优先级更新

本轮18项诊断已经完成，见 results.md。内部cap减少层Jacobian但未带来一致收益；32→256步没有消除主要生成差距；K1更偏离exp但在Gaussian更好。因而下一轮第一优先是保留P/K、比较input-dependent gate与time-only gate的同参数消融。结构化P的exact-vs-Taylor作为第二阶段，不把更精确求解预设为改进。另用Ring8解析oracle验证早期误差/模态分配假设。
