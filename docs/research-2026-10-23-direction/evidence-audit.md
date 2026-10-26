# 现有证据审计：下一步应解释什么

审计日期：2026-10-23。本文只重读仓库和重新汇总已经保存的数组，没有训练、重新采样、修改权重或改动 Report 1。较早 GPU 结果和最新 CPU 结果分别解释，不合成排行榜。

## 结论

最有根据的研究对象是 **共享 P 的多项式约束是否有独立于普通门控的作用**。已有实验支持输入门控能提供有用的局部导数，却没有证明其优势来自连续流；没有证据把 P 的谱过大作为主要故障。最新一项误差极大的轨迹案例，实际是模型没有拟合出 oracle 要求的强扩张，不能当作模型 Jacobian 爆炸的证据。

## 1. 已经回答的问题，不值得原样再跑

| 问题 | 证据 | 决策含义 |
|---|---|---|
| 把 P 的最大奇异值压到 1 是否稳定改善质量？ | 原 K1/K2、三数据集：六组内部层 Jacobian 均减小，SW2 只改善三组。Ring8 K2 内部层范数 3.014→2.099，SW2 0.18482→0.18893。完整速度场 t=.4–.5 最大对称特征值均值反而约 2.105→2.127。 | 不继续以“范数越小越好”为假设扫 cap。现有 cap 是整体缩放，并非奇异值均匀化。 |
| 更准确的 exp 是否直接改善已有低阶 checkpoint？ | K1/K2 冻结权重换 exp 的六组全部更差；Ring8 K2 oracle MSE .02871→.16944，SW2 .18482→.38133。Heun256 仍为 .18613→.38371。 | “修正截断误差”不能作为默认动机；从头训练 exp 尚未测试，但不必为证明精度直觉再扫求解器。 |
| 增大 K 是否均匀有效？ | 同步数 Gaussian K1 最低；Ring8 K2/K4/K8 为 .18482/.17518/.17483，预算匹配后 SwiGLU 在 K2/K4 对比更低。 | 高阶收益没有形成质量/算力优势。K8 作用接近 exp，不代表 K4/K8 功能等价。 |
| 学一个常数 s 是否必要？ | 最新 fixed/scalar Ring8 oracle MSE .07462/.07373；8192-sample SW2 .14059/.13965。 | 与常数可吸收到自由 P 的等价性一致，不需要再扩大 scalar sweep。 |
| 输入门控是否有用？ | 最新 Ring8 原版/固定/scalar oracle MSE .03250/.07462/.07373；离各簇中心均超过 .6 的比例 8.35%/19.87%/19.68%。 | 是有价值的信号，但取消门控同时减少约 2.5 万参数；不能单独证明特定 ODE 结构的必要性。 |
| 门控导数是否只是在盲目放大？ | 锚定门控导数对照：Gaussian 公共插值点 Jacobian 误差降低 44.5%；Ring8 降低 58.5%。 | 原版训练后门控通常修正了完整场导数。局部有用不等于重新训练时不可替代。 |
| 门控是否特有地用于模态边界？ | Gaussian 同样获益；Ring8 t=.9 的全部 512 个公共插值点均低熵、局部收缩，误差仍从 3.31231 降到 .44414。 | 应放弃“已证明专门用于多模态分离”的狭窄归因。 |

出处：[最初谱约束](../analysis-2026-10-13/analysis-report.md)、[完整场诊断](../research-2026-10-19/results.md)、[阶数与预算](../research-2026-10-20/stage-report.md)、[冻结换阶与求解器检查](../research-2026-10-20-followup/analysis-report.md)、[最新门控对照](../research-2026-10-21-gates/findings.md)、[锚定导数](../research-2026-10-21-gate-geometry/findings.md)。

## 2. 这次从已保存数组中得到的两个补充判断

### 2.1 最大误差案例：缺少扩张，而非模型放大失控

原始文件：`/Users/mr.z/ode-workspace-prep/gate-geometry-20261021/output/gmm_ring8_2d-ode_original-trajectory-0.9.npz`，索引 79。此次先核对了全部 20 个原版 ODE 面板的 SHA256，与[归档审计](../research-2026-10-21-gate-geometry/audit.json)完全一致。

| 量 | 数值 |
|---|---:|
| 位置 | (-1.148805, -2.729369) |
| 目标路径 log density | -15.1434 |
| oracle Jacobian 特征值 | -1.509434, +54.463977 |
| 模型 Jacobian 谱范数 | 2.649029 |
| oracle Jacobian 谱范数 | 54.463977 |
| 沿 oracle 扩张特征向量的模型 Rayleigh 值 | 2.636548 |
| 沿相同方向的 oracle 值 | 54.463977 |
| 速度 MSE | 1.620469 |
| Jacobian 平方 Frobenius 误差 | 2686.293174 |

实际矩阵为：

```text
J_model  = [[ 2.062824, -1.483871],
            [-1.209912, -0.522908]]
J_oracle = [[46.266861, -19.789589],
            [-19.789589, 6.687682]]
```

这一个点贡献了 t=.9 面板总 Jacobian 误差的 94.7%。该面板误差均值为 22.1656，中位数为 .2011，95% 分位数为 3.9952。不能把均值读成“典型轨迹都很不稳定”。冻结门控时该点误差为 2682.4078，原版为 2686.2932；门控的总体收益也不等于每一点都受益。

**这个案例也不意味着应强行拟合低密度处的 +54。** 需要先区分：生成轨迹为何进入这个位置，偏移从何时出现，还是该区域本来就在 oracle 轨迹的有限样本支持中。局部高导数误差是事后诊断，既可能是偏离路径之后的结果，也可能参与后续误差。它只直接反驳“这个大误差就是网络 Jacobian 爆炸”的解释。对所有隐藏层收紧 P 更不能从这个案例推出来。

下一项若选择轨迹分析，应比较同一初始点的 oracle 轨迹和模型轨迹，定位首次偏离，再做事先指定时间窗的小幅/oracle 替换。当前只保存五个时间的模型状态，不能从这些快照恢复首次偏离时间；此处没有执行新增求解。

### 2.2 门控主要修正对称部分，不能把收益都归于消除旋转

本路径的 oracle Jacobian 对称。记 `sym(J)=(J+J^T)/2`、`skew(J)=(J-J^T)/2`，则有精确正交分解：

```text
||J-J*||_F^2 = ||sym(J)-J*||_F^2 + ||skew(J)||_F^2.
```

使用相同原版模型数组，五个时间等权汇总：

| 数据/面板 | 总误差：锚定→原版 | skew 误差：锚定→原版 | 门控总改善中来自 skew 的比例 | 原版剩余误差中 skew 比例 |
|---|---:|---:|---:|---:|
| Gaussian / 公共插值 | .033639→.018676 | .007160→.002788 | 29.2% | 14.9% |
| Gaussian / 模型轨迹 | .033006→.018450 | .007525→.002976 | 31.3% | 16.1% |
| Ring8 / 公共插值 | 3.575498→1.485390 | .320812→.107102 | 10.2% | 7.2% |
| Ring8 / 模型轨迹 | 7.828940→5.966779 | .265482→.084195 | 9.7% | 1.4% |

这是对现有数组的事后描述，不是新假设的独立验证。Ring8 公共插值点中约 89.8% 的门控改善来自对称部分，剩余误差也有约 92.8% 在对称部分。即使假想把当前 J 的 skew 完全消掉而保持其 sym 不变，也只去除该面板约 7.2% 的误差。实际改成对称 P 不会只改变输出 J 的 skew，所以这不是任何训练约束的效果上界；它说明“强制对称就解决主要问题”缺少现有证据。

## 3. s 的实际变化有多大：不要假定它始终接近常数

来源：[training-trajectories.json](../research-2026-10-21-gates/results/training-trajectories.json)，原版模型 step5000，256 个固定 probe 的实际隐藏输入。这里不同外部时间混合在一个面板中，不能据此声称变化独立于 t。

把每层 s 在该面板的均值记作 s0，有**精确**展开：

```text
I+sP+s²P²/2 = B0 + (s-s0) B1 + R(s)
B0 = I+s0 P+s0² P²/2
B1 = P+s0 P²
R(s) = (s-s0)² P²/2

mean ||R(s)||2 <= sigma_max(P)² * Var(s) / 2.
```

下表最后一列由已保存的 P 谱范数和总体标准差计算，是该 probe 面板上**平均算子余项范数的上界估计**，不是实际 `R(s)h` 幅度、整网误差或生成影响。s 的统计在 float32 中计算，数值有正常舍入误差。

| 数据 | 层 | s 均值 | s 标准差 | s 最小/最大 | 平均余项上界 |
|---|---|---:|---:|---:|---:|
| Gaussian | block0.ode1 | .7563 | .0666 | .6558/.8854 | .00634 |
| Gaussian | block0.ode2 | .8676 | .0678 | .6965/.9939 | .00809 |
| Gaussian | block1.ode1 | .6608 | .0489 | .5994/.8041 | .00279 |
| Gaussian | block1.ode2 | .7472 | .0707 | .6201/.9298 | .00776 |
| Gaussian | block2.ode1 | .5853 | .0921 | .4635/.8127 | .01286 |
| Gaussian | block2.ode2 | .6251 | .0914 | .5376/.8894 | .01204 |
| Ring8 | block0.ode1 | .7218 | .1664 | .3646/.9821 | .03800 |
| Ring8 | block0.ode2 | .7495 | .2550 | .2187/.9999 | .11518 |
| Ring8 | block1.ode1 | .7068 | .1625 | .2968/.9629 | .04102 |
| Ring8 | block1.ode2 | .7783 | .2251 | .2974/.9990 | .13242 |
| Ring8 | block2.ode1 | .6219 | .0849 | .4382/.7612 | .00927 |
| Ring8 | block2.ode2 | .8354 | .1923 | .4243/.9976 | .06183 |

Ring8 部分门控几乎覆盖整个允许区间，因此不能未经测量就把它当作常数；Gaussian 较小的波动则使“主要是静态变换加线性门控响应”成为可检验解释。已有日志没有保存对应的逐点 h、s、P²h 联合数组，无法据此算紧的实际曲率作用或后续层放大。一个受限的 checkpoint 分析即可补齐，无需先训练不同的 K 或不同 P 约束。

## 4. 当前比较中仍然存在的具体混淆

1. **容量与计算量。** 原版 82,056 参数、105,152 前向线性 MAC；fixed 56,706/80,192。SwiGLU 81,792/80,000。SwiGLU 参数接近而计算更便宜，始终应保留为实际基线。
2. **初始化不是只看输出为零。** [VelocityNet](../../synthetic/models.py) 的最终输出投影全部初始化为零，所以所有架构初始速度都为零；但内部特征、第一步输出层梯度和后续优化过程并不相同。固定门控与自由矩阵的下一项对照应匹配内部函数，不能拿“初始输出相同”替代。
3. **普通 FFN 对照的结构尚不完全相同。** 常数门控 ODEFFN 是两个 width64 的无 bias 方阵夹 SiLU；当前 StandardFFN 是 hidden128、有 bias。可以做映射 `B_i = T2(.6 P_i)` 后初始函数严格一致的无 bias、hidden64 控制，回答参数化问题。K2 与自由 B 的全局可表示矩阵集合也不同：`2B-I=(A+I)^2`，例如 `B=diag(0,1,...,1)` 会令左侧行列式为负，不可能是实矩阵平方。因此该对照同时检验更新几何与可表示集合，不是纯优化消融；只有局部参数映射可逆时，才能作局部坐标变化的解释。旧的宽 SiLU 分数不能直接回答这一问题。
4. **只用高阶替换不能隔离“曲率有用”。** 将 K2 改成 K1 同时移除静态项、线性门控响应的一部分和真正的 s 曲率。上面的中心化展开能给出更有针对性的工具；其有效性仍需测量，而不能预先宣称更好。
5. **二维完整场不适合证明低秩优势。** 单个隐藏层门控产生 rank-one 修正；两个 ODELayer 的 FFN 局部差分秩至多二。但主干有三块、六个门控，输出也只有二维。当前数据能检查方向是否对，不足以建立可扩展的低秩容量论证。
6. **模型轨迹不是公共状态。** 各模型 trajectory 面板来自各自生成流；其误差差异同时包含所访问状态的差异。公共插值面板可以比较同状态拟合，轨迹面板只能比较各自访问分布上的行为。
7. **synthetic 与 JiT 有真实断层。** [synthetic/models.py](../../synthetic/models.py) 是教授双层原版、直接预测 v；[src/ffn_blocks.py](../../src/ffn_blocks.py) registry 的 `ode` 是默认 K1、delta RMS 归一化、附加输出门控的 ODEOnlyFFN。正 s 的 K1 delta 经 RMS 归一化后会近似消去 s（epsilon 除外）。JiT 预测干净图像后转换为 v，所有比较还有外部时间条件。历史图像 `ode` 不能直接给当前 block 背书。

## 5. 建议先后顺序

**第一，测已有 K2 checkpoint 的中心化曲率项是否真的在用。** 区分局部隐藏动作、完整场在公共输入上的变化和生成路径变化；逐层和全部层干预应明确区分。若去除曲率后几乎不变，就没有充分理由继续优化 P² 的 ODE 含义。若明显影响，也只说明模型使用了该项，下一步才比较普通门控能否承担相同作用。

**第二，按结果选一项训练对照。** 研究静态参数化时，固定门控 K2 与映射初始化的自由 B2 SiLU(B1h) 最直接，但差异可能来自更新几何或可表示矩阵集合，不能仅据胜负归因于优化；研究动态结构时，保留输入 gate、区分共享 P 与普通标量门控/解耦算子。每次只选一条，不一起堆结构。先声明参数和计算差异，训练曲线、oracle 场误差、mode mass 与 SW2 共同报告。

**第三，有明确错误传播证据后再设计 P 约束。** 对称/反对称分解、谱上限、奇异值均匀化都不能只凭漂亮数学推出收益。当前 tail 案例和内部 cap 结果恰好提醒：目标需要局部扩张，约束可能同时压掉有用自由度。

不建议重跑全部阶数、全部 gate 来源、更密的 NFE 表，也不建议立即回到大图像训练。更换评价样本数已展示 SW2 细微排序会翻转，新增设计不能靠一个终点小差值决定。

## 补充计算的复现

从仓库根目录运行下面的只读脚本即可复现本文新增的分解、尾部案例和 gate 余项上界：

```python
import hashlib, json
from pathlib import Path
import numpy as np

raw = Path('/Users/mr.z/ode-workspace-prep/gate-geometry-20261021/output')
audit = json.loads(Path('docs/research-2026-10-21-gate-geometry/audit.json').read_text())
sq = lambda x: np.sum(x*x, axis=(-2, -1))
sk = lambda x: (x-x.transpose(0, 2, 1))/2
for dataset in ('gaussian_aniso_2d', 'gmm_ring8_2d'):
    for panel in ('interpolation', 'trajectory'):
        rows = []
        for t in (.1, .3, .5, .7, .9):
            p = raw/f'{dataset}-ode_original-{panel}-{t}.npz'
            assert hashlib.sha256(p.read_bytes()).hexdigest() == audit['panel_sha256'][p.name]
            with np.load(p, allow_pickle=False) as a:
                j, j0, target = a['jacobian'], a['anchored_jacobian'], a['oracle_jacobian']
                rows.append([sq(j0-target).mean(), sq(j-target).mean(),
                             sq(sk(j0)).mean(), sq(sk(j)).mean()])
        total0, total, skew0, skew = np.mean(rows, axis=0)
        print(dataset, panel, total0, total, skew0, skew,
              'skew share of gain', (skew0-skew)/(total0-total),
              'skew share of error', skew/total)

with np.load(raw/'gmm_ring8_2d-ode_original-trajectory-0.9.npz', allow_pickle=False) as a:
    error = sq(a['jacobian']-a['oracle_jacobian'])
    i = 79
    q = a['eigenvectors'][i, :, -1]
    print('tail', a['x'][i], a['eigenvalues'][i], error[i],
          np.linalg.norm(a['jacobian'][i], 2), q@a['jacobian'][i]@q,
          np.mean((a['prediction'][i]-a['oracle'][i])**2),
          'median/p95', np.median(error), np.quantile(error, .95))

trajectories = json.loads(Path('docs/research-2026-10-21-gates/results/training-trajectories.json').read_text())
for dataset in ('gaussian_aniso_2d', 'gmm_ring8_2d'):
    row = next(r for r in trajectories[dataset+'/ode_original'] if r['step'] == 5000)
    for layer in row['ode_layers']:
        a, sigma = layer['activation'], layer['effective']['sigma_max']
        print(dataset, layer['layer'], a['s_mean'], a['s_std'], a['s_min'],
              a['s_max'], .5*sigma*sigma*a['s_std']**2)
```
