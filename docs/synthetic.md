# Synthetic v1：无类别条件的 flow matching

## 数据范围

| 名称 | 目标分布 | 目的 |
|---|---|---|
| gaussian_iso_2d | 均值 (1,-1)，单位协方差 | 最简单的可解析起点 |
| gaussian_aniso_2d | 均值 (1,-1)，对角方差 (0.09,4) | 各方向不同尺度 |
| gmm_ring8_2d | 半径 3 上均匀的 8 个 Gaussian，分量方差 0.04 | 多模态、空间变化的速度场 |
| gmm_aniso4_2d | 四角 Gaussian，分量方差交替为 (0.04,0.64)/(0.64,0.04) | 多模态与方向性同时存在 |
| two_moons_2d | 居中的双月牙，加标准差 0.08 的 Gaussian 噪声 | 弯曲几何结构 |
| checkerboard_2d | [-2,2) 内 8 个单位方格的均匀混合 | 多区域与边界，后续压力测试 |
| gaussian_aniso_16d | 零均值，对角方差从 0.04 到 4 几何递增 | 检查更高维的谱行为 |

每个分布：100,000 个训练点、10,000 个验证点、10,000 个测试点、2,048 个独立的固定探测配对。基础 seed=20260923，各数据集/划分使用独立随机流。所有保存的数据坐标均为 float32，无类别标签，不自动标准化：逐维白化会移除我们要研究的各向异性。

`train/validation/test.npz` 只含 `x1`。`probe.npz` 含固定 `x0,x1,xt,t,pair_velocity`，Gaussian/GMM 额外含 `oracle_velocity`，仅用于机制诊断与评估，不用于训练。`distribution.json` 记录分布参数和理论矩；根目录 `manifest.json` 记录配置、NumPy 版本和 SHA256。复现检查比较数组内容；ZIP 文件封装元数据不保证跨时间逐字节一致。

第一轮优先使用 `gaussian_aniso_2d`、`gmm_ring8_2d`、`two_moons_2d`；其余已准备好但不要求立即训练。Two Moons / Checkerboard 尚未实现解析速度场，manifest 中明确标记 `has_velocity_oracle=false`，不能用 pair label 冒充 oracle。选型依据与论文中的实际用法见 [文献核对](synthetic-literature.md)。

## 训练接口

```python
import numpy as np
import torch
from synthetic.data import fm_batch

with np.load('data/synthetic/v1/gmm_ring8_2d/train.npz') as f:
    targets = f['x1']
rng = np.random.default_rng(0)  # 训练随机种子独立于数据种子
idx = rng.integers(len(targets), size=256)
xt, t, target_velocity = fm_batch(targets[idx], rng)
xt, t, target_velocity = [torch.from_numpy(a).to(device) for a in (xt,t,target_velocity)]
prediction = model(xt, t)  # 无 class label；模型见 synthetic/models.py
loss = (prediction - target_velocity).square().mean()
```

每一步重新采样 x0 和 t，不固定训练噪声-数据配对。当前训练为单 GPU，checkpoint 保存 RNG 状态以便准确续训；将来使用 DDP 时需分别管理各 rank 的随机流。在线无限采样也可直接调用 `suite()[name].sample(rng, batch_size)`；比较模型时统一静态/在线协议。

路径与标签：

```
x0 ~ N(0,I), x1 ~ target，二者独立
x_t = (1-t)*x0 + t*x1
target_velocity = x1-x0
```

采样时从标准高斯开始，沿模型速度场从 t=0 积分至 1。不使用 class embedding、label dropout 或 CFG。

## 解析速度场为什么有用

对分量 k，mu_k 是均值，var_k 是逐维方差：

```
C_k(t) = (1-t)^2 + t^2*var_k
b_k(t) = t*var_k - (1-t)
v_k(x,t) = mu_k + b_k(t)/C_k(t) * (x-t*mu_k)
r_k(x,t) = 该分量在 x_t=x 时的后验概率
v_star(x,t) = sum_k r_k(x,t)*v_k(x,t)
```

所有除法和乘法按维度进行。`GaussianMixture.velocity` 返回这个 v_star。它是独立配对条件下的最优边际速度场，**不是**每个训练配对的 x1-x0；pair loss 有不可约条件方差，不能把非零 pair loss 全部归因于模型。即便源和目标都是标准高斯，独立线性插值的中间分布也会先收缩再扩张，并非全程速度为零。

代码测试涵盖：t=0/1 端点、概率连续性方程的数值差分核验、采样矩、配对恒等式、重复生成数组一致性和覆盖保护。

## 后续实验约定

先在同一小型 time-conditioned 向量网络中比较普通 FFN、教授原版 ODEFFN、受约束 ODEFFN。内部 ODE 步长 s(x) 与外部 FM 时间 t 是不同概念；应明确区分，不能把后来 RMS-normalized `ode` 变体默认为原版。合成向量实验不是 DiT 图像实验，不能直接据此声称图像生成质量改善。

建议记录：

- step 0 和固定间隔保存完整 checkpoint（model、EMA 如有、optimizer、RNG、配置、Git revision），使用唯一文件名；last.pt 仅作为便捷入口。
- 保存各层 P、步长网络参数和门控；记录最大奇异值、奇异值分布、有效秩、特征值实部/模。
- 在固定 probe 上记录 s(x)、Taylor 各阶项与输入范数比、最终更新量；只保存 P 无法重建真实层行为。
- 对齐 train pair loss、validation oracle velocity MSE、分布指标（例如固定投影的 sliced Wasserstein）、采样步数/NFE 和耗时。不对二维向量使用 Inception FID。
- 测试集只用于最终比较；固定 probe 属于诊断集，不充当独立最终测试集。当前遵照用户要求只使用训练 seed0，不给跨seed统计结论；报告参数量与计算量。

此前已有18个完整训练run：上述三个数据集 × SiLU FFN / SwiGLU / ODE K1 / cap K1 / ODE K2 / cap K2，均seed0、5000步。2026-10-20另完成30条训练及54项诊断，覆盖input/time门控、K1/2/4/8及SiLU/SwiGLU计算预算对照，见[阶段报告](research-2026-10-20/2026-10-20--odelayer--r00--ablation-report.md)。这些轨迹含技术复跑，不能当48个独立seed。

后续冻结stage后统一用`scripts/submit_stage_monitored.py`提交并附带60秒监控，见[监控说明](research-2026-10-20/monitoring.md)。数据清单必须使用实际执行机器的版本；本地重建NPZ即使数组完全相同，归档及manifest哈希也可能不同。

Checkerboard 保留硬边界，仅作后续几何压力测试；边界附近的逼近与积分困难不能单独归因于 P 不稳定。数据采样参数是本项目明确选择的设置，不宣称精确复现论文；Meta 示例还额外做了 1/0.45 缩放，本版本没有。
