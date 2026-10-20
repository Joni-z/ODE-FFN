# ODE-FFN：当前工作入口

开发仓库：[Joni-z/ODE-FFN](https://github.com/Joni-z/ODE-FFN)。`main` 为当前 torch 版本，原 main 保存在 `archive`；新克隆的数据准备与多机开发方式见 [开发说明](docs/development.md)。

主工作目录：`/scratch/zz5070/ODE_FFN/jit-codebase`（torch）。

当前阶段：先用 **unconditional synthetic flow matching** 检查原版 ODELayer 的机制与 P 的谱，再决定如何回到图像实验。已有 ImageNet 实验是探索性结果；旧 checkpoint 不能还原完整训练轨迹。

最新进展（2026-10-20）：新增门控、阶数和预算两波实验，30/30 条训练、54/54 项诊断完成，全部seed0。[阶段报告与结论](docs/research-2026-10-20/2026-10-20--odelayer--r00--ablation-report.md)包含完整对照、P训练轨迹与局限。K4/K8表现接近，门控与计算预算收益不一致，下一研究方向由用户决定。原18条结果保留，见[SwiGLU 对照与诊断](docs/research-2026-10-19/results.md)。

后续synthetic stage使用`scripts/submit_stage_monitored.py`提交，自动附带监控；必须冻结实际torch数据manifest并设置新实验路径。监控每60秒核对作业、checkpoint、诊断和报告，详见[监控说明](docs/research-2026-10-20/monitoring.md)。它记录状态和异常，不自动重提训练或发送对外消息。

## 目录

| 目录 | 用途 |
|---|---|
| `synthetic/` | 新的低维/向量分布、FM 配对、解析速度场、数据生成与验证 |
| `configs/synthetic/` | 版本化数据参数；不是旧图像训练器的配置 |
| `data/synthetic/v1/` | 已生成的数据入口，实际位于项目外的 `../datasets/synthetic/v1/` |
| `../runs/synthetic/` | synthetic 训练输出；每配置独立目录与分步 checkpoint |
| `docs/synthetic-literature.md` | 论文与官方示例的数据集核对、首轮选择理由 |
| `docs/synthetic.md` | 数据说明、加载示例、理论核验和实验记录约定 |
| `docs/workspace.md` | torch 同名仓库说明、归档与回退办法 |
| `docs/legacy-overview.md` | 整理前 README 原文；历史设计记录，部分陈述尚未审计 |
| `docs/checkpoints.csv` | 旧 checkpoint 的完整位置与文件大小索引 |
| `docs/designs/` | 原 `Designs/` 设计文档 |
| `archive/imagenet/` | 旧配置与日志；原入口保留为相对软链接 |
| `src/`, `configs/ffn/`, `configs/Large/` | 原有图像训练实现与配置 |
| `results/`, `wandb/`, `fid_stats/` | 原有 checkpoint、追踪记录和 FID 参考；保留原位 |
| `analysis/` | 本次旧 P 的谱分析脚本与完整数据（大文件不进入 Git） |

## 使用已准备的数据

```bash
cd /scratch/zz5070/ODE_FFN/jit-codebase
PYTHON=/scratch/zz5070/ODE_FFN/envs/synthetic/bin/python
OMP_NUM_THREADS=1 "$PYTHON" -m synthetic.validate data/synthetic/v1 --preview
"$PYTHON" -m unittest discover -s tests -p test_synthetic_data.py -v
```

新版本数据生成（已有目录会被拒绝覆盖）：

```bash
"$PYTHON" -m synthetic.prepare --config configs/synthetic/data_v1.json --output ../datasets/synthetic/v2
```

`synthetic/train.py` 已实现 SiLU FFN、SwiGLU、教授原版 ODEFFN 与谱范数 cap 对照。向量模型通过该入口训练；`src/main_jit.py` 是 ImageNet 图像训练入口。

Unconditional 表示模型仅接收 `x_t, t`，不接收类别/混合分量标签；使用标准高斯源、独立配对和线性路径，时间方向为噪声 `t=0` 到数据 `t=1`。

## 旧训练注意事项

- `run.sh` 当前无条件追加 `--debug`；不应直接用来启动正式长训练。它使用位置参数选择配置，例如 `bash run.sh ffn/ode`。
- `ode` 注册到 `ODEOnlyFFN`，不是教授最初的两段 `ODEFFN`；一些变体含更新 RMS 归一化，解释 P 时必须核对配置和实现。
- 旧 `last.pt` 反复覆盖；本次只发现最终保存状态，不能据此声称 P 在某一 epoch 开始变化。
- 2026-04-18 有输入归一化修复；跨时期比较需确认实际训练代码。

旧图像模型、训练入口与 checkpoint 保留。Synthetic 训练使用独立脚本；历史 15-run 分析见 `docs/analysis-2026-10-13/`，SwiGLU 补充实验与机制诊断见 `docs/research-2026-10-19/`。
