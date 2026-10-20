# torch 工作区索引与整理记录（2026-09-23）

## 仓库边界

- **当前 ODE-FFN 主仓库**：`/scratch/zz5070/ODE_FFN/jit-codebase`，整理前 HEAD `3e8b067`，工作区干净，约 74 GB，其中 results 约 72 GB。
- `/scratch/zz5070/jit-codebase`：另一个 Git 仓库，HEAD `2f42e3f`，约 277 GB，有 5 个已修改的 Slurm 文件。没有移动、覆盖或合并此仓库。
- `/scratch/zz5070/jit-codebase-tmp`：检查时为空目录，保留原位。
- 其他研究项目、共享数据、Conda、缓存、SSH 配置不属于本次整理范围，保留原位。

`/scratch/zz5070/ODE_FFN/` 下分为：

```
jit-codebase/               # 代码、文档、兼容旧入口
datasets/synthetic/v1/     # 版本化数据，代码通过 data/synthetic/v1 访问
runs/synthetic/            # 后续新实验输出，代码通过 runs/synthetic 访问
workspace-inventory.json  # 同名仓库的只读盘点结果
README.md                  # 工作区入口
```

## 可逆归档

| 原入口（现在是相对软链接） | 实际归档目录 |
|---|---|
| configs_old | archive/imagenet/configs |
| logs | archive/imagenet/logs |
| logs_old | archive/imagenet/logs_old |
| Designs | docs/designs |

这些移动在同一文件系统内进行；没有删除旧数据或复制大 checkpoint。`src/` 与现有图像 configs、results、wandb、fid_stats 保持原位。旧 README 原文保存到 `docs/legacy-overview.md`；其中旧相对链接以原仓库根目录为准。

回退某个目录时，先确认原入口仍是指向表中目标的软链接，再 unlink 该软链接并将实际目录 rename 回原名；不要删除软链接所指向的目录。Git diff 会显示 tracked 文件的移动，未提交，供审阅。

## 新输出管理

- 数据生成拒绝覆盖已有版本目录。
- 后续训练使用 `runs/synthetic/<dataset>/<model>/<timestamp>_seed<seed>/`。
- `docs/checkpoints.csv` 索引全部旧 `last.pt`；它是文件清单，不代表 checkpoint 训练完整或质量合格。
- `analysis/ode_spectrum_audit_2026-09-23.json` 是三个旧实验的 raw/EMA 端点谱；没有历史轨迹。
- 新数据、实验产物、归档日志、analysis 大文件由 .gitignore 排除。旧的已追踪日志移动后由 Git 显式标记删除，归档原件仍在磁盘；没有修改 Git index。

## Synthetic Python 环境

`/scratch/zz5070/ODE_FFN/envs/synthetic` 是基于现有 VLM Python 创建的 `--system-site-packages` venv。它复用现有 NumPy/PyTorch 等包，新增绘图依赖安装在自己的目录，不修改 VLM。这是隔离新增依赖的 overlay，不是完全独立的锁定环境。环境记录见 `docs/synthetic-environment.json`；后续训练若需要升级 PyTorch，应建立完全隔离环境。
