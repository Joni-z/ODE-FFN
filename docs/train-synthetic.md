# Synthetic 首轮训练与作业记录

提交状态：pilot 已正常完成；用户已授权继续完成另两个数据集的 K=2 对照，以及三个数据集的 K=1 消融。最新作业记录见文末；消融设计见 `synthetic-ablation-v1.md`。

## 实验矩阵

`configs/synthetic/train_v1.json`：3 数据集 × 3 模型 × 1 seed（只用 0）= 9 个独立 run。每个 run 5000 步，batch 512，AdamW lr 3e-4，Heun 32 步抽样，固定测试来源和 128 个 sliced-Wasserstein 投影。先准备 pilot：各向异性 Gaussian × 3 模型 × seed 0，共 3 个单 GPU 作业，最多同时 3 个，每个限 2 小时/16 GB RAM/4 CPU，使用已验证的 `torch_pr_63_tandon_advanced` account，申请 1 张 A100 / L40S / H100 / H200 中任意可用的 GPU，总申请上限 6 GPU 小时。其余 6 个组合另有 `train_synthetic_remaining.slurm`，待 pilot 结果与耗时检查后再决定。后续 6 个组合根据 pilot 耗时将每任务时限改为 20 分钟；含已完成 pilot 的配置，全矩阵 9 个组合的总申请上限为 8 GPU 小时；这只是 Slurm 资源上限，不是已测出的消耗。

首轮数据：各向异性 Gaussian、8-Gaussian ring、Two Moons。模型：

- `ffn`：标准两层 SiLU FFN，hidden=128。
- `ode_original`：教授代码的两次 ODELayer 求和，中间 SiLU；每层 K=2、s(x)=0.2+0.8*sigmoid(g(x)/10)，内部 P 无约束。
- `ode_spectral_cap`：完全相同的 ODEFFN，前向使用 P_eff=P*min(1,1.0/sigma_max(P))。64×64 上每次精确 SVD，计算开销更高。P_raw 仍保留用于观察优化时是否试图超出上限。

外层网络对三者一致：输入投影、三组时间 FiLM + LayerNorm + residual FFN、输出投影；内部 FFN 参数量约匹配，最终实际参数数由 `provenance.json` 记录。ODEFFN 的内部 s(x) 与外部 flow-matching 时间 t 是不同变量。t 通过共享的时间条件网络进入每个 block。

不把这个 toy 向量模型称为 DiT；它只是用低成本受控任务检验 ODEFFN 机制。K=2 的泰勒多项式总存在；P 的谱上限是一个待验证的干预，不是级数收敛的必要条件，也不保证整个非线性网络 Lipschitz 常数受限。

## 必须看的输出

每个 run 目录 `../runs/synthetic/odeffn_synthetic_v1/<dataset>/<model>/seed<seed>/`：

- `metrics.jsonl`：第 0 步和每 500 步的固定验证配对 MSE；Gaussian/GMM 还有相对解析速度场的 MSE。后者更直接反映拟合误差；配对 MSE 包含无法消除的随机配对方差。
- 每 1000 步和最终结果：Heun 32 的生成样本 sliced-Wasserstein 距离及独立测试样本之间的估计噪声底。二维不使用 Inception FID。三模型相同 NFE 和同一固定噪声/测试参考。
- 每个评估点 `ode_layers`：P_raw 和 P_eff 的奇异值分位数、最大奇异值、有效秩、特征值实部/半径、非正规性；固定 probe 在**实际隐藏层输入**上的 s(x) 分布和一、二阶项相对输入 RMS。这些指标与同一行的误差对齐。
- `checkpoints/step_*.pt`：第 0 步、每 1000 步、最终步，包含模型、优化器、随机状态与哈希。用唯一文件名；旧实验覆盖 last.pt 的问题不会重现。`--resume` 需要显式指定 checkpoint，配置、数据哈希、数据集、模型或 seed 不同时拒绝；续训事件写入 `resumes.jsonl`。
- `provenance.json`：完整配置、数据说明、代码 SHA256、Git 修订与脏工作区状态、参数量及依赖版本。

## 提交前检查

在 torch 的项目目录运行：

```bash
PYTHON=/scratch/zz5070/ODE_FFN/envs/synthetic/bin/python
OMP_NUM_THREADS=1 "$PYTHON" -m unittest discover -s tests -p 'test_synthetic_*.py' -v
OMP_NUM_THREADS=1 "$PYTHON" scripts/preflight_synthetic.py
"$PYTHON" -m synthetic.grid
bash -n scripts/train_synthetic_pilot.slurm
bash -n scripts/train_synthetic_remaining.slurm
mkdir -p ../runs/synthetic/slurm
sbatch --test-only scripts/train_synthetic_pilot.slurm
sbatch --test-only scripts/train_synthetic_remaining.slurm
```

Pilot 提交命令（已获用户授权；避免重复提交）：

```bash
sbatch scripts/train_synthetic_pilot.slurm
```

Pilot 的三个任务均为 COMPLETED/0:0。用户已授权提交其余 6 个 array task，其命令是 `sbatch scripts/train_synthetic_remaining.slurm`。这两个脚本专用于新 synthetic 实验。GPU 申请按 Torch 当前规则和账户权限单独验证，不引用旧训练脚本。数据集及环境入口见 `docs/synthetic.md`；数据和旧结果依然保持原位。

## GPU 申请规则（2026-10-06 验证）

```bash
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|l40s|h100|h200"
#SBATCH --account=torch_pr_63_tandon_advanced
```

`|` 是 OR：每个 array task 申请一张上述任意型号的 GPU，不是同时申请四种卡。Torch 会自动选择账户可访问的匹配分区，不手动指定 `--partition`。依据：[Torch 官方作业提交说明](https://services.rt.nyu.edu/docs/hpc/submitting_jobs/slurm_submitting_jobs/)。

使用相同账户分别对 A100、L40S、H100、H200 执行 `sbatch --test-only`，四者均通过；RTX6000 的检查返回 `No partition fits this job request`，因此暂不纳入范围。`--test-only` 不创建作业，返回的预计开始时间不代表 GPU 已分配。

脚本保留 Slurm 设置的 `CUDA_VISIBLE_DEVICES`，启动时检查 CUDA 并把可见 GPU 的实际型号和 PyTorch/CUDA 版本写入 Slurm 日志。比较不同 run 的墙钟耗时时须注明 GPU 型号；模型、数据、seed 和训练步数仍按实验矩阵执行。

## 提交记录

- 2026-10-06：用户授权提交 pilot。首次作业 19273725 使用 seed 11；按用户随后要求撤销。最终配置只包含 seed 0，pilot 的 array index 为 0–2；其余 6 个组合的 index 为 3–8，仅准备脚本。
- 2026-10-06：seed 0 的 pilot 已正式提交，Slurm array job ID 为 19273875（任务 0–2，三个模型，各向异性 Gaussian）。修改后的 9 项映射、三模型 CPU dry-run、三份数据哈希及脚本语法检查均通过。其余 6 个组合未提交。
- 2026-10-06：pilot 19273875 的三个任务完成 5000 步并保存 COMPLETE 与最终 checkpoint，实际 GPU 均为 L40S。用户要求继续跨数据集验证及 K=1 对照，所有实验仍只用 seed 0。
- 2026-10-06：跨数据集 K=2 的六个任务已提交，Slurm array job ID 为 19283509。K=1 已准备并通过检查，但提交前 SSH 认证失效，等待恢复。
- 2026-10-09：九个 K=2/FFN run 均完成。K=1 的六个 run 正式提交为数组作业 19469259，只使用 seed 0；任务与比较口径见 synthetic-ablation-v1.md。
- 2026-10-13：包含 K=1 消融的全部 15 个 run 完成，分析和图已整理。完整分析入口为 docs/analysis-2026-10-13/analysis-report.md（torch 仓库路径）。
