# ODELayer 研究推进

主仓库：torch:/scratch/zz5070/ODE_FFN/jit-codebase。
本地准备目录：/Users/mr.z/ode-workspace-prep。

- design.md：从原版公式、能量变化、门控 Jacobian、Taylor 反例到 FM 误差关系的完整推导；结构化 P 候选与后续消融。
- research-question-card.md：研究问题、证据、支持与否定条件。
- notes.md：原始文献与证据边界。
- math-verification.json / verify_math.py：代数数值检查，可复现。
- verify_baseline.py：旧行为回归与 SwiGLU 检查；models-before-swiglu.py 为本地审计快照。
- diagnostics/*.json：Slurm 完成后输出的六模型/数据集完整诊断。

本轮只补训缺失的 SwiGLU（三个数据集、seed0）；已有15个实验不重复训练。作业20314159；诊断作业20314171依赖前者全部成功。全部训练与诊断已完成；结果见 results.md。新 P 结构尚未训练，优先做保留P/K的门控消融，再做同族生成元的exact-vs-Taylor对照。

查看统一18-run摘要：

```sh
python scripts/summarize_synthetic.py --configs train_v1.json train_k1_v1.json train_swiglu_v1.json --json-output docs/research-2026-10-19/comparison-summary.json --require-complete
```

数学/代码检查在 CPU 执行。训练仅使用 Slurm GPU，申请 a100|l40s|h100|h200；不手动指定 partition。Heun256 是数值参考，必须检查128→256收敛量；局部 Jacobian probes 不是全局 Lipschitz 界。所有性能结论仅限seed0当前协议。
