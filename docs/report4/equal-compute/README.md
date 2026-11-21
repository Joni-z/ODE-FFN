# Equal training compute on Ring8 and Product16

The midpoint model retains its Ring8 lead when ordinary FFNs spend the same training budget. On Product16, both ordinary models finish ahead of midpoint. Free ISRU has the best Product16 endpoint, while the original 20k SwiGLU checkpoint still has a lower SW2 than any of the extended controls.

| Model at the training budget | Ring8 SW2 | Product16 SW2 |
|---|---:|---:|
| Free ISRU | 0.091 | **0.043** |
| SwiGLU | 0.085 | 0.047 |
| Four-step midpoint | 0.071 | 0.052 |
| Same midpoint weights, two steps at inference | **0.069** | 0.050 |

The values above pool 32,768 generated examples per task. Ring8 native midpoint has lower SW2 than each ordinary control in five of eight individual banks. On Product16, each ordinary control has lower SW2 than native midpoint in five of eight banks. An independent NumPy recomputation using saved samples, targets and all 256 directions reproduced the four new pooled scores within 2e-17.

On Ring8, both ordinary models fit the velocity field more accurately at their endpoints (MSE 0.006 for each, compared with 0.011 for native midpoint), yet their samples have higher SW2. On Product16, continuing Free ISRU improves its pooled SW2 from 0.051 to 0.043. Continuing SwiGLU raises it from 0.034 to 0.047. Every main comparison uses the preassigned budget endpoint.

[Endpoint scores, checkpoint hashes and individual banks](results.json)

Compare the native 20,000-update, four-step midpoint model against Free ISRU and SwiGLU at their original widths. The primary metric is pooled SW2 on the fixed Heun128 confirmation banks. The two-step inference version of the same midpoint checkpoint is the secondary comparison.

Each ordinary run resumes from the complete seed-0 step-20,000 checkpoint, including optimizer and RNG states. The four target update counts were fixed before training. The original data, batches of 512 and constant learning rates continue. Existing training directories remain separate from the continuations.

[Budget by operation and model](budget.json) · [Profiler, continuation and monitor](../../../studies/muon_pd_20261102/equal_compute_20261122/)

The budget counts arithmetic in the model and loss, backward pass, gradient clipping and optimizer, including Muon's Newton-Schulz products. Matrix multiply-add counts as two FLOPs. Fused normalization and activation kernels follow fixed scalar-operation estimates; elementary functions count as one each. Memory operations, random sampling, monitoring and held-out evaluation are outside the training budget. Each control stops at the nearest integer update to its dataset's midpoint budget. The monitor writes phase and step every 15 seconds and starts the eight-bank evaluation when an arm finishes.

## Fixed training budgets

| Task | Midpoint training FLOPs (trillion) | Midpoint updates | Free ISRU updates | SwiGLU updates |
|---|---:|---:|---:|---:|
| Ring8 | 14.637 | 20,000 | 83,609 | 69,061 |
| Product16 | 14.731 | 20,000 | 81,945 | 44,089 |

Widths and optimizers follow the original ordinary models. Ring8 uses width-64 SwiGLU with Muon for its projection matrices. Product16 uses width-192 SwiGLU with AdamW. Both Free ISRU runs and midpoint use AdamW.

## Reproducing the comparison on Mac mini

From the repository root, use the original synthetic environment. The monitor writes a live `progress.json`, per-arm console logs, separate continuation checkpoints and per-bank evaluation outputs under the work directory.

```sh
prep="$HOME/ode-workspace-prep"
py="$prep/.venv-test/bin/python"
study="studies/muon_pd_20261102/equal_compute_20261122"
work="$prep/report4-equal-compute-20261122"
"$py" "$study/budget.py" "$prep/coefficient-20261116" docs/report4/equal-compute/budget.json
"$py" "$study/monitor.py" "$work" "$prep/coefficient-20261116/assets"
"$py" "$study/collect.py" "$work"
```
