# Development and server layout

Canonical repository: [Joni-z/ODE-FFN](https://github.com/Joni-z/ODE-FFN),
branch `main`. The root Python files follow the pinned JiT author's layout.
The [repository map](README.md) separates active image work from historical
synthetic studies. For the current run, use [STATUS.md](../STATUS.md).

## Code and data on torch

| Path under `/scratch/zz5070/ODE_FFN/` | Purpose |
|---|---|
| `ODE-FFN/` | Mutable Git development checkout |
| `runtimes/celeba128-20261112/` | Frozen source 65864ad used by the current jobs |
| `envs/jit-celeba-cu128/` | Existing Python 3.10/PyTorch 2.7.1+cu128 environment |
| `datasets/celeba/` | Original aligned CelebA images and official split metadata |
| `runs/celeba/20261112-b16-swiglu-s0/` | Baseline checkpoints, epochs and GPU records |
| `runs/celeba/20261112-b16-isru_d-s0/` | Candidate checkpoints, P trajectories and GPU records |
| `operations/celeba128-20261112/` | Submission receipts and live monitor.json |

The local editing checkout is `/Users/mr.z/ODE-FFN-upstream`. Code sync uses
GitHub; keep each checkout clean before a fast-forward update. A queued
Slurm job stores its wrapper at submission and executes the frozen model
runtime passed through ODE_FFN_CODE. Editing the mutable checkout does not
change those queued jobs.

The older `jit-codebase/` path and Mac/Lenovo runs are retained as history.
They are not the current torch development checkout. The original development
page is preserved [verbatim](history/2026-11-12-repository-organization/development-before.md).

## Where to make a change

| Change | File |
|---|---|
| Candidate feature transformation | [ffn.py](../ffn.py) |
| JiT backbone or native SwiGLU definition | [model_jit.py](../model_jit.py) |
| Training budget, batch, workers and GPU request | [slurm/celeba.slurm](../slurm/celeba.slurm) |
| Entry-point data/checkpoint adapter | [main_jit.py](../main_jit.py) |
| Objective, EMA and sampler | [denoiser.py](../denoiser.py) |

The last file contains copied author mathematics; FFN work normally uses the
hook rather than changing that objective. [UPSTREAM.md](../UPSTREAM.md)
records the source boundary. Use the existing
[invocation and validation guide](celeba128-20261112/validation.md) for checks.

Datasets, environments, checkpoints and raw logs remain on scratch and are
excluded from Git. Publish code, bounded configuration records, reviewed
metrics and figures as Zhizhe Zhang. Preserve the completed reports and
frozen run source. Keep current plan/notes short; archive completed journals
under [history](history/README.md).
