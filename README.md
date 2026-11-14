# ODE-FFN

ODE-inspired feedforward layers for flow-matching image models. The active
trainer is copied from [the JiT authors](https://github.com/LTH14/JiT/tree/cbc743a2ada5e9762697da2c83f8c4f8379e8c17);
the model extension replaces only each Transformer block's FFN.

## Start here

| What you want | Open |
|---|---|
| Current experiment, budget and jobs | [STATUS.md](STATUS.md) |
| Candidate FFN implementation | [ffn.py](ffn.py) |
| Training parameters and GPU request | [slurm/celeba.slurm](slurm/celeba.slurm) |
| Research evidence and older experiments | [docs/README.md](docs/README.md) |
| Final reports | [reports/README.md](reports/README.md) |
| Development and server paths | [docs/development.md](docs/development.md) |

## Code map

| File | Responsibility |
|---|---|
| [main_jit.py](main_jit.py) | Arguments, dataset, optimizer, checkpoints and epoch loop |
| [engine_jit.py](engine_jit.py) | One training epoch and image/FID evaluation |
| [denoiser.py](denoiser.py) | Flow-matching objective, EMA and sampling; installs the FFN hook |
| [model_jit.py](model_jit.py) | Author's JiT backbone, attention and native SwiGLU |
| [ffn.py](ffn.py) | ISRU-D and Free ISRU replacements |
| [util/](util/) | Dataset/crop, LR, model and checkpoint utilities |
| [scripts/](scripts/) | Dataset preparation, checks and read-only Slurm monitoring |
| [slurm/](slurm/) | GPU readiness and training submission wrappers |

Read the forward path in this order:
`engine_jit.py -> denoiser.py -> model_jit.py -> block.mlp`.
For the candidate, `block.mlp` is replaced by the implementation in `ffn.py`.

## Current comparison

K1/K8 are [continuing from100to160epochs](docs/celeba128-wide-20261114/continuation-160/README.md),
with standard FID at120/140/160. The original native SwiGLU also
[continues to the same160epochs](docs/celeba128-wide-20261114/continuation-160/baseline/README.md). The completed100epoch results below remain
the original fixed-budget comparison.

Native **SwiGLU** reached FID50k **35.91785** at 100 epochs; its architecture
is unchanged while the baseline also continues to 160.
The current [CelebA128 study](docs/celeba128-wide-20261114/README.md) completed
**widened ISRU-D K8 and K1**, seed 0,100 epochs, with FID50k every20epochs.
Both use internal width 1200 and 130,402,560parameters, versus native 130,405,632.
Final FID50k is 37.92486 (K1) and 38.13574 (K8); see the
[completed results](docs/celeba128-wide-20261114/results.md).
Use `--ffn isru_d --ode_hidden_dim 1200 --ode_order 8` (or1).
ISRU-D holds each entrance diagonal fixed during the Taylor series and uses
two independent operators around SiLU. `--ffn isru` is ordinary Free ISRU.

[STATUS.md](STATUS.md) holds the current budget and job snapshot;
[the budget record](docs/celeba128-wide-20261114/budget.json) explains the estimate.
Runtime flags in `slurm/celeba.slurm` and the recorded submission command
specify execution. JSON files under the study's docs record it; the trainer
does not load those JSON files as configuration.

## Setup and provenance

Use the [current validation/launch guide](docs/celeba128-wide-20261114/README.md) with
the existing [CUDA12.8 profile](requirements-cuda128.txt). The
[development guide](docs/development.md) distinguishes the mutable checkout,
frozen source, environment, dataset and run outputs.

[UPSTREAM.md](UPSTREAM.md), [UPSTREAM.json](UPSTREAM.json) and
[UPSTREAM.patch](UPSTREAM.patch) attribute the copied source and local edits.
The author's [MIT license](LICENSE) is retained. The complete earlier
framework is preserved on [research-archive-20261101](https://github.com/Joni-z/ODE-FFN/tree/research-archive-20261101).
