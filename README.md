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

Native **SwiGLU** versus **ISRU-D K8**, unconditional CelebA, model seed 0.
ISRU-D keeps each entrance diagonal fixed throughout eight Taylor terms,
with two independent operators around SiLU. The existing `--ffn isru`
option is ordinary Free ISRU; it is a separate control.

The [current protocol](docs/celeba128-20261112/protocol.md) fixes the data and
method. [STATUS.md](STATUS.md) holds the current budget and job snapshot;
[the budget note](docs/celeba128-20261112/budget-20261112.md) explains the estimate.
Runtime flags in `slurm/celeba.slurm` and the recorded submission command
specify execution. JSON files under the study's docs record it; the trainer
does not load those JSON files as configuration.

## Setup and provenance

Use the [Torch invocation guide](docs/celeba128-20261112/validation.md) with
the existing [CUDA12.8 profile](requirements-cuda128.txt). The
[development guide](docs/development.md) distinguishes the mutable checkout,
frozen source, environment, dataset and run outputs.

[UPSTREAM.md](UPSTREAM.md), [UPSTREAM.json](UPSTREAM.json) and
[UPSTREAM.patch](UPSTREAM.patch) attribute the copied source and local edits.
The author's [MIT license](LICENSE) is retained. The complete earlier
framework is preserved on [research-archive-20261101](https://github.com/Joni-z/ODE-FFN/tree/research-archive-20261101).
