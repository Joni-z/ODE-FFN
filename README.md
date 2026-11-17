# ODE-FFN

ODE-inspired feedforward layers for flow-matching image models. The active
trainer is copied from [the JiT authors](https://github.com/LTH14/JiT/tree/cbc743a2ada5e9762697da2c83f8c4f8379e8c17);
the model extension replaces only each Transformer block's FFN.

## Start here

| What you want | Open |
|---|---|
| Whole-project findings and every recorded experiment | [17 November project review and full ledger](docs/research-assessments/2026-11-17-project-review/README.md) |
| Final Report3 and professor responses | [PDF](reports/report3/report3.pdf), [source and evidence](reports/report3/README.md) |
| Current experiment, budget and jobs | [STATUS.md](STATUS.md) |
| What the ODE constraints contribute | [Completed matched stage-order study](docs/research-assessments/2026-11-16-ode-mechanism/schedule-order/numerical-results/analysis-report.md) |
| Does that change the learned target response? | [Independent all-bank output-Jacobian comparison](docs/research-assessments/2026-11-16-ode-mechanism/target-geometry/README.md) |
| Current mechanism result and next action | [Parameter-free hidden-bias prediction](docs/research-assessments/2026-11-16-ode-mechanism/solver-bias-results/analysis-report.md), [complete dimension control](docs/research-assessments/2026-11-16-ode-mechanism/hidden-null-results/analysis-report.md) |
| Which hidden change reaches the output? | [Completed single-layer directional intervention](docs/research-assessments/2026-11-16-ode-mechanism/hidden-readout-results/analysis-report.md) |
| Can that mechanism be tested during generation? | [Completed512/1024 distribution comparison](docs/research-assessments/2026-11-16-ode-mechanism/hidden-sampling-extension-results/analysis-report.md), [original sampling protocol](docs/research-assessments/2026-11-16-ode-mechanism/hidden-sampling-protocol.md) |
| Does endpoint linearization measure generation quality? | [Exact actual-law counterexample](docs/research-assessments/2026-11-16-ode-mechanism/target-geometry/product-oracle-control-results/analysis-report.md) |
| Baseline/K1/K8 at matching epochs | [Matched FID comparison](docs/celeba128-wide-20261114/continuation-160/matched-results.md) |
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

## Mechanism study

A matched synthetic control keeps P/B parameters, initialization and matrix
work fixed while changing the order of four hidden ODE stages. The symmetric
schedule improves Ring8 relative to alternating stages and worsens Product16;
both effects survive the registered internal refinements and dyadic outer grids.
[Analysis, figures and statistics](docs/research-assessments/2026-11-16-ode-mechanism/schedule-order/numerical-results/analysis-report.md).
The same focal ordering appears in whole-output Jacobian error on every
independent input bank. This is a confirmed association; ordinary Product16
SwiGLU still samples better despite greater average Jacobian error.
[Derivation and complete confirmation](docs/research-assessments/2026-11-16-ode-mechanism/target-geometry/README.md).
The initial trajectory calibration failed its1024step numerical cap; subsequent
uniform refinement now qualifies all twelve models.
An actual-model temporal-alias construction shows why dyadic-grid agreement
alone does not establish a continuous-field explanation.
[Precision audit](docs/research-assessments/2026-11-16-ode-mechanism/target-geometry/time-alias-results/analysis-report.md).
All twelve full oracle/tangent/blend refinements now qualify. The first-order
endpoint account is modest on Ring8 and fails on Product16at the trained scale.
An exact actual-law counterexample preserves every marginal while failing that
approximation. The next diagnosis returns to the hidden solver's energy/direction.
[Complete results and decision](docs/research-assessments/2026-11-16-ode-mechanism/target-geometry/decision-after-calibration.md).
These mechanism tests use velocity MLPs; the image code below remains the
author-derived JiT trainer.

## Image comparison

The [100-to-160-epoch continuation](docs/celeba128-wide-20261114/continuation-160/README.md)
records FID evaluation at 120/140/160, including the
[native baseline](docs/celeba128-wide-20261114/continuation-160/baseline/README.md).
The 100-epoch results below retain the original fixed training budget.

Native **SwiGLU** reached FID50k **35.918** at 100 epochs.
The [CelebA128 study](docs/celeba128-wide-20261114/README.md) completed
**widened ISRU-D K8 and K1**, seed 0,100 epochs, with FID50k every20epochs.
Both use internal width 1200 and 130,402,560parameters, versus native 130,405,632.
Final FID50k is 37.925 (K1) and 38.136 (K8); see the
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
