# Research and documentation map

For the current run, start at [STATUS.md](../STATUS.md). Historical evidence
keeps its original paths so report links, source bindings and replay scripts
continue to resolve.

## Current image work

| Topic | Entry |
|---|---|
| Current run and actual job snapshot | [Project status](../STATUS.md) |
| Completed wide K8/K1 and fixed native baseline | [Results and FID curves](celeba128-wide-20261114/results.md) |
| Continuation of the same K1/K8 checkpoints to 160 epochs | [Live continuation results](celeba128-wide-20261114/continuation-160/results.md) |
| Completed 100-epoch results | [Training and P figures](celeba128-20261112/results.md) |
| What to test next: order and capacity | [Evidence audit and bounded decision](celeba128-20261112/next-step-audit.md) |
| FID every 20 epochs and standard metric audit | [Periodic evaluation](celeba128-20261112/periodic-fid.md) |
| Fixed-checkpoint FID50k settings | [Evaluation](celeba128-20261112/evaluation.md) |
| CelebA128 comparison and fixed method | [Protocol](celeba128-20261112/protocol.md) |
| Epoch/time budget and measurements | [Budget](celeba128-20261112/budget-20261112.md) |
| Torch commands and checks | [Invocation](celeba128-20261112/validation.md) |
| Recorded flags and numerical configuration | [Launch](celeba128-20261112/launch.json), [configuration record](celeba128-20261112/configuration.json) |
| Selected K8 operator and its derivation | [Candidate](research-assessments/2026-11-11-image-candidate.md), [Taylor order](research-assessments/2026-11-11-taylor-order.md) |
| JiT recipe and newer baseline review | [Recipe review](research-assessments/2026-11-12-training-recipe/assessment.md) |
| Source and server layout | [Development](development.md), [upstream changes](../UPSTREAM.md) |

The dated CelebA folder contains the protocol, measured readiness, launch
receipts and superseded attempts for one study. Its JSON documents record
execution; active flags live in [the Slurm wrapper](../slurm/celeba.slurm).

## Completed research

| Question or deliverable | Entry |
|---|---|
| Submitted progress reports | [Reports index](../reports/README.md) |
| Full ODE-FFN attempt ledger | [Report2 attempts](../reports/report2/attempts.md) |
| Muon, P spectra and input-dependent diagonals | [Research map](autoresearch/professor-20261103/research-map.md) |
| ISRU-D versus ordinary controls on Ring8/Product16 | [Complete common-bank comparison](autoresearch/professor-20261103/cycle14-results-complete/analysis-report.md) |
| Field fitting versus generated mode allocation | [Transport diagnosis](autoresearch/professor-20261103/cycle15-results/analysis-report.md) |
| Earlier synthetic evidence audit | [Audit](synthetic-audit-2026-11-01/EXPERIMENT_AUDIT.md) |
| Earlier64px image allocation | [CelebA64](celeba-k8-20261111/README.md) |
| Earlier CIFAR/Windows attempt | [Lenovo baseline](lenovo-baseline/README.md) |

`autoresearch/`, `research-*/` and `literature-*/` retain detailed historical
experiments and readings. `studies/` is the corresponding separate synthetic
implementation. These folders are not imported by the active image trainer.

## History and older layouts

The [history index](history/README.md) preserves earlier long journals and
navigation text. [legacy-overview.md](legacy-overview.md),
[workspace.md](workspace.md) and [organization-manifest.json](organization-manifest.json)
describe the old `jit-codebase/src` workspace and earlier migrations.
They are historical records; use the current development guide for paths.
The full older training framework remains on
[research-archive-20261101](https://github.com/Joni-z/ODE-FFN/tree/research-archive-20261101).
