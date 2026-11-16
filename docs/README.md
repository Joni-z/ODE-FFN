# Research and documentation map

For the current run, start at [STATUS.md](../STATUS.md). Historical evidence
keeps its original paths so report links, source bindings and replay scripts
continue to resolve.

## Current image work

| Topic | Entry |
|---|---|
| Current run and actual job snapshot | [Project status](../STATUS.md) |
| What does the ODE structure contribute? | [Falsifiable mechanism plan](research-assessments/2026-11-16-ode-mechanism/research-question-card.md), [mathematics](research-assessments/2026-11-16-ode-mechanism/mathematical-basis.md), [primary literature checks](research-assessments/2026-11-16-ode-mechanism/literature-notes.md) |
| Does the stage-sharing comparison change optimizer coordinates? | [Completed mean-preserving control](research-assessments/2026-11-16-ode-mechanism/coordinate-control/analysis-report.md), [paired-bank statistics](research-assessments/2026-11-16-ode-mechanism/coordinate-control/stats-appendix.md) |
| Are the new control's sampling and density comparisons numerically qualified? | [Completed density analysis and three figures](research-assessments/2026-11-16-ode-mechanism/coordinate-control/density-results/analysis-report.md), [all paired statistics](research-assessments/2026-11-16-ode-mechanism/coordinate-control/density-results/stats-appendix.md), [registered protocol](research-assessments/2026-11-16-ode-mechanism/coordinate-control/observation-protocol.md), [resource handoff](research-assessments/2026-11-16-ode-mechanism/coordinate-control/observations/resource-handoff.md) |
| Completed matched stage-order test | [Full numerical results and figures](research-assessments/2026-11-16-ode-mechanism/schedule-order/numerical-results/analysis-report.md), [paired statistics](research-assessments/2026-11-16-ode-mechanism/schedule-order/numerical-results/stats-appendix.md), [registered same-P/B design](research-assessments/2026-11-16-ode-mechanism/schedule-order-control.md) |
| What does stage ordering add beyond the average generator? | [Direct derivation](research-assessments/2026-11-16-ode-mechanism/coordinate-control/mean-flow-derivation.md), [native diagnostic and figures](research-assessments/2026-11-16-ode-mechanism/coordinate-control/schedule-decomposition/analysis-report.md), [completed numerical sweep](research-assessments/2026-11-16-ode-mechanism/coordinate-control/numerical-resolution/analysis.md) |
| Does sharing reduce joint density error? | [Product16 result and geometry](research-assessments/2026-11-16-ode-mechanism/density-transport/population-analysis.md), [paired statistics](research-assessments/2026-11-16-ode-mechanism/density-transport/population-stats-appendix.md), [mean-update control](research-assessments/2026-11-16-ode-mechanism/sharing-optimization-control.md) |
| Completed coefficient/state-flow/sharing controls | [Primary findings](research-assessments/2026-11-16-ode-mechanism/primary-findings.md), [sharing analysis and figures](research-assessments/2026-11-16-ode-mechanism/sharing-results/analysis-report.md), [completed numerical controls](research-assessments/2026-11-16-ode-mechanism/numerical-findings.md), [completed symmetry intervention](research-assessments/2026-11-16-ode-mechanism/sharing-symmetry/analysis-report.md), [density-error derivation](research-assessments/2026-11-16-ode-mechanism/post-symmetry-decision.md), [numerical qualification](research-assessments/2026-11-16-ode-mechanism/density-protocol.md) |
| Completed eight-response K1/K8 screen and independent confirmation | [Full comparison](research-assessments/2026-11-15-response-screen/current-comparison.md), [CelebA decision](research-assessments/2026-11-15-response-screen/analysis-report.md), [paired-bank statistics](research-assessments/2026-11-15-response-screen/stats-appendix.md) |
| Response selection before CelebA transfer | [Bounded comparison and selection rules](research-assessments/2026-11-14-next-architecture/response-screen-proposal.md) |
| Completed wide K8/K1 and fixed native baseline | [Results and FID curves](celeba128-wide-20261114/results.md) |
| Continuation of the same K1/K8 checkpoints to 160 epochs | [Live continuation results](celeba128-wide-20261114/continuation-160/results.md) |
| Native SwiGLU continuation through the same 160 epochs | [Baseline continuation](celeba128-wide-20261114/continuation-160/baseline/README.md) |
| Baseline/K1/K8 at the same epochs | [Matched FID table and curves](celeba128-wide-20261114/continuation-160/matched-results.md) |
| Next architecture: frozen versus evolving ISRU response | [Execution plan](research-assessments/2026-11-14-next-architecture/execution-plan.md), [derivation](research-assessments/2026-11-14-next-architecture/recommendation.md) |
| Latest professor comments: spectra, Ring8 symmetry, activation control | [Evidence and revised priority](research-assessments/2026-11-15-professor-comments.md) |
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

## 15 November professor follow-up

- [Completed K1/K8 by ISRU/tanh comparison and term diagnostics](research-assessments/2026-11-15-order-response/findings.md)
- [Execution and four-GPU allocation](research-assessments/2026-11-15-professor-execution/launch.md)
- [Registered response and symmetry tests](research-assessments/2026-11-15-professor-execution/protocol.md)
- [Actual weights versus updates: completed spectral analysis](research-assessments/2026-11-15-professor-execution/spectrum-findings.md)

- [Completed15November findings, figures and next ODE hypothesis](research-assessments/2026-11-15-professor-execution/findings.md)
