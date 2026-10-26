# Scope and comparison units

All fifteen trials follow the user's fixed seed0 protocol. Reported ratios use full stored values and are descriptive. No between-training significance test or confidence interval is estimated. The three ODE arms have the same parameter counts and linear MACs; both practical SwiGLU budgets remain in every comparison.

The paired transform keeps the initial hidden function exact, as checked with nonzero output heads on native CPU and CUDA. The unpaired tau control changes it and therefore is not described as a pure initial-function-matched intervention. Both retain the same global feasible function class. All reference/candidate/control snapshot data-generator states match under the identical bank sampler. This is distinct from Stage10's independently regenerated, per-minibatch byte-fingerprint audit and is not reported as that stronger check.

Primary metric: final random-time validation oracle MSE. Supporting outcomes: all curves, common Jacobian error, validation sampling SW2, costs and P/controller trajectories. The registered continuation rule is failed and unchanged. Six internal matrices/controllers summarized within a trained run are repeated layer measurements, not six independent model replications. Probe sample variation includes both state and time.

Native audit verifies90 checkpoints/75 arrays and initial-tensor/data-RNG invariants. Selected raw collection and Mac source/weight replay have separate recorded scopes in collection.json and their audit summaries. No new holdout or sampling-trajectory replay is claimed. Mathematical notes added during readout clarify the tangent-kernel identity and the zero-output-head limitation; frozen runtime/protocol documents remain preserved and no decision threshold changed.
