# Current project status

## Numerical bias affects generation, but field MSE does not rank its value

The crossed-solver comparison is complete. On Product16, RK4-trained weights
sampled with midpoint reduce confirmation pooled SW2 from 0.065 to 0.060, with the
same direction on the selection split. Both paired adjusted intervals favor
the change. The field-risk forecast predicted the opposite direction. On Ring8,
the clear field-MSE improvement does not yield consistent sampling improvement.
Ordinary Product16 Free ISRU(0.055) and SwiGLU(0.044) remain stronger; no image
architecture is promoted from this within-family result.

[Complete crossed samples and figures](docs/research-assessments/2026-11-16-ode-mechanism/cross-solver-results/analysis-report.md),
[exact data](docs/research-assessments/2026-11-16-ode-mechanism/cross-solver-results/summary.json),
[audit](docs/research-assessments/2026-11-16-ode-mechanism/cross-solver-audit.json).

## What the completed causal chain establishes

1. Equal-work RK4 training reduces hidden continuous-solve error 45.580x/22.245x
   on Ring8/Product16 without improving native target SW2.
2. Full 4,096-input, all-layer interventions show the numerical bias reaches the
   complete field. Both Ring8 weight sets prefer midpoint; a simple own-solver
   compensation story fails. Removing the midpoint bias can worsen field fit.
3. Crossed generation now shows a useful Product16 discretization effect, but
   not one explained by average field MSE. The prescribed field ranking succeeds
   on 4/8 point comparisons and 2/8 adjusted comparisons.

[Matched training](docs/research-assessments/2026-11-16-ode-mechanism/rk4-results/analysis-report.md),
[complete-field diagnosis](docs/research-assessments/2026-11-16-ode-mechanism/field-exposure-results/analysis-report.md).

## Next missing mathematical link

Use the actual learned baseline and a controlled solver perturbation. Differentiate
its native Heun computation and the sorted-projection target metric, then test
whether the ODE-derived hidden bias predicts the observed sampling effect.
This avoids treating source-paired endpoint error or field MSE as distribution
quality. The derivation and calibration/confirmation sequence are registered;
its implementation passes deterministic controls and all eight checkpoint/split
witnesses, including a fresh documented execution at a6f4d42. Fresh evaluation
banks are bound before outcomes. Full transport is now running on four Mac CPU
workers: queue98117, native98125-98128, original tool session15465. No result is
claimed from the64-source witnesses. All four frozen cases, both splits, eight
4096-source banks and four fixed interpolation strengths remain in the queue.
[Mathematical plan](docs/research-assessments/2026-11-16-ode-mechanism/target-aware-bias-transport.md).
[Registered numerical protocol](docs/research-assessments/2026-11-16-ode-mechanism/target-transport-protocol.md).

## Execution and preserved benchmark

All eight native crossed evaluations exit 0. The original verification queue
exit 1 is retained: its confirmation check used selection input references.
Corrected matched-split FP64 references verify all 4,096 field inputs and all original
samples without resampling. A fresh full-cache pass and 16 children exit 0;
independent metric error 5.55e-17 and independent statistics error 2.17e-19.
[Repair record](docs/research-assessments/2026-11-16-ode-mechanism/cross-solver-verification-repair.md).

Earlier field/crossed-sampling workers are terminal. Target-transport workers
are live and near100% of one CPU core each; stages and actual child exits are
recorded by the queue.145 physical4-7 were freshly checked: each uses about14GB,
so none is used;0-3 are prohibited. No packages changed.
Reports1/2 and their 97 protected files/11 bindings, plus five native image source
files, remain unchanged. CelebA128 at 160 epochs stays native 24.304/K1 27.789/K8
28.083 FID50k. Current raw study:
/Users/mr.z/ode-workspace-prep/coefficient-20261116/target-aware-transport/.
Completed field/crossed results remain under coefficient-20261116/rk4-control/.
