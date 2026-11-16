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
workers: queue98117, original tool session15465. No result is
claimed from the64-source witnesses. All four frozen cases, both splits, eight
4096-source banks and four fixed interpolation strengths remain in the queue.
Ring8 is complete: four full cases, all eight native/independent exits0 and all
numerical gates pass. Latest recorded progress is52/64model-bank evaluations,
49qualified. Three Product16 banks fail the original target finite-difference
check at epsilon1e-4 while primal/endpoint-tangent checks pass. On two debugging banks, independent
Torch autograd agrees with each infinitesimal metric derivative within2.17e-19;
projection-rank changes explain the finite-difference discrepancies within
1.93e-14. The original failures remain failures; this debugging subset does not
qualify the full cohort or full-scale prediction. Product16 continues under
native PIDs1713/1715/1737/1741, about9minutes/bank with four CPUworkers.
Collector98379/session82357 watches this specific queue identity and will run
the source-bound complete-cohort analysis once the queue actually exits.
[Completed Ring8 execution/verification](docs/research-assessments/2026-11-16-ode-mechanism/target-transport-ring-completion.json).
[Target-derivative diagnosis and unchanged original gates](docs/research-assessments/2026-11-16-ode-mechanism/target-derivative-diagnosis.md).
[Fresh diagnostic readiness](docs/research-assessments/2026-11-16-ode-mechanism/metric-rank-readiness.json): exact commands0/0/0, three source hashes unchanged. A single cached-only actor waits for the original postprocessor332, then checks all64banks and8pooled cases; no new model evaluation.
[Mathematical plan](docs/research-assessments/2026-11-16-ode-mechanism/target-aware-bias-transport.md).
[Registered numerical protocol](docs/research-assessments/2026-11-16-ode-mechanism/target-transport-protocol.md).

An analytic Gaussian FM control now proves how temporal cancellation can give
better final distribution matching at larger field MSE. Native Heun endpoints
and tangents verify it within1.07e-6. It also separates endpoint-linearization
error from the squared-distance functional's own nonlinear remainder. The
original primary transport gate remains unchanged.
[Derivation, Section5](docs/research-assessments/2026-11-16-ode-mechanism/target-aware-bias-transport.md),
[deterministic control](docs/research-assessments/2026-11-16-ode-mechanism/gaussian-metric-control.json).
The cached-output decomposition is implemented; all64witness metric rows agree
with native Torch SW2 within8.88e-16. The original primary gate remains unchanged.
[Exact postprocessing protocol](docs/research-assessments/2026-11-16-ode-mechanism/metric-remainder-execution.md).
That protocol now passes a fresh exact three-command witness (0/0/0,5026source
hashes unchanged). Postprocessor332/session64468 watches the existing collector
and will run the three full cached checks once, after actual completion. Do not
duplicate its commands while it is live. No new sampling or training is added.

The [six-paper source audit](docs/research-assessments/2026-11-16-ode-mechanism/literature-scope-audit.md)
confirms that useful coarse-solver bias is established prior work. The pending
contribution is the unfitted hidden-bias/full-field/target-effect chain, with
the original primary gate and ordinary controls retained.

## Execution and preserved benchmark

All eight native crossed evaluations exit 0. The original verification queue
exit 1 is retained: its confirmation check used selection input references.
Corrected matched-split FP64 references verify all 4,096 field inputs and all original
samples without resampling. A fresh full-cache pass and 16 children exit 0;
independent metric error 5.55e-17 and independent statistics error 2.17e-19.
[Repair record](docs/research-assessments/2026-11-16-ode-mechanism/cross-solver-verification-repair.md).

Earlier field/crossed-sampling workers are terminal. Target-transport workers
are live and near100% of one CPU core each; stages and actual child exits are
recorded by the queue.145physical4-7now each use2MiBwith no resident compute PID; the current
Mac queue remains unchanged.0-3are prohibited. No packages changed.
GitHub4a409d9is staged cleanly in a separate sparse checkout on145, using its
existing py312/Torch2.8.0+cu128; CUDAvalidation for this observer is pending.
[Resource/source preparation](docs/research-assessments/2026-11-16-ode-mechanism/resource-preparation-145.json).
The conditional full-model defect observer now passes fresh documented
execution73860/actual0at4a409d9: all5041source hashes unchanged, whole-model
JVP8.14e-20/FD8.95e-12, nonzero fixtures and exact zero/sign/native-replay
controls. Twenty-five generated bytecode files are recorded and retained; no
source changes. No learned-state application has run yet.
[Full-model derivation](docs/research-assessments/2026-11-16-ode-mechanism/hidden-defect-transport.md),
[fresh execution receipt](docs/research-assessments/2026-11-16-ode-mechanism/hidden-defect-readiness.json).
Reports1/2 and their 97 protected files/11 bindings, plus five native image source
files, remain unchanged. CelebA128 at 160 epochs stays native 24.304/K1 27.789/K8
28.083 FID50k. Current raw study:
/Users/mr.z/ode-workspace-prep/coefficient-20261116/target-aware-transport/.
Completed field/crossed results remain under coefficient-20261116/rk4-control/.
