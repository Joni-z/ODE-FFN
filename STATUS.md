# Current project status

## Current precision question

**Native endpoint precision now qualifies for all twelve frozen models.**
Ten pass at4096steps; Ring8released-stage and Product16SwiGLU require8192.
Every case retains256sources across all eight banks. All twelve native tasks
and all twelve independent verifiers exit0; reconstruction error<=4.17e-17.
Nearby1000/1024/1032grids still give different endpoints at fixed weights.
[Complete native precision analysis and figures](docs/research-assessments/2026-11-16-ode-mechanism/target-geometry/native-time-results/analysis-report.md).

Next refine the full oracle/tangent/four-blend calculation under the same
component tolerances. Native qualification does not yet validate its
first-order mechanism. [Registered extension](docs/research-assessments/2026-11-16-ode-mechanism/target-geometry/refined-trajectory-protocol.md).
No new training or image architecture is launched.145physical4-7remain
occupied; MacCPUis used without sharing or stopping another job.

All twelve256-source error-transport calibrations finished. None passes every
registered endpoint/tangent component gate by1024steps, so the first-order
mechanism remains unresolved. No sources or models were removed.

An actual-model time-alias construction passes on all twelve checkpoints:
the highest frequency1000.000122can be shifted by-1024without changing the
dyadic-grid function in exact arithmetic. Measured FP64Heun128endpoint change
is at most1.65e-13; FP32at most3.02e-5. Off-grid fields and selection Jacobian
metrics change. The focal ordering does not reverse. This exposes a blind spot
in the earlier step checks; it does not invalidate fixed-gridSW2or prove the
cause of every precision failure.

[Calibration](docs/research-assessments/2026-11-16-ode-mechanism/target-geometry/trajectory-results/analysis-report.md),
[alias construction and figures](docs/research-assessments/2026-11-16-ode-mechanism/target-geometry/time-alias-results/analysis-report.md),
[next precision protocol](docs/research-assessments/2026-11-16-ode-mechanism/target-geometry/native-time-resolution-protocol.md).
Both native CPUtasks, both independent verifiers and the source audit exited0.
145physical4-7were occupied, and no foreign job was touched. No new training.

## Target-response observation

The Bayes clean-data predictor has a posterior-covariance Jacobian; this is
an existing whole-output identity, not a hidden-P constraint. The initial
twelve-checkpoint observation and independent all-bank confirmation are done.

| Full independent inputs | PAL J error | ALT J error | PAL SW2 | ALT SW2 |
|---|---:|---:|---:|---:|
| Ring8 |0.156|0.191|0.086|0.108|
| Product16 |0.261|0.236|0.072|0.066|

Every one of eight banks supports each task's focal direction. Conditional
two-contrast intervals exclude zero. This confirms an output-response
association, not its causal role. Ordinary Product16SwiGLU samples better
despite higher averageJerror, ruling out a general cross-model ranking.

[Derivation](docs/research-assessments/2026-11-16-ode-mechanism/target-geometry/derivation.md),
[complete selection analysis and figures](docs/research-assessments/2026-11-16-ode-mechanism/target-geometry/results/analysis-report.md),
[completed independent confirmation and figures](docs/research-assessments/2026-11-16-ode-mechanism/target-geometry/confirmation-results/analysis-report.md).
Both one-thread CPUtasks and both independent verifiers exited0; all655360
interpolation states and original source/target arrays are retained. NoGPU
was used. No new training or image architecture follows from this association.

## Completed matched stage-order study

**Palindrome helps Ring8 relative to alternating stages and hurts Product16.
Both effects survive the full registered numerical checks.**

| Frozen weights / Heun128 | Ring8 PAL | Ring8 ALT | Product16 PAL | Product16 ALT |
|---|---:|---:|---:|---:|
| Trained internal4 |0.086|0.108|0.072|0.066|
| Internal16 |0.084|0.107|0.087|0.075|

The two schedules share P/B coordinates, capacity, initial functions and work.
The predicted leading bracket cancellation holds in trained palindrome
features. Its utility is task dependent. Finer hidden integration worsens
Product16sampling at these frozen trained weights. Ordinary Product16
Free ISRU/SwiGLU remain 0.055/0.044at the original Heun128 setting.

[Complete numerical analysis and actual figures](docs/research-assessments/2026-11-16-ode-mechanism/schedule-order/numerical-results/analysis-report.md),
[paired statistics](docs/research-assessments/2026-11-16-ode-mechanism/schedule-order/numerical-results/stats-appendix.md),
[all earlier mechanism findings](docs/research-assessments/2026-11-16-ode-mechanism/primary-findings.md).

## Execution and verification

Four 20k seed-0 trajectories and complete common-bank evaluation are done.
All eight full numerical tasks at frozen 303d213 exited 0. NumPy independently
reconstructs all 24 rows (twenty unique cells) within 2.78e-17, retaining every
source. The complete CPU bankwise run is the analysis population. Its earlier
full-batch duplicate was intentionally retired after 10 exact prefix audits;
actual exits and all partial outputs remain archived. All these processes
have ended; no GPU or foreign job was touched.

[Complete handoff record](docs/research-assessments/2026-11-16-ode-mechanism/schedule-order/bankwise-handoff-complete.json).
Both finalized reports and the native JiT image baseline remain fixed.
Raw evidence: /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order.

## Current research priority

Native fields now meet the registered precision test. Resolve the
transported-error approximation before another architecture. The hidden ODE time/global
FM time connection remains open. No additional training or image promotion is
queued. The broader ODE-mechanism research goal remains active.
[Registered bounded calibration](docs/research-assessments/2026-11-16-ode-mechanism/target-geometry/trajectory-protocol.md).

## Image benchmark

The native SwiGLU/K1/K8image comparison and its recorded continuation stay at
[the matched FID table](docs/celeba128-wide-20261114/continuation-160/matched-results.md).
Synthetic studies use velocity MLPs and carry their own fixed data/weights.
Earlier detailed execution snapshots are archived under docs/history/.
