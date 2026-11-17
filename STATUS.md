# Current project status

## User-requested pause and handoff

17 November: the user requested closing the current code/evidence changes.
Autonomous development and new experiment decisions are paused. Existing
registered queues remain running:145controller1477869on physical4-7,
Macall-source controller15486, ordinary-control controller19767and the fixed
cached-comparison follower20623(owner20622/tool67368). Their progress/logs
remain under the raw roots recorded below. The follower waits for the full
ordinary cohort before generating its comparison; public conclusions still
require independent checks and visual review. The investigation is unfinished.

## Analytic target forecast: complete Heun128 diagnostic, Heun2048 running

Both primary Heun128 streams retain all eight original banks and independent
reconstructions. The ODE-derived forecast has the correct sign and beats
predicting zero change on both complete 32,768-sample pools.

| Product16 RK4 weights | Actual squared-loss change | Analytic forecast | Forecast score |
|---|---:|---:|---:|
| Selection |-4.122e-4|-4.878e-4|0.966|
| Confirmation |-2.789e-4|-4.161e-4|0.758|

Score is one minus squared forecast error relative to predicting zero change.
This is the Heun128 diagnostic; the registered Heun2048 decision remains pending.
The old four-lambda caches also reproduce: analytic scores rise to 0.998/0.951
at lambda 0.125, without a fitted amplitude or another sampling run.
[Complete diagnostic data](docs/research-assessments/2026-11-16-ode-mechanism/analytic-target-128-diagnostic.json),
[cached continuation](docs/research-assessments/2026-11-16-ode-mechanism/analytic-target-cached-continuation.json).

Actual-endpoint geometry now independently reconstructs all16banks and both
pools. The top10analytic particle residuals hold62.173%/95.167%of squared
error but12.274%/22.079%of absolute target-gradient contributions. The actual
projected-displacement quadratic is6.947e-3/6.778e-3; optimal one-dimensional
rematching cancels98.841%/99.056%of it. Remaining nonlinear terms are
8.053e-5/6.396e-5. These exact decompositions explain why particle and scalar
forecast errors differ without asserting an ODE-specific performance gain.
Independent dense-array reconstruction agrees within3.34e-16and full-gradient
ADwithin1.44e-20. All original flags remain unchanged.
[Cached geometry](docs/research-assessments/2026-11-16-ode-mechanism/analytic-target-endpoint-geometry.json).

Original bank numerical gates pass 15/16. Confirmation bank 0 retains its failed
epsilon 1e-4 metric check. Independent AD agrees with the derivative within
5.43e-19; the exact sorting correction reconstructs that finite-difference
error. Smaller four epsilons pass. The original flag remains unchanged.
[Cached diagnosis](docs/research-assessments/2026-11-16-ode-mechanism/analytic-target-retained-metric-failure.json).

145 controller 1477869 continues the complete 128-task cohort on physical 4-7.
The first four primary Heun2048banks and independent verifiers finish0and
qualify numerically; the current four workers run selection banks4-7. These
four completed banks are not a pooled quality decision.
[First fine-mesh controls](docs/research-assessments/2026-11-16-ode-mechanism/analytic-target-first2048-banks.json).
Mac CPU controller 15486 follows
completed banks with four one-thread all-source reference workers. Both
queues are monitored. The first four all-source banks and their independent
verifiers have finished; all four qualify. The complete all-source cohort
and primary2048prediction remain open.
[First four completed controls](docs/research-assessments/2026-11-16-ode-mechanism/trajectory-reference-readiness/first-completed-banks.json).

## Ordinary controls on the same current banks

The current-bank ordinary-control evaluation is also active on MacCPU.
It reuses four original20kseed0checkpoints: Ring8FreeISRU/SwiGLU64and
Product16FreeISRU/SwiGLU192, all original optimizer/capacity settings retained.
The same current sources/targets/directions are evaluated at128and2048steps;
older-bank ordinary scores are not used to rank this cohort. Fresh exact five
commands pass, historical anchors replay exactly, and independent witness
metric errors are<=4.45e-16. Controller19767runs two one-thread workers
19768/19769beside the four reference workers. Full comparison is pending.
[Protocol](docs/research-assessments/2026-11-16-ode-mechanism/ordinary-population-protocol.md),
[fresh check](docs/research-assessments/2026-11-16-ode-mechanism/ordinary-population-fresh.json),
[launch](docs/research-assessments/2026-11-16-ode-mechanism/ordinary-population-launch.json).

A frozen cached postprocessor now follows the original ordinary controller.
It waits for all four producers and independent verifiers, then joins the48
complete model/stream/mesh rows and32primary paired contrasts. It creates no
partial report or figure. The separate recursive verifier checks source joins,
bootstrap intervals and exact sign/Holm arithmetic. Full analysis and visual
review remain pending.
[Analysis follower](docs/research-assessments/2026-11-16-ode-mechanism/ordinary-comparison-launch.json).

## The analytic ODE correction predicts the complete field change

All four frozen models and both 4,096-input field streams are complete. Both
primary Product16 RK4 streams pass the unfitted prediction criterion. The
correction uses both solver defects and the unchanged full network.

| Weights | Selection score | Confirmation score |
|---|---:|---:|
| Ring8 midpoint |0.982|0.982|
| Ring8 RK4 |0.983|0.982|
| Product16 midpoint |0.957|0.957|
| Product16 RK4 |0.962|0.962|

Score compares squared prediction error with predicting zero change. Product16
RK4 relative RMS error is 0.196 and alignment 0.987 on both streams. Its
oracle-field MSE increase is also predicted, although generation improves.
All eight refined numerical cells qualify; original coarser-reference flags
remain unchanged. Producer/NumPy exits 0/0, independent error <=2.17e-18.
[Full results and two figures](docs/research-assessments/2026-11-16-ode-mechanism/learned-defect-field-results/analysis-report.md),
[source/protection audit](docs/research-assessments/2026-11-16-ode-mechanism/learned-defect-field-results/audit.json).

## The generation benefit survives outer-resolution changes

Four weights, two streams and all Heun127/128/129/512/2048 meshes complete.
All 720 metrics and 32 adjusted contrasts independently reproduce. Product16
RK4 weights improve with midpoint inference on every prescribed mesh.

| At Heun2048 | Native SW2 | Midpoint SW2 | Banks improving |
|---|---:|---:|---:|
| Selection |0.053|0.049|8/8|
| Confirmation |0.048|0.045|8/8|

The registered finer-mesh criterion passes. This is a same-weight solver effect;
ordinary architecture rankings require controls on these same banks.
[Complete generation results](docs/research-assessments/2026-11-16-ode-mechanism/population-resolution-results/analysis-report.md).

## Active: analytic correction through the sampler and target loss

The remaining link is the distributional response predicted from the analytic
correction, rather than the already measured field difference. The original
eager256-step observer took median7.232s per4,096 inputs, projecting about132
GPUhours for primary Heun2048 transport. The validated faster implementation
is now executing that study on145; this is no longer an unqueued projection.
Original compilation gives1.240x. Exact algebraic factorization and power
reuse now pass all8x4096input comparisons and independent reconstruction,
with4.475xresident speedup. Both fresh documented invocations exit0and
all5,213source hashes remain unchanged. FP64,256steps, every bank and the
scientific equations stay fixed. Selected observer projection:28.008GPUhours
for the primary test, excluding the remaining native sampler overhead.
[Measured execution comparison](docs/research-assessments/2026-11-16-ode-mechanism/analytic-runtime-results/analysis-report.md).

The joint analytic/measured sampler kernel passes all eight learned CPUprefix
checks and linear8/128/2048controls. Both fresh documented GPUentrypoints and
independent reconstructions pass, all5,229frozen source hashes unchanged.
The monitored full-bank queue is active on145physical4-7, controller
1477869, sourcefd04994. Primary128banks are complete;2048banks precede
all other models. All128bank/mesh tasks and independent outputs remain
required. The full-source CPUfollower is source46a9f7a/5244files, with a
fresh five-command pass and unchanged packages. No final primary decision yet.
[Launch](docs/research-assessments/2026-11-16-ode-mechanism/analytic-target-launch.json),
[fresh readiness](docs/research-assessments/2026-11-16-ode-mechanism/analytic-target-fresh-summary.json),
[registered all-source trajectory follow-up](docs/research-assessments/2026-11-16-ode-mechanism/analytic-target-full-probe-control.md).
[Registered runtime preparation](docs/research-assessments/2026-11-16-ode-mechanism/analytic-transport-runtime-protocol.md).

## Runtime and preserved artifacts

Full field52369/NumPy7963 and analyses90318/56920 terminate0. Source055a860,
5,175 hashes unchanged; physicalGPU4 released after186.177s. Population29874
and all eight children terminate0; its failed collector analysis and cached
repair remain preserved. No original evaluation restarted on a timeout.

Mac04f11f75 and14541450454 environments unchanged. Only empty physical145
GPUs4-7;0-3prohibited, no foreign job touched. Reports1/2(97protected files,
11final bindings) and five native image files unchanged. CelebA160 FID50k
remains native24.304/K1 27.789/K8 28.083. No new image architecture promoted.
