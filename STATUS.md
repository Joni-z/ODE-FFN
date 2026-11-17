# Current project status

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

Original bank numerical gates pass 15/16. Confirmation bank 0 retains its failed
epsilon 1e-4 metric check. Independent AD agrees with the derivative within
5.43e-19; the exact sorting correction reconstructs that finite-difference
error. Smaller four epsilons pass. The original flag remains unchanged.
[Cached diagnosis](docs/research-assessments/2026-11-16-ode-mechanism/analytic-target-retained-metric-failure.json).

145 controller 1477869 continues the complete 128-task cohort on physical 4-7;
the current four workers run primary Heun2048. Mac CPU controller 15486 follows
completed banks with four one-thread all-source reference workers. Both
queues are monitored; all-source qualification and the complete primary
prediction remain open.

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

## Next: analytic correction through the sampler and target loss

The remaining link is the distributional response predicted from the analytic
correction, rather than the already measured field difference. The eager
256-step observer takes median7.232s per4,096 inputs. Primary full Heun2048
transport projects to about132 GPUhours at that rate; it is not queued.
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
