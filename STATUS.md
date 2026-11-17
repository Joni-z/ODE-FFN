# Current project status

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
checks and linear8/128/2048controls. Main GPUentrypoint validation precedes
the monitored full-bank queue. No analytic target-quality result yet.
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
