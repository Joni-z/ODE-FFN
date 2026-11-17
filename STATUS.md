# Current project status

## Report3 draft

The [complete Report3 draft](reports/report3/report3.pdf) is written:
three pages of main text and four supporting pages. It separates the
sharing/coefficient/solver controls, includes the professor-response evidence
and Ring8case examples, and retains the completed common-bank/image results.
Detailed tables and figures are outside the main text, following the professor's
readability feedback. [Source and evidence](reports/report3/README.md).
No experiment was launched for this report; Reports1/2remain unchanged.
Report3is not finalized.

18 November revision: the penultimate report now leads with the shared-iteration
hypothesis and ends with one decisive compute-matched comparison on Ring8 and
Product16: midpoint, tied residual iteration, Free ISRU and SwiGLU. The proposed
outcomes select midpoint, a simpler recurrence, or an ordinary FFN. No new run
is launched; numerical evidence and finalized Reports1/2remain unchanged.

## User-requested pause and handoff

17 November: the user requested closing the current code/evidence changes.
Autonomous development and new experiment decisions are paused. The user then
moved subsequent synthetic work to Mac mini.145dispatcher1477869is stopped
withSIGSTOP(stateTs): its104pending jobs cannot launch. The user then explicitly
canceled its four remaining workers. All four exited bySIGTERM(exit15); their
CUDAcontexts on physical4-7are released. Completed and partial outputs remain
intact. The canceled banks do not establish a completed fine-mesh analytic test.
General research resumption does not authorize resuming this145dispatcher.
[Resource-policy receipt](docs/research-assessments/2026-11-16-ode-mechanism/resource-policy-20261117.json).
[Cancellation receipt](docs/research-assessments/2026-11-16-ode-mechanism/user-canceled-four-workers-20261117.json).
Existing Macqueues remain running: all-source controller15486,
while ordinary-control controller19767and its cached-comparison follower20623
have now completed. All four ordinary producers/checks pass; the comparison
terminates0with source unchanged. Both generated figures have been reviewed.
The all-source reference cohort remains incomplete. No canceled145task was
resumed. [Report3 working outline](reports/report3/outline.md).

The [whole-project review and full experiment ledger](docs/research-assessments/2026-11-17-project-review/README.md)
now collect the legacy and later evidence. This read-only retrospective launches
no experiment and leaves finalized Reports1/2unchanged.

## Analytic target forecast: complete Heun128 diagnostic, Heun2048 incomplete

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

145completed20of128bank tasks; the next four were intentionally canceled.
Dispatcher1477869remains paused under the Mac-only synthetic policy.
The first four primary Heun2048banks and independent verifiers finish0and
qualify numerically; selection banks4-7were canceled. The four completed
fine-mesh banks are not a pooled quality decision.
[First fine-mesh controls](docs/research-assessments/2026-11-16-ode-mechanism/analytic-target-first2048-banks.json).
Mac CPU controller 15486 follows
completed banks with four one-thread all-source reference workers. Both
queues are monitored. The first four all-source banks and their independent
verifiers have finished; all four qualify. The complete all-source cohort
and primary2048prediction remain open.
[First four completed controls](docs/research-assessments/2026-11-16-ode-mechanism/trajectory-reference-readiness/first-completed-banks.json).

## Ordinary controls on the same current banks: complete

The current-bank ordinary-control evaluation completed on MacCPU.
It reuses four original20kseed0checkpoints: Ring8FreeISRU/SwiGLU64and
Product16FreeISRU/SwiGLU192, all original optimizer/capacity settings retained.
The same current sources/targets/directions are evaluated at128and2048steps;
older-bank ordinary scores are not used to rank this cohort. Fresh exact five
commands pass, historical anchors replay exactly, and independent witness
metric errors are<=4.45e-16. All four cases and independent checks complete.
The cached comparison independently reconstructs48rows/32primary contrasts,
maximum scalar discrepancy1.74e-18. The earlier
[job snapshot](docs/research-assessments/2026-11-17-project-review/job-status.json)
is a historical snapshot taken before the final two controls completed.
[Protocol](docs/research-assessments/2026-11-16-ode-mechanism/ordinary-population-protocol.md),
[fresh check](docs/research-assessments/2026-11-16-ode-mechanism/ordinary-population-fresh.json),
[launch](docs/research-assessments/2026-11-16-ode-mechanism/ordinary-population-launch.json).

A frozen cached postprocessor waited for all producers and verifiers, then
joined the complete model/stream/mesh rows and primary paired contrasts.
The separate recursive verifier checks source joins, bootstrap intervals and
exact sign/Holm arithmetic. Both figures are now visually checked.

| Heun2048, confirmation | Ring8 pooled SW2 | Product16 pooled SW2 |
|---|---:|---:|
| State flow, midpoint training/inference |0.068|0.047|
| Free ISRU / AdamW |0.091|0.051|
| SwiGLU64/Muon, SwiGLU192/AdamW |0.105|0.034|

All native/crossed solvers and both streams remain in the
[completed comparison](docs/research-assessments/2026-11-16-ode-mechanism/ordinary-population-results/analysis-report.md).
Ring8pooled ordering favorsODEin16/16primary comparisons;3adjusted upper
bank-bootstrap bounds are negative. Product16favorsODEin4/16,none with a
negative upper bound. State flow uses about4.1timesFreeISRUdense forward matrix
work; this is not a total training FLOPs comparison or a new image candidate.
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

## Partially executed analytic correction through the sampler and target loss

The remaining link is the distributional response predicted from the analytic
correction, rather than the already measured field difference. The original
eager256-step observer took median7.232s per4,096 inputs, projecting about132
GPUhours for primary Heun2048 transport. The validated faster implementation
partially executed that study on145. Its four remaining workers are canceled,
and subsequent GPUdispatch stays paused under the new resource policy.
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
The original monitored queue used145physical4-7, controller1477869,
sourcefd04994. Its dispatcher is now paused; the final four active workers
were subsequently canceled by the user. Primary128banks are complete;
four2048banks are complete.
All128bank/mesh tasks and independent outputs remain required, with a future
CPUcontinuation needed. The full-source CPUfollower is source46a9f7a/5244files, with a
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
