# ODE mechanism investigation

Goal active: identify which ODE constraints contribute and why. Training
seed 0 only; pinned native source and finalized reports fixed. On 145,
physical GPUs 0-3 are forbidden; 4-7 only when empty. Never stop/share a
foreign task. Mac mini is available freely.

## Completed

- [x] Eight 20k coefficient/state/stage/sharing trajectories, native witnesses,
  matched initialization/RNG/gradient controls and ordinary cost references.
- [x] All 25 earlier numerical/symmetry cases and 78 scores independently checked.
- [x] Sharing primary paired banks plus full P0 solver refinement.
- [x] P1 eight-model symmetry intervention: Ring8 reverses the sharing ranking,
  Product16 retains it. Full independent checks and five inspected figures.
- [x] P2 exact density-ratio derivation, native oracle derivatives, continuity,
  Gaussian examples, eight model observers and eight full-map finite differences.
- [x] Fresh documented CPU witnesses; exact unaltered native sampling endpoints.
- [x] Bind stream 2026112300 and four 256-source banks before measurement.
- [x] Complete 48 pilot and 10 targeted refinement integrations, frozen
  8691431 / 5864256. Independent verification errors <=2.94e-14.
- [x] Qualify all 16 pilot cells: six at 512 steps, ten at 1024. No 2048 pilot needed.
- [x] Inspect two diagnostic plots and preserve all frozen evidence.

## Current: population-density comparison (running)

- [x] Implement formal entry with unchanged density_ops.py, all eight models
  and both fields, four 256-source banks, common 1024/2048 steps: 128 rows.
- [x] Pass actual CPU entries, fresh documented CUDA readiness and a timing-only
  256-source benchmark before choosing host routing.
- [x] Launch on idle 145 physical GPUs 4-7, with exclusive UUID guards
  and live progress/PID/resource records. Retain gates on all full-source points.
- [ ] Independently reconstruct per-point ratios, time bins and resolution
  changes; compare paired density contrasts with Monte Carlo uncertainty.
- [ ] Prepare the fixed-weight statistics reader while integrations run: paired
  path bootstrap stratified by the four common source banks; retain failed
  numerical cells; expose time-bin discovery on bank 0 only.
- [ ] Only if the contrast supports it, register one time-local oracle
  intervention, select on bank 0 and confirm on banks 1-3. No response search.

Current finding: local field MSE and density-source RMS do not give the known
Product16 sampling ordering. The two completed Ring8 ordinary controls have independently reproducible
density observations, but all four model/field cells include resolution-sensitive
paths. Population density contrasts remain unqualified. No architecture or image run is promoted.

Canonical study: docs/research-assessments/2026-11-16-ode-mechanism/
Current: density-transport/analysis-report.md; density-formal-next.md.
Raw: /Users/mr.z/ode-workspace-prep/coefficient-20261116/density-transport
Pilot queue 32943 and refinement queue 33293 are terminal; all workers exited.
P1/P2 journals are in docs/history/. Final reports remain untouched.
Previous goal turn: progress. Current turn also completes new observations,
validates the exact observer and qualifies all sixteen pilot cells.

Formal-density execution started this turn. Previous goal turn is progress:
it produced new field observations, validated the exact observer and completed
all16pilot numerical qualifications. Current145GPUs4-7are occupied, so begin
CPUentry/throughput checks while preparing guarded CUDAdeployment. Keep the
full128-row population-density comparison and mechanism decision as the target.

Formal author CPUentry passes all eight derivative/reference cases and two
actual formal-entry witnesses. Benchmarks and full jobs are distinct phases.
Source/data/sampler/operator mathematics stay fixed. Next: publish/freeze,
fresh CPU/CUDAverbatim readiness, actual256-source benchmark and routing.

Execution-only amendment from actual256-source CPUtimings: projected Product16
state flow requires29.520sfor32steps; the full case extrapolates to220minutes.
Before formal launch, test exact batched reverse VJPs against the frozen
coordinate-Jacobian observer. Keep density_ops.py, every sample, step size and
architecture unchanged. An adapter is allowed only after all-model Jacobians,
complete observer arrays and native endpoints agree. Record both backend
speeds; retain the original if batching does not help. This is not a stochastic
trace approximation or a new model intervention.

Launch verified: Mac queue35441 / workers35443,35444; GPUqueue3447050 /
workers3447077-3447080on physical4-7. CPUcases2/3and GPUcases1,7,0,6,5,4
are disjoint and coverall8models. Routing is published beforelaunch. Mac
sourcef79f027; GPUsource3a7d5fe; producer file hashes match. Full rows need
independent numerical validation before interpretation. Do not edit the
running formal_density.py,density_ops.py,jacobian_backend.pyor dispatch core.

Latest numerical checkpoint: both Mac cases completed, queue35441 exit0.
All32rows independently reconstruct within5.33e-15. At the unchanged gates,
FreeISRU original/projected has58/23of1024failing source observations;
SwiGLU has25/11. Retain all points. Wait for the remaining GPUcases, then
register bounded refinement of unresolved paths before density ranking.
Do not treat pilot readiness as full-source numerical convergence.
