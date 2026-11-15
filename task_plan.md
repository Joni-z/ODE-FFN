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

## Next: population-density comparison (not launched)

- [ ] Implement formal entry with unchanged density_ops.py, all eight models
  and both fields, four 256-source banks, common 1024/2048 steps: 128 rows.
- [ ] Pass actual CPU entries, fresh documented CUDA readiness and a timing-only
  256-source benchmark before choosing host routing.
- [ ] Use only idle 145 physical GPUs 4-7 for CUDA, with exclusive UUID guards
  and live progress/PID/resource records. Retain gates on all full-source points.
- [ ] Independently reconstruct per-point ratios, time bins and resolution
  changes; compare paired density contrasts with Monte Carlo uncertainty.
- [ ] Only if the contrast supports it, register one time-local oracle
  intervention, select on bank 0 and confirm on banks 1-3. No response search.

Current finding: local field MSE and density-source RMS do not give the known
Product16 sampling ordering. Signed along-path density remains unmeasured
at population scale. No architecture or image run is promoted.

Canonical study: docs/research-assessments/2026-11-16-ode-mechanism/
Current: density-transport/analysis-report.md; density-formal-next.md.
Raw: /Users/mr.z/ode-workspace-prep/coefficient-20261116/density-transport
Pilot queue 32943 and refinement queue 33293 are terminal; all workers exited.
P1/P2 journals are in docs/history/. Final reports remain untouched.
Previous goal turn: progress. Current turn also completes new observations,
validates the exact observer and qualifies all sixteen pilot cells.
