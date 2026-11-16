# ODE mechanism investigation

Goal active: identify what ODE structure contributes to generative modeling.
Seed 0 only. Mac mini may be used freely. On 145, physical GPUs 0-3 are
prohibited; 4-7 require <500 MiB and no compute PID. Never stop or share another
job. Preserve Reports 1/2 and native image model/trainer/loss/sampler. Publish
verified code and evidence to GitHub main as Zhizhe Zhang.

## Completed evidence

- [x] Coefficient/state-flow/sharing/optimizer-coordinate/symmetry/density controls.
- [x] Four matched P/B schedule trajectories and full registered numerical sweeps.
- [x] Exact Bayes target Jacobians; native oracle checks; targeted four-paper review.
- [x] Twelve-checkpoint spatial response observation and complete independent
  confirmation on 32,768 pairs per task/five times. The focal response ordering
  matches sampling in all eight input banks per task.
- [x] Native error-sensitivity observer: three exact Gaussian controls, fourteen
  oracle JVP cells and twelve real discrete-tangent/checkpoint witnesses pass.
- [x] Full 256-source/twelve-checkpoint calibration at four error scales and all
  registered 128/256/512/1024 levels. Both tasks and verifiers exit 0. None meets
  every endpoint/tangent precision gate; the scientific explanation is unresolved.
- [x] Temporal-alias construction on all twelve models, fresh exits 0/0 and
  independent NumPy reconstruction. Native grid/endpoints nearly unchanged;
  off-grid velocities and instantaneous Jacobian scores change. All tensors restored.

## Findings that decide the next action

The synthetic time embedding reaches 1000.000122 cycles/unit t. Shifting its
highest feature by -1024 leaves exact Heun stage values unchanged whenever
N divides 1024. Measured FP64 endpoint differences <=1.65e-13; FP32 <=3.02e-5.
Off-grid changes are 0.059-0.154 of oracle-fitting RMS error. This exposes a
blind spot in the old dyadic checks; it does not show that aliasing causes every
precision failure. The focal Jacobian ordering does not reverse.

Keep original fixed-grid SW2 values and the independent spatial association.
Do not claim that continuous error transport explains generation before its
numerics qualify. The ordinary Product16 SwiGLU remains stronger, so no new
architecture or image promotion follows this phase.

## Next concrete actions

1. [ ] Implement native-time-resolution-protocol.md: all twelve checkpoints,
   same 256 sources, native Heun 1000/1024/1032 for grid sensitivity, then
   2048/4096 and only if needed 8192 for componentwise qualification.
2. [ ] Validate any batched four-blend execution against every existing separate
   native evaluation before using it. Preserve the original unresolved pilot.
3. [ ] Freeze/publish source, run a fresh documented witness, measure throughput
   and dispatch with explicit progress/PIDs. Recheck 145 before GPU use; latest
   check found 4-7 occupied. Warm Mac CPU remains available.
4. [ ] Resolve oracle-tangent/blended-flow precision and return to the registered
   full-scale explained fraction. Then decide which mechanism is supported.
   Keep this phase read only; a failed numerical gate is not evidence for a new architecture.

## Authoritative state

Source: /Users/mr.z/ODE-FFN-upstream. Runtime freezes 4eeec25 (calibration) and
130a5ed (alias audit) remain unchanged. Raw:
/Users/mr.z/ode-workspace-prep/coefficient-20261116/trajectory-response.
Both pilot PIDs 63336/63337 and all documented check/verification commands are
terminal; no scientific job from this phase remains running. Warm Python:
/Users/mr.z/ode-workspace-prep/.venv-test/bin/python. No packages or GPUs used.

Canonical docs: docs/research-assessments/2026-11-16-ode-mechanism/target-geometry/
  {trajectory-results/,time-alias-results/,native-time-resolution-protocol.md}.
Earlier independent response evidence remains in results/ and confirmation-results/.
Current goal turn: progress, completed calibration and a verified source-level
sampling counterexample. The broader objective remains active and incomplete.
Prior execution plans/errors are archived verbatim under docs/history/.
