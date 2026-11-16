# ODE mechanism investigation

Goal active: identify which ODE constraints help generative modeling and why.
Seed0only. Mac mini is available. On145, physical0-3are prohibited and4-7may
be used only without compute PIDs and below500MiB. Never stop/share foreign
jobs. Reports1/2and native image model/trainer/loss/sampler remain fixed.
Publish verified changes to GitHubmain under Zhizhe Zhang.

## Completed evidence

- [x] Coefficient release, evolving state flow, operator sharing, optimizer
  coordinates, qualified symmetry/density and their independent numerical checks.
- [x] Four20kmatched P/Bschedule trajectories and complete common-bank sampling.
  All registered inner/outer refinements preserve Ring8PALlead/Product16ALTlead.
- [x] Exact Gaussian-source target Jacobians and wrong-object/sampling controls;
  ten native oracle task/time cells pass. Four-paper targeted primary review.
- [x] Twelve frozen checkpoints/60time slices on128selection pairs; restored
  FP32anchors, FP64JVPs/batch separation and independent NumPy checks pass.
- [x] Independent confirmation on all8x4096pairs/five times/four focal models:
  655360evaluated interpolation states; both native and both verifier exits0.
  Every source/target array equals its original bound NPZ. NumPy error1.11e-16.
  Exhaustive bootstrap intervals agree with a second enumeration exactly.

## Findings that decide the next action

| All independent pairs | PAL J error | ALT J error | PAL SW2 | ALT SW2 |
|---|---:|---:|---:|---:|
| Ring8 |0.156|0.191|0.086|0.108|
| Product16 |0.261|0.236|0.072|0.066|

Eight/eight banks support each focal direction. Adjusted conditional input
intervals exclude zero. This confirms an output-space association. It does
not establish its cause or ODE-specific value. Ordinary Product16SwiGLU
samples better despite greater averageJerror. Hidden time/globalFMtime remain
unconnected by a demonstrated useful constraint.

## Next concrete actions

1. [ ] Implement read-only oracle-path field-error sensitivity, following
   target-geometry/trajectory-protocol.md. Keep all twelve existing checkpoints.
2. [ ] Check independent exact Gaussian examples, native endpoint identity and
   Heun128/256qualification. Fix executable numerical tolerances before pilot;
   freeze/publish and obtain a fresh documented witness. Reuse warm CPUenv.
3. [ ] Run the bounded256-source/eight-bank calibration at lambda.125/.25/.5/1.
   Report the full-scale explained fraction for every model; no favorable-time
   or model selection. No new training/image run is included.
4. [ ] Only if the approximation is useful at learned error scale, register
   its all-source distribution-level test. Otherwise reject that approximation
   as an architecture guide. Keep the broader research goal active.

## Authoritative state

Source: /Users/mr.z/ODE-FFN-upstream. Published selection freeze49f6e19and
confirmation freeze4e96911remain unchanged. Raw:
/Users/mr.z/ode-workspace-prep/coefficient-20261116/target-geometry.
Both confirmation PIDs60825/60826ended. The selection PID59854ended.
No scientific observation or new training remains live from this phase.
Warm Python: /Users/mr.z/ode-workspace-prep/.venv-test/bin/python.
No packages changed and noGPUwas needed. Report checks97files/11final bindings pass.

Canonical results: docs/research-assessments/2026-11-16-ode-mechanism/
  target-geometry/{README.md,derivation.md,results/,confirmation-results/}.
Earlier complete mechanism findings: .../primary-findings.md.
Current goal turn: progress, completed theory/native checks and independent
all-bank output-response evidence. The broader objective is not complete.
Prior execution journals are archived under docs/history/.
