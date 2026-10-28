# Training-time clock and parameterization controls

## Goal
Test whether the 20k K1 image advantage survives training without input-dependent
clocks, while separating a constant-P representation from a direct SiLU
representation with the same initial function. This is a control study before
nominating another architecture.

## Actions
- [x] Read the completed time-clock experiment, native trainers, initialization
  definitions, earlier identity-location stop and compute state. Previous goal
  turn is progress: quality study seal 8917402e, public main 8cbc07f.
- [ ] Fix the two cold controls, one matched-inference reference, mathematics,
  stream/checkpoint/sample budgets and decision rules; commit runnable source.
- [ ] Pass local identities and a fresh documented preflight: active full-model
  gradients, native replay, three-step physical resume, cost and metric witness.
- [ ] Run exactly two seed 0 cold 20k training trajectories on the historical CPU
  stream, preserving P/effective-W histories and boundary-update witnesses.
- [ ] Audit training boundaries/streams, generate all seven prescribed banks,
  independently replay predictions/samples/features/FIDs and retain all scores.
- [ ] Plot the complete comparison, write report/limits, verify preservation,
  seal and publish. No automatic calibration, optimizer or constant rescue.

## Decisions
Use the warm Mac mini CPU stack to retain comparability with the native
baselines. Lenovo SSH probe 38914 exits 255 (timeout). Available RAM 4.75 GiB and
disk 55.79 GiB cover the bounded two-worker cohort; no GPU reservation/install.
The earlier synthetic identity-location factorial remains stopped. New image
initialization/coordinate controls are ordinary comparators, not its rescue.

## Status
Protocol and scoped [derivation](theory.md) precede all new fitted outcomes.
Runtime: `/Users/mr.z/ode-workspace-prep/autoresearch/oct28-image-clock-training`.
