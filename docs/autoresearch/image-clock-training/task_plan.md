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
- [x] Fix the two cold controls, one matched-inference reference, mathematics,
  stream/checkpoint/sample budgets and decision rules; commit runnable source.
- [x] Pass local identities and a fresh documented preflight: active full-model
  gradients, native replay, three-step physical resume, cost and metric witness.
- [x] Run exactly two seed 0 cold 20k training trajectories on the historical CPU
  stream, preserving P/effective-W histories and boundary-update witnesses.
- [x] Audit training boundaries/streams, generate all seven prescribed banks,
  independently replay predictions/samples/features/FIDs and retain all scores.
- [x] Plot all registered data, inspect five figures/144fixed examples and
  write the full report, numerical appendix and figure catalog.
- [ ] Verify final public artifacts, seal the private runtime and publish the
  complete closure. No automatic calibration, optimizer or constant rescue.

## Acceptance work while the original controller runs
- [x] All seven1024-image banks complete; sampling phase exits0in793.22s.
  Independent sample replay and feature/FID checks remain pending.
- [x] Training-boundary audit exits0: all14saved checkpoints,42stream checks
  and12saved boundary updates pass; full40000-step replay is not claimed.
- [x] Re-poll actual session5632 and verify controller70719/create-time; the
  preceding authentication turn also confirmed direct GitHub publication.
- [x] Hash the actual external pixels, split and manifest against native17;
  all match and their metadata predates this training launch. The unused
  download tarball is not an input to `data_for` and is not a native binding.
- [x] Register the curator before new FIDs: all32 endpoint rows,14primary rows,
 26primary ratios,75time-panel scores,210matrix summaries and144fixed examples.
- [ ] Verify report numbers, plotted arrays, actual tool exits, external data,
  protected scopes and source/parent preservation before sealing the runtime.

## Decisions
Use the warm Mac mini CPU stack to retain comparability with the native
baselines. Lenovo SSH probe 38914 exits 255 (timeout). Available RAM 4.75 GiB and
disk 55.79 GiB cover the bounded two-worker cohort; no GPU reservation/install.
The earlier synthetic identity-location factorial remains stopped. New image
initialization/coordinate controls are ordinary comparators, not its rescue.

## Status
Source28be1e8published; fresh documented native47808/metric32097both terminate0. All four native prediction banks, full shared tensors including timestep MLP gradients, physical resumes, costs and cached features pass. All25cached FIDs also replay exactly; default-SiLU64shared initialization/buffers/data RNG match. Passed v1readiness retained under hashes; v2only strengthens witness coverage.

Both20ktrajectories finished with actual worker exits0; the training phase consumed1747.45seconds. Original controller70719/create1793185015.834589 and tool session5632 continue through independent sample replay after all seven sampling banks complete. Curator441a6d1is published and privately registered before new FIDs. Actual external pixels/split/manifest match native bindings. There is no new generative quality conclusion; sample/feature/FID audits and report/archive remain pending.

All six phases and original controller tool5632 terminate0. Every new sample/feature row and the original reference reproduce bitwise; all32FIDs independently agree. Proceed to the registered curator, visual review, complete report and archive.

Curator17551 exits0. All32comparisons/14primary rows,26ratios,75panels and210matrix summaries are retained. Every five figure and144fixed example has been viewed. Fixed06passes the original allocation/matched-cost criteria atFID174.708609; matched-init SiLU178.056840fails the margin while preserving strict ordering; frozen same-c native216.070195is attribution only. No new ODE-specific model is accepted. Final artifact/archive verification is next.
