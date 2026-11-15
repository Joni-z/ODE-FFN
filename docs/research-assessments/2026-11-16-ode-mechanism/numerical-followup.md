# Numerical and symmetry follow-up after six completed trajectories

The six new training trajectories and primary confirmation evaluations are
complete. Fixed/free Taylor coefficients are close; fixed/free midpoint
stages are also close. State feedback is the consistent positive comparison
within the ODE family. It still loses Product16 to ordinary controls.

## CPU resolution fallback

The145GPU6 witness passed eight portable model cases, then another task
appeared (PID2254862). Its occupancy guard stopped our process. No full GPU
evaluation was launched and no other task was stopped. The incomplete witness
does not pass its COMPLETE gate.

Use the unchanged Mac mini CPU environment and a frozen source worktree for
the remaining numerical checks. The portable cohort contains14CPU-trained
checkpoints with hashes, their original CPU predictions/samples, exact
configurations and confirmation assets. Two inference workers run:

- All14models: native Heun64/128/256/512 on the same512source/target points
  from bank0. Record scores and paired endpoint differences to512; these are
  numerical controls on a subset, not new full-bank quality rankings.
- The four state/free-stage models: native Heun128 with internal4and16 on
  all32768source/target samples. Record pooled and eight bank SW2s. Internal8
  and16 full-field and bounded-sample checks already exist. This tests
  whether the main effect survives near-continuous hidden dynamics.

Use the original sampler and metric implementation. The CPU queue is
`resolution_queue.py`; its18resolution tasks write separate COMPLETE.json artifacts and
a terminal queue receipt. Preserve all samples for independent NumPy checks.
The first subset reference invocation is tested before launch. No training,
parameter changes, new optimizer or new response is added.

## Ring8 symmetry explanation

For each of seven Ring8 architectures, evaluate256common field probes over
their full C8orbits (2048states). Pull the predicted vectors back into one
frame. Their mean is the C8projection. The finite-orbit squared error splits
exactly into invariant error and symmetry-breaking variance. Verify oracle
equivariance and the decomposition numerically.

Then compare the original and C8projected velocity fields under the native
Heun128 sampler, using the same512source/target points. Record SW2 and sector
counts and preserve the generated samples. This is a frozen-field intervention
with eight predictions per field call, not a proposed cheap replacement FFN.
If the state-flow gap disappears after projection, allocation/symmetry offers
a candidate explanation; if it persists, pure C8breaking is insufficient.
The512sample check is bounded; do not substitute it for the primary32768
sample comparison. Ordinary baselines receive the identical intervention.

The initial512sample intervention has a large finite-sample floor, so a
second paired protocol uses4096base source/target points and all eight C8
rotations, giving32768correlated points per side. Compare plain and projected
fields on exactly those same rotated inputs. The projected field is
equivariant and native Heun preserves linear rotation equivariance. Integrate
the4096base points through the projected field, then rotate their endpoints.
This uses the same neural batch work as plain inference on all32768points,
instead of eight times as much. A double-precision native trajectory witness
checks the identity before using it. The finite arrays are group-stratified,
not iid; their absolute scores must not be mixed with pooled-bank scores.

Seven full symmetry interventions join the two-worker CPU queue, for25total
tasks. This tests whether the state-flow ranking remains after removing C8
breaking without adding training or a new architecture.
