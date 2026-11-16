# Frozen output-Jacobian diagnostic

Registered before execution. No training, hidden-resolution change, new response,
image run or density trajectory is part of this observation.

## Question and decision

Does full velocity-Jacobian error explain the opposite matched stage-order
sampling effects? Primary: average ||J_v-J_v*||_F^2/d, equal weight to each
point/time. The explanation predicts palindrome < alternating on Ring8 and
palindrome > alternating on Product16. Either contrary direction rejects this
two-task explanation at the registered budget. Agreement supports an association
to test causally; it does not show an ODE-specific benefit.

## Fixed population and states

Use all twelve 20k checkpoints in the existing schedule-order anchors.json:
shared P, anchored stages, Free ISRU, SwiGLU, palindrome and alternating on
each of Ring8/Product16. No model or checkpoint selection follows this diagnostic.
Both focal schedules keep identical optimizer coordinates and training recipes.
Ordinary models retain their recorded selected settings; cross-family observations
are descriptive, while the matched pair is the primary mechanism comparison.

Use the first128 source_0/target_0 pairs from each existing **selection** NPZ.
At each fixed t=.1,.3,.5,.7,.9, z=(1-t)source+t target, yielding640 states per
checkpoint. These are independent-path interpolation states, not generated
trajectories. Existing confirmation pooled SW2 is contextual, not recalculated
or used to select a geometry metric. Sample banks do not create training replicates.

## Measurements

- Primary full velocity-Jacobian error as defined above, time-resolved and averaged.
- Full field MSE and mean-bias energy on those same states.
- J_m=I+(1-t)J_v; corresponding target error, skew energy and negative-eigenvalue
  energy/fraction of sym(J_m). Do not call a nonsymmetric learned J_m PSD.
- Native FP32 prediction against saved common-bank field outputs, FP64 conversion
  against FP32, central finite-difference JVP on four states at t=.5, and input
  batch-separability. Preserve all parameters/runtime buffers during observation.
- Independently recompute summaries from raw arrays with NumPy; retain arrays
  outside Git, publish their hashes and complete summaries. No significance claim
  over the twelve heterogeneous checkpoints or the five repeated time slices.

For the two focal primary gaps, estimate input-sampling uncertainty by resampling
the128paired source/target indices, keeping all five times together. Use10000
bootstrap draws with analysis RNG0; report paired SD, percentile95% intervals
and97.5% intervals for the two-contrast Bonferroni family. These are conditional
on the fixed checkpoints and interpolation grid, not training uncertainty.
Keep the registered point-direction decision; intervals describe its precision.

## Required checks and resources

Run oracle_check.py first: native NumPy velocities, independent posterior moments,
unchanged Torch oracle Jacobians and central differences at ten task/time cells.
Include explicit nonunit-noise algebra and analytic counterexamples showing both
the wrong-object and sampling-certificate failures. Root JiT code is read only.

Warm Mac mini CPU environment at response-screen-cpu@04f11f75, one thread,
CUDA_VISIBLE_DEVICES=-1. These small full-Jacobian observations need no GPU.
Physical145 GPUs0-3 remain prohibited; no foreign task may be stopped or shared.

Execution documentation and a fresh verbatim witness must pass before the full
observation. Record source/checkpoint/data hashes, process identity, progress,
terminal status, all twelve cases and the exact numerical tolerances.
