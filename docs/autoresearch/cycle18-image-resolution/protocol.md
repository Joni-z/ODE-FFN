# Frozen image sampling: a numerical-confound check

## Question and scope

Cycle17 has lower ODE FID but worse paired validation fitting. Before treating
that ordering as evidence about the continuous learned field, test whether it
persists under a finer discretization of the **same** field. This is a frozen-
checkpoint diagnostic. It changes no architecture or training, and cannot
reverse cycle17's failed fitting and order gates. No solver is selected for
production, and no new architecture is nominated.

## Fixed cohort

All four cycle17 final-5,000-update models: ode_k1, ode_k2, swiglu86, swiglu129.
Use every one of the original 1,024 Gaussian starts, in the same order and
64-image chunks. Restore all native parameters and runtime buffers. Parent:
`/Users/mr.z/ode-workspace-prep/autoresearch/oct28-cycle17-v2`, seal SHA256
`62d178c4e2b88780495c9ea45c6aeef96453ce7098a8aeee84076cf58f081172`.
No best-checkpoint, image, time, seed, model or metric selection. Model seed 0
was used for these existing checkpoints; this study makes no optimizer update.

## Only intervention

Uniform explicit Heun on [0,1], with N = 32, 64, 128 steps, exactly 2N model
calls per image. Velocity remains `(D(x,t)-x)/max(1-t,.05)`. CPU float32, one
thread per child, at most two sampler children. No clipping of sampler states,
noise injection, adaptive mesh or endpoint correction. All N=32 outputs must
match their sealed parent outputs bitwise before accepting finer-grid data.

Use the parent training environment (spec475428fb) and metrics environment
(specdc729a50), unchanged. Existing seeded kernel/doc-pass receipts apply to
these unchanged environments and the same model/extractor; no dependency is
installed or rebuilt. Feature extraction remains MPS, 32-image chunks, the
same Inception weights and uint8 conversion. Reference features are exactly
the same 2,048 validation images. Verify source/weight/bank bindings and every
parent-seal entry before and after. Preserve all outputs, even on failure.

Private runtime: `/Users/mr.z/ode-workspace-prep/autoresearch/oct28-cycle18`.
Ceilings: 1,200 seconds per sampler arm; 600 seconds for each feature/audit
phase; no automatic retry. At most 1 GB extra disk expected. A stopped or
failed phase is not a completed diagnostic. Record exact commands/PIDs/exits
and 5-second process CPU/RSS telemetry; poll the same handle on timeouts.

## Outputs and interpretation fixed before finer-grid outcomes

Report all 12 FIDs (1,024 generated / 2,048 reference) and their unchanged
feature-mean/covariance accounting, finite checks and raw out-of-range rates.
Report raw-pixel RMS/max differences for 32-to-64, 64-to-128 and 32-to-128,
and the ratio of the first two RMS changes where defined. N=128 is a finer
reference, not ground truth. Do not assert order two from only this ratio.

Preserve each original ODE-versus-each-SwiGLU FID contrast at every N.
A sign reversal is direct evidence that the observed ranking depends on
the tested solver resolution. If every original contrast stays negative,
32-step discretization alone does not explain the ordering on these tested
grids; this does not establish convergence, field accuracy or a new method.
Also report K2-versus-K1 at all N without changing the failed order gate.
Show the first eight starts for every model/grid, without visual selection.
No significance tests or uncertainty over independent trained repetitions.

## Required checks and stopping rule

Before fitted finer-grid outputs: a zero-clean-predictor affine fixture against
its scalar Heun recurrence, a constant-velocity fixture, and exact all-source
N=32 native replay. Commit the source/protocol before execution. Independently
replay every finer-grid output, check feature identity on the same extractor
and recompute all FIDs using SciPy gesvd instead of the producer NumPy SVD.
Raw sampler replay must be bitwise. Feature replay must be bitwise on this
same deterministic MPS backend; FID differences must be <1e-8. No tolerance
changes based on fitted outcomes. Preserve the parent seal unchanged.

This diagnostic ends after its registered grids. No extra steps, new solver,
clipping, longer training or metric change is added to obtain a favorable
result. All older stopped architecture families remain stopped.
