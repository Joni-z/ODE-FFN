# Frozen external-solver diagnostic, registered before execution

Question: Product16 raw-D improves field error over learned PD at some rates but
has worse sampled distance. Does coarse integration explain that discrepancy?
Do not change training or replace the primary Heun32 result.

Use the same six final20k checkpoints selected for the precision case: rawD
Ring8Muon.02/Product16Muon.006; strongest tested scalar Ring8Muon.006/Product16Muon.02;
compute-matched SwiGLU128 Ring8Muon.02/Product16AdamW.0003. These choices use prior
validation, not the new sampling readout. All checkpoints/source/data stay frozen.

For each case generate from the identical4096-point fixed test source using
Heun32/64/128 (64/128/256NFE), keeping targets and256 projection directions fixed.
Heun32 must reproduce its saved sample arrays and independent NumPy SW2 before
the other resolutions are interpreted. Record all distances, reference-versus-
reference floor, source/checkpoint hashes and synchronized sampling time.128 steps
are not a theoretical ground truth or a convergence certificate. No outcome-based
selection of checkpoint, sampler or inference budget is permitted.

Also bin the unchanged fixed validation-field errors into ten equal time bins,
using the original random-time bank/oracle. This is a descriptive error profile,
not a new evaluation target. It can identify where the average error difference
occurs; it cannot alone establish a trajectory-shift or integration mechanism.

This is read-only inference after cycle3 training frees the GPU, not a new
architecture family. If resolution changes the ordering, report the NFE tradeoff;
if it does not, stop blaming a coarse solver without further evidence.
