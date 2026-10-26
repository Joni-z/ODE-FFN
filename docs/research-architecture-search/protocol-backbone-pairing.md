# Shared-backbone initialization control

Registered before its formal training. This is a narrowly scoped comparison
control, not a new ODE architecture or continuation of failed additive settings.
No architecture passes merely because a newly initialized baseline weakens.

## Problem and intervention

Stage15 mul/add ODE pairs have identical surrounding initial tensors. However,
constructing different FFNs consumes different random draws before the FiLM
condition layers, so their three condition weights/biases differ from SwiGLU.
The same backbone architecture and seed therefore did not give literally
identical surrounding initialization. The zero initial output head hid that
difference in a simple initial-output comparison.

Rerun **only** the two SwiGLU sizes, leaving each one's native FFN tensors and
all RNG states unchanged. After constructing a model and before constructing
AdamW, copy every non-FFN state tensor and non-FFN runtime buffer from the exact
Stage15 vector_mul step0 checkpoint for that dataset. The saved initialization
reference path and SHA256 are explicit in the plan. The helper verifies all
names and shapes before copying; it includes nonpersistent Fourier frequencies.
No optimizer state, FFN tensor, generator state, final checkpoint or label is
copied. This intervention is opt-in; absent configuration preserves defaults.

## Fixed cells and training

Six cells: small SwiGLU (`ffn_hidden=96`, actual hidden64) and full SwiGLU
(`ffn_hidden=128`, actual hidden85), on Gaussian2D, Ring8 and rotated product16.
Raw labels remain `swiglu_budget` and `swiglu`. Width64/depth3, training seed0,
5000 updates, batch512, AdamW3e-4, weight decay0, clipping1, original cached-bank
pair labels, dataset/version hashes, validation streams, five fixed-time field
diagnostics and Heun32 are unchanged. Both plans run on the Mac mini CPU with
two one-thread workers and 10-second process/memory/progress monitoring. No GPU
is reserved. No rate, temperature, identity, normalization or loss change is
combined with this initialization control.

The runner propagates each arm's explicit dataset-to-reference map into the
generated trainer config. Returned source hash, copied key lists and maximum
before-copy difference are recorded in each run's provenance. Resuming restores
the complete training checkpoint after initialization, under the same config;
the reference must still satisfy its recorded hash.

## Comparison scope and decisions

Keep the original Stage15 baseline results visible. Compare these newly paired
baselines with the immutable Stage15 scalar/vector mul/add ODE outcomes, whose
shared surrounding initial tensors already match the chosen reference. Those
ODE outcomes are reused, not repeated independent results. Reuse is justified
only after source-delta inspection, exact reference-initialization checks and
bit-identical default three-update trajectory controls. No existing ODE source,
configuration, checkpoint or reported endpoint is modified.

Retain the existing practical screen: a candidate must improve oracle MSE by at
least10% against both practical baseline sizes on a nonlinear task and improve
its paired parent there, stay within5% of the better baseline on other primary
tasks, and keep full Jacobian error and SW2 within10% of the better baseline
on both nonlinear tasks. This control does not itself nominate a new candidate
or authorize extending a previously stopped setting. If rankings change, the
conclusion is that surrounding initialization contributed to the old comparison;
any useful-architecture claim still requires a separately chosen matched-budget
confirmation, actual quality/compute evidence and a fresh holdout. If the gap
persists, that strengthens the case for diagnosing learned features/controller
efficiency. Do not silently replace the older baselines with whichever result is
more favorable to ODE.

## Preflight and evidence

Freeze a new runtime by copying Stage15's source and explicitly overlaying only
the initialization helper, opt-in trainer/runner changes, tests, new plans and
this protocol. Record every source file hash plus the base-to-overlay delta.
Do not edit any previous runtime. Before formal launch require:

- The three helper tests, including atomic invalid-source rejection and an
  identical three-step clipped-Adam trajectory when copying a model to itself.
- Eight default/reference model trajectories unchanged relative to frozen
  Stage15, including nonzero output heads and three optimizer updates.
- All six intended SwiGLU initializations match every shared tensor/buffer of
  their recorded ODE reference, while native FFN tensors and RNG stay unchanged.
- A full opt-in trainer/checkpoint/resume smoke check, separate from formal
  experiment outputs, with exact final states and provenance checks.
- A real monitor heartbeat written by the frozen runtime using psutil, not
  merely an import check. Report any coverage failure instead of inferring it.
- All frozen source and initialization-reference hashes remain unchanged after
  preflight. Formal output directories must not exist before launch.

After completion, audit all saved checkpoints and common-field arrays, retain
full curves and original/new baselines, and independently replay endpoints.
Sampling or new-holdout verification must be named separately from a saved-field
replay. Report1 and the existing research goal remain unchanged.
