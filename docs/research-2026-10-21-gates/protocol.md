# Does the hidden-state-dependent duration help?

Registered before the new training runs, 21 October 2026.

The frozen-order experiment showed that trained low-order layers depend on their
polynomial form. It did not establish whether their input-dependent duration is
useful. We now keep K=2 and change only how the duration is generated.

## Comparisons

Each of Gaussian anisotropic, Ring8 and Two Moons receives five runs:
the original input gate, fixed s=0.6, one learned scalar per ODELayer, SiLU and
SwiGLU. All use training seed 0, 5,000 updates, batch 512, AdamW 0.0003,
zero weight decay and gradient clipping at 1. Width, depth, data splits and
external time conditioning remain unchanged.

The learned scalar is s=0.2+0.8*sigmoid(a/10), initialized with a=0.
The original input gate keeps its original initialization, which is near, but
not exactly, 0.6. The fixed and scalar models have fewer parameters and MACs;
we report the actual counts rather than call these parameter-matched controls.
Common weights, including every P, have exactly the same initialization across
the three ODE variants. Initialization consumes the original gate's RNG draws
before discarding that gate; no unused trainable gate is retained.

The original input gate and both baselines are rerun in the new environment.
Earlier test results are context, not extra replicates.

## Outcomes and endpoint rule

Primary endpoint: step 5,000, fixed in advance. Secondary endpoint: minimum
validation SW2 among steps 1,000, 2,000, 3,000, 4,000 and 5,000, breaking ties
in favor of the earlier step. The same rule applies to all five architectures.
Training-time sample evaluation uses validation targets and independent fixed
source draws. Test targets are not used for checkpoint selection. Evaluate test
SW2 only after all training is complete, at the primary and selected checkpoints.
Report both, even when they disagree.

SW2 uses 2,048 samples, 128 directions and Heun32. Pair regression and analytic
oracle errors (where available) are complementary outcomes. Save checkpoints
every 1,000 steps, with P spectra and gate distributions along each trajectory.
At endpoints decompose the hidden-layer Jacobian into its polynomial and gate
terms. Probes and layers are within-run measurements, not independent runs.

## Interpretation

If fixed or scalar durations preserve sample quality, the input gate is not
necessary on these targets at this budget. A scalar improvement over fixed
reflects optimization or parameterization, since a constant duration can be
absorbed into unconstrained P. It cannot demonstrate greater expressivity.
An input-gate improvement over scalar suggests
input adaptation may matter, but gate-network capacity and initialization remain
alternative explanations. Such a result motivates a separate capacity control,
not an immediate claim about flow matching or image generation.

## Execution

Use one GPU when it can be held exclusively. Benchmark real training updates with one and two
workers after warmup. Use two workers only if aggregate throughput improves by
at least 10%. Record GPU utilization, memory and per-run progress every 10 seconds.
Freeze source/config/data hashes before execution. Refuse to overwrite unfinished
or completed runs. Do not change constraints on P or add regularizers this round.

### Hardware amendment before full training

GPU0 on 145 was reclaimed twice with the owner's authorized helper, but the
owner's controller restarted its workers both times. Our resource guard stopped
the second attempt during GPU smoke checks. No full training trajectory or
benchmark was completed on the GPU.

CPU execution is permitted if the real training-kernel benchmark makes this
small study practical. Use at most two concurrent single-threaded training
workers, chosen by the same 10% throughput criterion. Run all fifteen models on
the same backend; do not mix historical GPU results into the new comparison.
The same seed value does not imply identical CPU/CUDA random training batches.
The GPU smoke outputs are software checks and are excluded from research results.
