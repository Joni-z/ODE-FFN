# CelebA128: native JiT-B/16 and ISRU-D K8

Authorized 12 November 2026 after the [official-recipe audit](../research-assessments/2026-11-12-training-recipe/assessment.md).

Test whether the selected frozen-diagonal K8 FFN improves image generation
relative to native SwiGLU in an author-derived JiT backbone with a genuine
patch-input bottleneck. This is a new CelebA transfer recipe, not an ablation
isolating which difference in the earlier 64px setting mattered.

## Fixed method and controls

- Original aligned CelebA, official train split of 162,770 images. Apply the
  author's center crop to 128x128 and random horizontal flip. Validation/test
  images do not enter training or the FID reference.
- Author's native JiT-B/16: 12 blocks, width 768, 12 heads, bottleneck 128;
  64 image tokens and 32 context tokens inserted at block 4. Each RGB patch is
  768-dimensional before the 128-dimensional input projection.
- Native SwiGLU versus ISRU-D K8, model seed 0. K8 retains its two independent
  square operators around SiLU. Each entrance diagonal is
  `d0 = rsqrt(1 + 1.2*h0^2)` and remains fixed in
  `term = P(term*d0)/k`, for k=1 through 8, summing the entrance and all terms.
- Same author clean-image prediction, velocity loss, time law, sampler,
  initialization of shared components and per-update EMA mathematics.
  Unconditional class count 1, label drop 1, CFG 0. No Muon, REPA, spectral
  penalty, new normalization or learned clock.
- Proposed physical/global batch 256 on one GPU; author `blr=5e-5` gives actual
  LR 5e-5. GPU readiness measures both full models at that exact batch. If 256
  fails, register a common smaller batch and its scaled LR before submitting
  training. Do not change batch between allocations or independently by arm.
- Noise std .5 follows the paper's resolution rule extrapolated to 128, with
  logit-normal mean -.8/std .8 and denominator clip .05. This low-resolution
  extrapolation is not an author-validated CelebA optimum.
- AdamW (.9,.95), weight decay 0, warmup 5 epochs then constant; per-update EMA
  .9999/.9996. Keep the same decays in both arms; the averaging window in
  image counts differs from the ImageNet/global-batch 1024 setting.

## Budget and stopping

Revised by the user on 12 November: target **100 epochs per arm** in one
**six-hour allocation**, one GPU, six CPUs, four data workers and 24 GiB host
RAM on priority. Keep all six GPU types joined by OR and omit a partition.
The budget is about 63,500 optimizer updates. The published JiT 200/600-epoch
ImageNet schedules are context, not a validated CelebA stopping point.
See the [measured estimate and source comparison](budget-20261112.md).
The existing full-model readiness 22813320 passed; do not repeat it for a
duration/worker-count change. Its source and batch guard remains in place.

Use a clean detached source runtime; later development cannot modify it.
Both arms start from scratch in new 128px output directories. The old 64px
checkpoints/runtimes stay untouched. A fresh start requires an explicit flag
and is permitted only at restart count 0 with no prior source marker. Missing
checkpoint state from an existing run must never restart training.

Disable the planned runtime stop (`max_train_seconds=0`) and automatic
requeue. Save model, optimizer, both EMAs and RNG at every completed epoch,
and retain each P snapshot. Stop at 100 epochs or an execution failure. The
Slurm six-hour cap is an exceptional backstop; if it is reached, inspect
the last checkpoint and measured speed before a manual recovery. There is
no predetermined chain of short jobs.

## Evaluation and monitoring

Keep the author 50-interval Heun sampler, including its final Euler step, and
fixed evaluation noise/EMA settings across arms. Formal quality uses 50,000
generated images and deterministic 128px training-split crops in the pinned
torch-fidelity implementation. FID below 50 for the ordinary baseline is the
minimum gate before interpreting an FFN advantage. Training loss alone is
not that gate. No final quality metric has yet been produced for this recipe.

Progress/epoch timing/checkpoints and P trajectories are saved under
`/scratch/zz5070/ODE_FFN/runs/celeba/20261112-b16-{ffn}-s0/`.
Training records GPU utilization/memory/power every 30 seconds; readiness
records every 5 seconds. Collector startup failure stops the allocation.
The lightweight existing Slurm monitor polls job/progress state every 60 seconds
and never launches or restarts a job. It writes to
`/scratch/zz5070/ODE_FFN/operations/celeba128-20261112/monitor.json`.
