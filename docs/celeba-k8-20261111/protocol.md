# CelebA ISRU-D K8 transfer

Authorized 11 November 2026. Compare the selected frozen-diagonal K8 FFN with
the native SwiGLU in the pinned author's JiT implementation. The hypothesis
is that repeated use of the input-dependent generator improves image modeling
at a smaller FFN parameter count. Report measured runtime as well as quality;
K8 has twice the native FFN's forward dense MACs at this model width.

## Fixed comparison

- Original aligned CelebA, official 162,770-image training split. Keep the
  validation/test splits separate. Resize/center crop to 64 with the author's
  `center_crop_arr`; training uses horizontal flips.
- Full JiT-B body: 12 blocks, width 768, 12 heads, bottleneck 128 and 32 context
  tokens. A patch-size-4 adapter gives 256 image tokens at 64 resolution.
- Candidate: two independent square P matrices around SiLU. Each layer fixes
  d0 = rsqrt(1 + 1.2 h0^2) at its entrance and evaluates
  sum(k=0..8) [P diag(d0)]^k h0/k!. Kaiming-uniform P, no added rate,
  normalization, spectrum penalty or loss term.
- Native SwiGLU control, unchanged objective, time/noise law, attention and
  two EMA equations. Both arms use AdamW (.9,.95), weight decay 0, absolute
  LR 1e-4, five warmup epochs, then constant. This is a declared CelebA port,
  not an author-published CelebA recipe.
- Physical batch 64, model seed 0 only. Unconditional label dropout 1 and CFG
  0, using the author's null-label training and sampling path.

## Initial budget and records

Target 200 epochs. This submission authorizes one **two-hour allocation per
arm**; stop cleanly at an epoch boundary before the allocation expires and
retain a resumable checkpoint. Do not treat a wall-time pause as completion
of 200 epochs or launch an unbounded continuation chain.

One GPU of any eligible type, four CPUs, 24 GB host memory. Use priority
account torch_pr_637_tandon_priority; let the scheduler assign partitions.
CUDA 12.8 / PyTorch 2.7.1 supports Blackwell as well as the older card types.
Preserve the earlier CUDA12.4 environment separately.

Save the complete model, optimizer, both EMAs and RNG at each completed epoch.
Retain separate P-only snapshots at initialization and every completed epoch
in `operators/`; overwriting checkpoint-last must not erase P trajectories.
At job start, extract the verified archive into a job-owned node-local `/tmp`
directory and copy the five metadata files, retaining the canonical data on
scratch. This keeps small-file I/O off the network training path. Remove only
that temporary cache at job exit.
Record epoch loss/time, progress and 30-second GPU
telemetry. Stop on nonfinite loss; investigate a failure before relaunching.

## Evaluation

Use the author's pinned torch-fidelity fork and deterministic 64-pixel center
crops of the **training split** as the FID reference. The original 50-interval
sampler uses 49 Heun intervals followed by final Euler. Retain evaluation
images and exact checkpoint identity. Formal comparisons require FID50k and
an ordinary baseline below 50. Initial training losses and short CUDA checks
measure progress and correctness; the final FID evaluation is a separate
checkpoint-bound stage. No new coefficient, order or optimizer sweep is
included in this submission.
