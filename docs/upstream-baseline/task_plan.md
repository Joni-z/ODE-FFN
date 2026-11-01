# Upstream baseline migration

## Goal
Replace the bespoke image experiment path with a small upstream-derived codebase.
Preserve upstream training, sampling and evaluation semantics and expose only
an FFN replacement. The user's minimum image-baseline gate is FID below50;
FID200-plus rankings cannot support architecture claims.

## Actions
- [x] Review primary official repositories for a published CIFAR-10 setup,
  model family, prediction objective, license and replaceable FFN.
- [x] Select the closest compatible upstream without silently changing JiT's
  clean-image-prediction/Transformer research question.
- [x] Pin the source commit and copy upstream code/license with attribution.
- [x] Keep a small baseline/config/FFN interface, recording every local change.
- [x] Verify the default baseline against upstream computation and a real
  small kernel, without changing or restarting the current training job.
- [x] Publish code and source provenance on GitHub and document remaining
  baseline-reproduction work before interpreting any ODE comparison.

## Preservation
Existing reports, checkpoints and sealed experiments remain intact. No original
checkout, raw data or runtime is deleted; no force push or new long training.
Existing training environments stay unchanged. A separate pinned Mac validation
environment is needed for the copied OpenCV/TensorBoard/FID imports.
The active runtime is a frozen external source snapshot and stays unchanged.

## Status
Source imported and verified. Author and fresh documented checks both pass.
Default SwiGLU/loss/sampler/EMA are copied; ports are explicit. The repository
now has one training entrypoint and one FFN extension. Publication is next;
FID below50 has not yet been reproduced.
