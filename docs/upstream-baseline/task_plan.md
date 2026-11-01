# Upstream baseline migration

## Goal
Replace the bespoke image experiment path with a small upstream-derived codebase.
Preserve upstream training, sampling and evaluation semantics and expose only
an FFN replacement. The user's minimum image-baseline gate is FID below50;
FID200-plus rankings cannot support architecture claims.

## Actions
- [ ] Review primary official repositories for a published CIFAR-10 setup,
  model family, prediction objective, license and replaceable FFN.
- [ ] Select the closest compatible upstream without silently changing JiT's
  clean-image-prediction/Transformer research question.
- [ ] Pin the source commit and copy upstream code/license with attribution.
- [ ] Keep a small baseline/config/FFN interface, recording every local change.
- [ ] Verify the default baseline against upstream computation and a real
  small kernel, without changing or restarting the current training job.
- [ ] Publish code and source provenance on GitHub and document remaining
  baseline-reproduction work before interpreting any ODE comparison.

## Preservation
Existing reports, checkpoints and sealed experiments remain intact. No deletion,
force push, package rebuild or new long training is selected by this migration.
The active runtime is a frozen external source snapshot and stays unchanged.

## Status
Inspecting upstream compatibility. Implementation uses daily-coding and
planning-with-files; keep changes minimal and reuse authoritative source.
