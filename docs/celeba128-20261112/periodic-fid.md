# FID every 20 completed epochs

Requested 13 November 2026. Subsequent image training enables the author's native
online evaluation every 20 epochs and at the final epoch. A 100-epoch run evaluates
at 20, 40, 60, 80 and 100. Loss, FFN, optimizer, sampling equations and EMA remain fixed.
No replay training or duplicate evaluation was submitted by this change.

## Metric and fixed comparison

Use [Option1, torch-fidelity](https://github.com/toshas/torch-fidelity).
The environment retains the author-pinned fork bfe72b995473e4fee9351705d7b005f17442a4ca.
Its FID feature/statistics functions, Inception forward/blocks and resize
function match the current official upstream definitions (18 AST comparisons).
This retains the author's precomputed-statistics support without a new FID
implementation. [Source audit](FID-current-upstream-audit.json).

Each point uses 50,000 generated PNGs, seed 0, first EMA .9999, author Heun50 with
final Euler, CFG0, Inception2048 features and the same 162,770 official CelebA
training images with deterministic 128px crops. Inception weights remain the
official file whose SHA256 is
`6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2`.
Do not switch implementations or reference preprocessing between FFNs.

## Code and retained state

The active [Slurm wrapper](../../slurm/celeba.slurm) now passes
`--online_eval --eval_freq 20 --num_images 50000 --gen_bsz 256 --sampling_method heun`.
Before each evaluation, save a full `checkpoint-0020.pth`, `checkpoint-0040.pth`,
etc., containing the model, both EMAs, optimizer and training RNG, alongside
`checkpoint-last.pth`. The numbered files remain available after later epochs.
`--resume` accepts an explicit checkpoint file or the existing run-directory
form. Evaluation logs/sample folders use completed epochs rather than epoch 0
or zero-based training indices. Original scalar tags and FID logging remain.

Sampling uses a forked Torch RNG seeded 0; NumPy/Python RNG states are restored.
The [entrypoint regression check](periodic-eval-check.json) ran 100 tiny CPU JiT
epochs with and without evaluation. It verified evaluations 20/40/60/80/100,
complete numbered checkpoints, explicit-file loading, identical sampling seeds,
and bitwise-identical final model, optimizer, both EMAs and training RNG.
The fixture replaces expensive FID with RNG-consuming callbacks; this is a
control-flow check and does not supply a model-quality measurement.

## Existing completed runs and time budget

The two completed 100-epoch training outputs contain only their last full
checkpoint. Their earlier P-only snapshots cannot reconstruct the20/40/60/80
models. Existing checkpoint100 evaluations 22993694/22993695 continue unchanged
on their frozen runtime, using the same standard metric and 50k samples.

The former 6-hour limit covered training alone. Five full FID50k evaluations
must be included in subsequent training budgets. A provisional K8 default
of 14 hours and native 8 hours includes measured L40S training 3.2/5.6 hours,
current A100generation about 30/50 minutes per evaluation, feature extraction,
and a hardware/runtime allowance. These are scheduling estimates across
hardware, not a measured complete periodic run. Re-estimate with completed
FID timings before registering a new run. No new training allocation is
requested here; existing 3-hour evaluation allocations are unchanged.
