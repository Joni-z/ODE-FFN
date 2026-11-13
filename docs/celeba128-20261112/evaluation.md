# Checkpoint100 evaluation

Registered 13 November 2026 before reading generation/FID results. Both training
jobs completed 100 epochs. Evaluate each saved checkpoint without additional
training through the frozen author's `main_jit.py --evaluate_gen` entrypoint.

The primary comparison uses first EMA .9999, seed 0, native Heun50 intervals
including the final Euler step, CFG0, generation batch256 and 50,000 saved
images. Use deterministic 128px author crops of all 162,770 official training
images as the reference, through the pinned author torch-fidelity fork.
The second EMA is not substituted after seeing these results.

Both models share these evaluation settings. Retain model/checkpoint SHA256,
sampler/reference identities, generated PNGs, original metric logs and GPU
telemetry. Training loss and FID measure different quantities. The ordinary
baseline must meet the existing FID50k below 50 gate before architecture claims.

GPU evaluation requests one allowed GPU, six CPUs, 24 GiB and three hours,
priority account and all six GPU constraints. Local extraction keeps real
reference image reads off the scratch small-file path. Stage the official
Inception weights through the installed library on the login node first.
The checkpoint100 model and math remain at frozen source 65864ad. The active
Slurm wrapper records execution and extracts the author's logged metric; it
does not implement a new sampler or FID estimator.

After `ssh torch`, validate resource admission without allocating:

```bash
cd /scratch/zz5070/ODE_FFN/ODE-FFN
sbatch --test-only --export=ALL,ODE_FFN_CODE=/scratch/zz5070/ODE_FFN/runtimes/celeba128-20261112 slurm/celeba_eval.slurm swiglu
```

Formal submission replaces `--test-only` with `--parsable`, sets a distinct
job name, and passes `swiglu` or `isru_d`. Submit each arm once; inspect a
failed/incomplete result before a retry. A complete `result.json` is the
evaluation completion gate, not a finished training job or a generated image.

Outputs are under
`/scratch/zz5070/ODE_FFN/runs/celeba/evaluations/20261113-b16-{ffn}-s0-ckpt100-ema9999`.
The frozen entrypoint's inner sample folder says `epoch0000` because its
evaluation call passes step0 to the logger. The parent directory, checkpoint
hash and result record bind these images to completed epoch100.

The queue monitor uses a separate evaluation launch receipt under
`operations/celeba128-20261112/evaluation-ema9999-50k/`. It preserves the
completed training manifest and outputs.
