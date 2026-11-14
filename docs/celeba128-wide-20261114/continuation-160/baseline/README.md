# Continue the native SwiGLU control to160epochs

Authorized14November2026: the user requests the ordinary baseline alongside
K1/K8 at the same training exposure. Resume the qualified native checkpoint
after100epochs and train through160, with FID50k at120/140/160. Keep seed0,
batch256, original SwiGLU, JiT-B/16,128px, AdamW5e-5, constant LR, warmup5
already completed, noise/time distribution, both EMAs and the standard metric.

## Source and checkpoint preservation

Original100epoch run/source65864ad and FID35.9178543 remain preserved. The
original trainer predates complete numbered periodic checkpoints. Continuation
uses the validated periodic-evaluation runtime17b61a5, also used by K1/K8.
Native backbone/crop/LR code is byte-identical; native field/loss/solver/EMA
functions and optimizer parameter grouping have identical syntax trees.
Checks compare native initialization/RNG and full128px predictions after
loading the real trained checkpoint. The original checkpoint SHA256 is
verified, with complete model/optimizer/twoEMA/RNG states restored.

Use a new output folder:
`/scratch/zz5070/ODE_FFN/runs/celeba/20261114-b16-swiglu-s0-to160`.
Copy the original checkpoint into immutable checkpoint-0100.pth and
checkpoint-last.pth there before launch. Subsequent writes advance the copied
last checkpoint and append0120/0140/0160; the original run stays untouched.
No model reinitialization or second warmup occurs.

## Budget and validation

One priority GPU, all six GPU types,6CPUs,4workers,24GiB,6hours,no-requeue.
Sixty native L40S training epochs project1.90h. Three FIDs use the observed
near-equal-size K1/L40S overhead49min each as a conservative allowance.
With20% margin and15min startup, total5.47h; request6h in one allocation.

The existing CUDA environment/specd8a61f4e is reused. Native GPU readiness
was already validated at batch256; source compatibility and the existing
periodic-evaluation RNG regression establish that the bookkeeping change
preserves the native model. No new GPU probe or package install is needed.

After native compatibility and staged-checkpoint validation, check admission
from the active Torch checkout without submitting:

```bash
sbatch --test-only --job-name=celeba-swiglu-s0-to160 --time=06:00:00 --no-requeue --export=ALL,ODE_FFN_CODE=/scratch/zz5070/ODE_FFN/runtimes/celeba128-wide-20261114,ODE_FFN_EPOCHS=160,ODE_FFN_MAX_TRAIN_SECONDS=0,ODE_FFN_FRESH_START=0 /scratch/zz5070/ODE_FFN/ODE-FFN/slurm/celeba.slurm swiglu celeba128-baseline160 8
```

The8 argument is an unused legacy order field for native SwiGLU. It does not
introduce Taylor operations into the baseline. Formal launch changes only
`--test-only` to `--parsable`. The wrapper requires the baseline-specific
compatibility receipt; old source guards are retained for other recipes.

Use a separate60second queue/progress/FID monitor and per-job30second GPU
telemetry. Stop at160 or an execution failure. Keep all numbered milestones,
then compare the three architectures at matching120/140/160epochs.
