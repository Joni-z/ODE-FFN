# Continue the native SwiGLU control to 160 epochs

Authorized 14 November 2026: the user requests the ordinary baseline alongside
K1/K8 at the same training exposure. Resume the qualified native checkpoint
after 100 epochs and train through 160, with FID50k at 120/140/160. Keep seed 0,
batch 256, original SwiGLU, JiT-B/16,128px, AdamW 5e-5, constant LR, warmup 5
already completed, noise/time distribution, both EMAs and the standard metric.

## Source and checkpoint preservation

Original 100-epoch run/source 65864ad and FID 35.9178543 remain preserved. The
original trainer predates complete numbered periodic checkpoints. Continuation
uses the validated periodic-evaluation runtime 17b61a5, also used by K1/K8.
Native backbone/crop/LR code is byte-identical; native field/loss/solver/EMA
functions and optimizer parameter grouping have identical syntax trees.
Checks compare native initialization/RNG and full 128px predictions after
loading the real trained checkpoint. The original checkpoint SHA256 is
verified, with complete model/optimizer/twoEMA/RNG states restored.

Use a new output folder:
`/scratch/zz5070/ODE_FFN/runs/celeba/20261114-b16-swiglu-s0-to160`.
Copy the original checkpoint into immutable checkpoint-0100.pth there.
The initial checkpoint-last.pth links to this copied anchor; atomic
temporary-file replacement detaches later checkpoints from the anchor. Subsequent writes advance the copied
last checkpoint and append 0120/0140/0160; the original run stays untouched.
No model reinitialization or second warmup occurs.

## Budget and validation

One priority GPU, all six GPU types, 6 CPUs, 4 workers, 24 GiB, 6 hours, no requeue.
Sixty native L40S training epochs project 1.90h. Three FIDs use the observed
near-equal-size K1/L40S overhead 49min each as a conservative allowance.
With 20% margin and 15min startup, total 5.47h; request 6h in one allocation.

The existing CUDA environment/spec d8a61f4e is reused. Native GPU readiness
was already validated at batch 256; source compatibility and the existing
periodic-evaluation RNG regression establish that the bookkeeping change
preserves the native model. No new GPU probe or package install is needed.

After native compatibility and staged-checkpoint validation, check admission
from the active Torch checkout without submitting:

```bash
sbatch --test-only --job-name=celeba-swiglu-s0-to160 --time=06:00:00 --no-requeue --export=ALL,ODE_FFN_CODE=/scratch/zz5070/ODE_FFN/runtimes/celeba128-wide-20261114,ODE_FFN_EPOCHS=160,ODE_FFN_MAX_TRAIN_SECONDS=0,ODE_FFN_FRESH_START=0 /scratch/zz5070/ODE_FFN/ODE-FFN/slurm/celeba.slurm swiglu celeba128-baseline160 8
```

The 8 argument is an unused legacy order field for native SwiGLU. It does not
introduce Taylor operations into the baseline. Formal launch changes only
`--test-only` to `--parsable`. The wrapper requires the baseline-specific
compatibility receipt; old source guards are retained for other recipes.

Use a separate 60-second queue/progress/FID monitor and per-job 30-second GPU
telemetry. Stop at 160 or an execution failure. Keep all numbered milestones,
then compare the three architectures at matching 120/140/160 epochs.

## Submitted execution

Baseline job **23117422** targets 160 epochs from the staged 100-epoch checkpoint.
It is training on H100 gh012 and completed epoch101. Optimizer restore and
continued updates beyond63500 are confirmed. Monitor 2838752 polls 60 seconds;
GPU telemetry starts every 30 seconds within the allocation.

[Compatibility](compatibility.json), [check source](check-compatibility.py), [staged checkpoint](staged-checkpoint.json),
[launch](launch.json), [scheduler check](agent-doc-pass.json),
[monitor snapshot](monitor.json). Native model/training math and the original
100-epoch result are preserved; equal-exposure 120/140/160 comparisons follow
as each architecture completes its evaluations.
