# Continue the fixed widened models to160 epochs

14 November 2026. The user requests further training because neither
trajectory has flattened. K1 FID improves50.10520->37.92486 from80to100
(24.31%); K8 improves50.30532->38.13574 (24.19%). The100epoch results are
a fixed-budget comparison, not evidence that either architecture converged.

Continue both existing seed0 checkpoints to160epochs, adding60epochs/38,100
updates per arm. Evaluate standard FID50k at120/140/160. Use the same
frozen source17b61a5, internal1200, batch256, AdamW5e-5, constant LR, original
optimizer moments, both EMA states and training RNG. Epoch numbers continue
from100; do not repeat warmup or initialize new weights.

Retain the original100epoch full checkpoints and published results. New
numbered checkpoints0120/0140/0160 and P snapshots append to the same run
folders; checkpoint-last/progress and training logs advance normally. The
previous100epoch SHA256 anchors and result receipt remain preserved. The
qualified native SwiGLU100epoch model remains fixed at35.91785. An ODE
score after160epochs must be labeled with its larger training exposure.

Stop this stage at160epochs or an execution failure. Observe the three new
FID points before selecting a further extension. Keep seed0, firstEMA.9999,
native Heun50/CFG0,50k samples and the same standard reference.

## Launch contract

Use the existing environment and frozen runtime. No package or model code
changes are needed. After checkpoint equality and recipe verification, invoke
the active Slurm wrapper with the original runtime inODE_FFN_CODE,
ODE_FFN_EPOCHS=160, ODE_FFN_FRESH_START=0 and MAX_TRAIN_SECONDS=0. Each run
must detect its existing checkpoint and print resume after100completed epochs.

Request one continuous priority allocation per arm,1GPU,6CPUs,4workers,24GiB,
all six registered GPU types, no requeue. Include three complete FID50k
evaluations in time limits. Use a separate continuation monitor/launch record
so completed100epoch receipts remain historical. GPU telemetry30seconds;
job/progress/FID monitoring60seconds.

The current conservative limits are14h for K8 and6h for K1 across all eligible
GPU types. On H100, the measured60epoch K8 continuation projects about4–5h
including three FIDs. K1 on L40S projects about5–6h with margin. The K8
limit also covers slower-card execution. Check scheduler admission without
submitting, from the active Torch checkout:

```bash
sbatch --test-only --job-name=celeba-wide-k8-s0-to160 --time=14:00:00 --no-requeue --export=ALL,ODE_FFN_CODE=/scratch/zz5070/ODE_FFN/runtimes/celeba128-wide-20261114,ODE_FFN_EPOCHS=160,ODE_FFN_MAX_TRAIN_SECONDS=0,ODE_FFN_FRESH_START=0 /scratch/zz5070/ODE_FFN/ODE-FFN/slurm/celeba.slurm isru_d celeba128-wide 8
sbatch --test-only --job-name=celeba-wide-k1-s0-to160 --time=06:00:00 --no-requeue --export=ALL,ODE_FFN_CODE=/scratch/zz5070/ODE_FFN/runtimes/celeba128-wide-20261114,ODE_FFN_EPOCHS=160,ODE_FFN_MAX_TRAIN_SECONDS=0,ODE_FFN_FRESH_START=0 /scratch/zz5070/ODE_FFN/ODE-FFN/slurm/celeba.slurm isru_d celeba128-wide 1
```

A new launch replaces only `--test-only` with `--parsable` after resume-state
verification. Previous terminal jobs must remain completed before submission.
