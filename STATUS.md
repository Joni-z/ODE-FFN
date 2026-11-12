# Current project status

Last checked 12 November 2026, 18:34 Shanghai time. This GitHub page is a snapshot; the scratch monitor
provides live job/progress state.

## Active experiment

Compare native SwiGLU with ISRU-D K8 on unconditional CelebA128 using the
author-derived native JiT-B/16. Both full models passed GPU readiness on H100.
Formal epoch throughput and FID are still pending.

| Setting | Current value |
|---|---|
| Training horizon | 100 epochs, 63,500 optimizer updates |
| Allocation | One GPU, 6 CPUs, 4 data workers, 24 GiB RAM, 6 hours per arm |
| Batch / seed | 256 / 0 |
| Optimizer / LR | AdamW (.9,.95), 5e-5, no weight decay |
| Noise / time | Noise std .5; logit-normal mean -.8, std .8 |
| Warmup / EMA | 5 epochs; .9999/.9996 |
| Restart behavior | No automatic requeue or planned runtime pause; checkpoint each completed epoch |
| Quality decision | Evaluate FID50k; ordinary baseline below 50 before architecture claims |

Estimated training is 4–5 hours per arm, excluding queue time. The
[budget calculation](docs/celeba128-20261112/budget-20261112.md) includes the
GPU update and data-loading measurements behind the six-hour request.

| Arm | Job | State at last check | Scheduler reason |
|---|---|---|---|
| swiglu | 22822267 | PENDING | None |
| isru_d | 22822268 | PENDING | None |

## Where the current work lives

| Object | Path |
|---|---|
| Mutable development checkout | `/scratch/zz5070/ODE_FFN/ODE-FFN` |
| Frozen training source | `/scratch/zz5070/ODE_FFN/runtimes/celeba128-20261112` at 65864ad |
| Environment | `/scratch/zz5070/ODE_FFN/envs/jit-celeba-cu128` |
| Data | `/scratch/zz5070/ODE_FFN/datasets/celeba` |
| Run outputs | `/scratch/zz5070/ODE_FFN/runs/celeba/20261112-b16-{ffn}-s0` |
| Live queue/epoch monitor | `/scratch/zz5070/ODE_FFN/operations/celeba128-20261112/monitor.json` |

The [Slurm wrapper](slurm/celeba.slurm) specifies current launch flags;
[launch.json](docs/celeba128-20261112/launch.json) records the actual submitted
commands. The frozen model runtime and scheduler-stored wrappers keep queued
experiments independent of later documentation edits.

## Next measurements

Collect actual epoch timings and checkpoint/P history, then evaluate the
trained baseline and candidate with the registered sampler/FID reference.
Use those measurements to decide whether 100 epochs need extension.

Reports 1/2 are final. The earlier 64px runs and completed synthetic evidence
are preserved in the [research map](docs/README.md). The separate
[synthetic result summary](notes.md) records what those experiments established.
