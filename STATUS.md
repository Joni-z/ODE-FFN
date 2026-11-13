# Current project status

Checked 13 November 2026, 21:51 Shanghai time.

## Baseline FID50k 35.92; K8 evaluation running

Both CelebA128/native JiT-B/16 runs completed 100 epochs/63,500 updates, seed 0,
with matching recipes and valid finite model/optimizer/two-EMA checkpoints.
No additional training is selected. Native SwiGLU reached FID50k 35.91785
at checkpoint 100; K8 evaluation remains in progress.

| Arm | Job | Completed | Final training loss | Elapsed | Mean GPU use after first 5 min |
|---|---|---:|---:|---|---:|
| Native SwiGLU |22822267 |100/100 |.02993375 |3h12m30s |95.73% |
| ISRU-D K8 |22822268 |100/100 |.03155256 |5h35m53s |98.05% |

Both used separate L40S GPUs on gl066. At matched exposure K8's training loss
is 5.41% higher and its steady epoch time 1.75 times native. Its 101 P snapshots
are complete and the last matches checkpoint-last exactly. Raw P mean
spectral norm grows 1.1486 to 2.1631 while stable rank drops 193.95 to 76.65.
See [training results and figures](docs/celeba128-20261112/results.md).

## Generation evaluation

The [evaluation protocol](docs/celeba128-20261112/evaluation.md) fixes EMA .9999,
seed 0, author Heun50, CFG0 and 50,000 images per model against the same
deterministic CelebA training-crop reference. It calls the frozen author's
entrypoint, without changing the model, sampler or metric. The ordinary
CelebA128 baseline now passes the FID50k below 50 gate. K8 has no FID ranking yet.

The original 100-epoch/six-hour training budget is complete. Fixed-checkpoint
evaluation was submitted separately with a three-hour limit per arm:

| Arm | Evaluation job | State at this check | FID50k |
|---|---|---|---:|
| Native SwiGLU |22993694 |COMPLETED 0:0, A100 ga013, 34m47s |35.91785 |
| ISRU-D K8 |22993695 |RUNNING on A100 ga022 |Pending |

The [submission receipt](docs/celeba128-20261112/evaluation-launch.json) binds
checkpoint-only commands to source 65864ad and wrapper fb72e7c. A separate
monitor polls every 60 seconds (PID3150165). The environment is warm-reused;
Inception weights are already staged. No training continuation or new model
search is scheduled. Both allocated evaluations passed checkpoint SHA256 checks
and began generation. [Initial examples](docs/celeba128-20261112/results.md#initial-generated-examples)
show recognizable faces from both models. [Baseline FID receipt](docs/celeba128-20261112/FID100-swiglu.json)
binds all 50,000 images to checkpoint 100 and the fixed reference; K8 FID is pending.

## Paths

- Development: `/scratch/zz5070/ODE_FFN/ODE-FFN`.
- Frozen training/evaluation model source: `/scratch/zz5070/ODE_FFN/runtimes/celeba128-20261112`, commit 65864ad.
- Environment: `/scratch/zz5070/ODE_FFN/envs/jit-celeba-cu128`.
- Training outputs: `/scratch/zz5070/ODE_FFN/runs/celeba/20261112-b16-{ffn}-s0`.
- Evaluation outputs: `/scratch/zz5070/ODE_FFN/runs/celeba/evaluations/20261113-b16-{ffn}-s0-ckpt100-ema9999`.

Reports 1/2 and historical experiment paths remain unchanged. Current evidence
is indexed in [docs/README.md](docs/README.md); concise decisions are in
[notes.md](notes.md).

## Next-training FID cadence

[Periodic FID](docs/celeba128-20261112/periodic-fid.md) now uses the native
online evaluator every 20 completed epochs and retains full numbered
checkpoints at 20/40/60/80/100. Training RNG is preserved across evaluation.
The 100-epoch CPU entrypoint regression and 18 upstream metric-definition
comparisons passed. Existing jobs 22993694/22993695 continue on frozen 65864ad.
Earlier 20/40/60/80 full checkpoints do not exist for those completed runs.
No new training or duplicate evaluation was submitted. Future allocations
include all five FID50k evaluations rather than reuse the old training-only
six-hour budget; the provisional K8/native limits are 14h/8h.
