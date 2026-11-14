# Current project status

## Native baseline continuation authorized

The user requests native SwiGLU through160 as well as K1/K8, with FID50k at
120/140/160. Native math, full-checkpoint outputs and initialization/RNG
compatibility checks passed. Baseline job **23117422** is submitted from the verified 100-epoch
checkpoint in a separate continuation folder. It has been allocated GPU
on H100 gh012, completed epoch106; optimizer step67310 confirmed. Optimizer restore is
confirmed; GPU93–97%. Monitor2838752 is active. The architecture and original100epoch
result are preserved.
[Baseline continuation](docs/celeba128-wide-20261114/continuation-160/baseline/README.md).

## Continuation to 160 epochs

K1 completed **160 epochs**, FID50k **27.78868**, job 23085013 exited0:0
on L40S gl024. K8 job 23085012 is running on L40S gl060, completed 142 epochs
and training epoch 143 at the checked snapshot; latest FID 30.08220 at 140.

| Epoch | Wide K1 FID50k | Wide K8 FID50k |
|---|---:|---:|
|100 |37.92486 |38.13574 |
|120 |32.50213 |32.85067 |
|140 |29.62906 |30.08220 |
|160 |27.78868 |Pending |

Both logs verify continuation from 100 with the original checkpoint and
optimizer. K1's full 160-epoch checkpoint has 101,600 optimizer updates and finite
model/twoEMA states; 161 P snapshots are retained. K8 GPU use 99–100%, ordinary
epochs ~305 seconds, remaining time ~3.5h including its final FID50k.
Monitor 2415208 remains active. No new jobs were submitted by this check.

[Continuation curves and verified status](docs/celeba128-wide-20261114/continuation-160/results.md).
Native 35.91785 is the fixed 100-epoch reference; the new scores use more training.

Checked 14 November 2026, 21:27 Shanghai time.

## Completed widened ISRU-D K8/K1

Both seed 0 runs completed 100 epochs,63,500 updates and all five FID50k
evaluations; both first-stage jobs exited 0:0. The continuation jobs are listed above.

| Model | Total parameters | Final FID50k |
|---|---:|---:|
| Fixed native SwiGLU |130,405,632 |**35.91785** |
| Widened K1 |130,402,560 |37.92486 |
| Widened K8 |130,402,560 |38.13574 |

Wide K8 improves over old small K8's43.90767 but still trails native by 2.21789.
K1 is .21088 below K8; the high-order expansion adds no final FID benefit here.
[FID curves and completed evidence](docs/celeba128-wide-20261114/results.md).

K8 job 23015406 used H100 gh011 for 6h08m11s; K1 job 23015408 used L40S gl068 for
7h15m28s, each including five FIDs. These different-GPU wall times are not a
speed ranking. Allocation-average GPU use was 83.92%/84.25% (K8/K1). All five
numbered full checkpoints and 101 P snapshots exist per arm. Final weights and
both EMAs are finite; all final P matrices match their trained checkpoints.

The persistent monitor 460860 recorded both COMPLETED states and exited
normally. Source remains frozen at 17b61a5; the qualified baseline and final
reports remain preserved. No additional experiment is submitted.

First 100-epoch completion was recorded earlier on 14 November 2026.

## Completed CelebA128 comparison

Both seed 0 native JiT-B/16 models finished 100 epochs and their FID50k
measurements. **Native SwiGLU wins this comparison: 35.91785 versus 43.90767.**
K8 FID is 7.98981 points higher (+22.24%), despite using 32.61% fewer parameters.
Its steady training epoch is 1.75 times as long. The earlier unexpanded K8 design has no
quality or training-time advantage at this 100-epoch budget.

| Measurement | Native SwiGLU | ISRU-D K8 |
|---|---:|---:|
| FID50k, checkpoint 100 | **35.91785** |43.90767 |
| Final training velocity loss |.02993375 |.03155256 |
| Full-model parameters |130,405,632 |87,879,936 |
| Training elapsed, L40S |3h12m30s |5h35m53s |
| Steady epoch time |113.73s |199.06s |
| Evaluation elapsed, A100 |34m47s |54m59s |

See [results, curves and examples](docs/celeba128-20261112/results.md).
The original result files and [comparison checks](docs/celeba128-20261112/FID100-comparison-verification.json)
verify exactly 50,000 PNGs per arm, matching evaluation settings, unchanged
checkpoint mtimes and agreement between original console and TensorBoard FID.

## Execution and fixed evaluation

Training jobs 22822267/22822268 and evaluation jobs 22993694/22993695 all
completed with exit 0:0. Training used distinct L40S GPUs on gl066; evaluation
used A100s on ga013/ga022. Both evaluation allocations have ended. Their
60-second monitor recorded both terminal results and exited normally.
No additional training is submitted for this comparison.

Evaluation uses the pinned standard torch-fidelity implementation, first
EMA .9999, seed 0, CFG0, author Heun50 with final Euler, 50,000 images and the
same 162,770 official CelebA training images with deterministic 128px crops.
The ordinary baseline meets the registered FID50k below 50 criterion.

All 101 K8 P snapshots are complete. Raw P mean spectral norm grows 1.1486 to 2.1631,
while stable rank drops 193.95 to 76.65. These describe learned P, without
identifying the cause of the FID gap.

## Evaluation support

The active code has [FID every 20 completed epochs](docs/celeba128-20261112/periodic-fid.md)
and complete numbered checkpoints 20/40/60/80/100. Regression checks verify
that evaluation preserves training RNG and parameter updates. Older 20/40/60/80
full checkpoints do not exist for the completed pair. New training budgets
include all five FID50k evaluations; provisional K8/native limits are 14h/8h.

- Development: `/scratch/zz5070/ODE_FFN/ODE-FFN`.
- Completed pair frozen source: `/scratch/zz5070/ODE_FFN/runtimes/celeba128-20261112`, commit 65864ad.
- Environment: `/scratch/zz5070/ODE_FFN/envs/jit-celeba-cu128`.
- Training outputs: `/scratch/zz5070/ODE_FFN/runs/celeba/20261112-b16-{ffn}-s0`.
- Evaluation outputs: `/scratch/zz5070/ODE_FFN/runs/celeba/evaluations/20261113-b16-{ffn}-s0-ckpt100-ema9999`.

Reports 1/2 and all raw training artifacts remain unchanged. Evidence is indexed
in [docs/README.md](docs/README.md); current research decisions are in [notes.md](notes.md).
