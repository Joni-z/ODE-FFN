# Current project status

## Running response comparison on Mac mini CPU

[Response-family comparison](docs/research-assessments/2026-11-14-next-architecture/response-screen-proposal.md):
compare ISRU,tanh,arctan,Softsign,erf,slope-normalized SiLU/GELU and linear
at K1/K8 on Ring8/Product16. The user authorized execution, then selected
Mac mini CPU. All32 focal trajectories and4ordinary references will use
the same CPU host/backend. Seed0 and native training/metrics remain fixed.
Independent selection/confirmation arrays are hash-bound before training.
Source b5fe7aa passed16 native continuations, kernel checks and a fresh
documented pass. The200step benchmark selected2workers (1.89x aggregate
throughput). At the latest audit,19of36 pipelines completed20k training and
selection evaluation with no failures. Ring8 is complete, including its
SwiGLU64/Muon reference: pooled SW2 0.11823 versus Softsign-K8 0.06956.
Product16 is in progress; the final shortlist and independent confirmation
remain pending. [Current comparison](docs/research-assessments/2026-11-15-response-screen/current-comparison.md).
Evaluation/selection/confirmation
watcher3767 uses separate analysis source0e78ce7 and is waiting for training.
[Execution amendment](docs/research-assessments/2026-11-15-response-screen/protocol.md).
[Launch receipt](docs/research-assessments/2026-11-15-response-screen/launch-receipt.json)
and [benchmark](docs/research-assessments/2026-11-15-response-screen/benchmark.json).
Raw output root: `/Users/mr.z/ode-workspace-prep/response-screen-20261115/`.
No other user's GPU jobs were stopped;
the user explicitly revoked earlier yp2841 preemption permission.

## Completed order-by-response comparison,15 November

Four seed 0 K8 runs, common-bank evaluations and frozen-order diagnostics
completed on 145 GPUs 4–7. All queues exited zero and released their cards.
ISRU-K8 worsens Ring8 pooled SW2 from 0.13261 to 0.16779 relative to K1;
Product16 changes from 0.07206 to 0.07321. Tanh-K8 is the positive case:
Product16 improves from 0.07006 to 0.06427, with seven of eight banks better.
High-order terms are active, and the measured K8 exponential error is small.
Their necessity for K8-trained weights does not establish an ISRU advantage.
Ring8 C8 projection still largely removes mode imbalance at either order.

[Findings and figures](docs/research-assessments/2026-11-15-order-response/findings.md),
[completion](docs/research-assessments/2026-11-15-order-response/completion.json),
[metric verification](docs/research-assessments/2026-11-15-order-response/results/verification.json).
Runtime source is frozen at 9f72bd3. The subsequent planning discussion
prioritizes the bounded CelebA checkpoint diagnosis described below. The
rotation-training control remains an optional small synthetic check.
No new image run is submitted; finalized Reports1/2 are preserved.

## Completed professor follow-up,15November

Five seed0 trajectories, all15k/20k common-bank evaluations, and the independent
original-script SwiGLU audit completed on145 GPUs4-7. Monitors ended; no pending
jobs. Training runtime`da8b8a4` and separate analysis runtime`14c65a7` are preserved.

The main finding: Ring8 ISRU-D's symmetric error improves15k->20k while its
asymmetric error grows. C8projection reduces pooledSW2 from0.13261 to0.05268;
Free ISRU, Tanh-D and SwiGLU show similar sampling gains. Matched tanh controls
show no consistent ISRU-specific advantage. Product16 weight/update spectra
confirm that high-rank Muon updates need not yield higher-rank final weights.

[Findings, figures and next decision](docs/research-assessments/2026-11-15-professor-execution/findings.md),
[metric verification](docs/research-assessments/2026-11-15-professor-execution/results/verification.json),
and [completion evidence](docs/research-assessments/2026-11-15-professor-execution/completion.json).
Ring8 replays are separate from unavailable Lenovo checkpoints. The Muon control
has a documented cross-host discrepancy, despite exact old/new entry agreement
on145. All before/after C8 comparisons use the same weights. Reports1/2 remain final.

## Completed baseline/K1/K8 continuation

All three completed160epochs and FID50k, exit0:0. Native SwiGLU leads at
every matching checkpoint.

| Epoch | Native SwiGLU | Wide K1 | Wide K8 |
|---|---:|---:|---:|
| 100 | 35.91785 | 37.92486 | 38.13574 |
| 120 | 29.47602 | 32.50213 | 32.85067 |
| 140 | 25.98559 | 29.62906 | 30.08220 |
| 160 | 24.30357 | 27.78868 | 28.08343 |

At160, K1 trails native by3.48511FID and K8 by3.77986. All three still improve
with added training. No experiment is extended beyond160.

[Matched curves and completed evidence](docs/celeba128-wide-20261114/continuation-160/matched-results.md).

The [next execution plan](docs/research-assessments/2026-11-14-next-architecture/execution-plan.md)
specifies a one-hour checkpoint diagnosis, numerical/readiness checks and two
previously proposed100epoch runs: frozen-D K2 versus refreshed-ISRU midpoint.
After review of the matched tanh results, that ISRU pair is retained as an
option rather than the default next experiment. It has not been implemented
or submitted. Keep seed0, original baseline and FID every20.

A later [professor-comment review](docs/research-assessments/2026-11-15-professor-comments.md)
recommends spectrum/C8-symmetry diagnostics and a matched activation control
before selecting that image pair. The authorized follow-up above now executes
this recommendation; the image pair remains unsubmitted.

Verified15November2026 from scheduler completion, progress, console and
TensorBoard. No new model code or finalized report was changed.

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
