# Current project status

## Authorized next comparison, 14 November

The native SwiGLU baseline qualifies and remains fixed at FID50k35.91785.
Widened ISRU-D K8 and K1 use internal width1200 and 130,402,560 total parameters,
versus native130,405,632. Both are authorized for seed0, 100 epochs and FID50k
at20/40/60/80/100. Implementation checks and full-model CUDA profiling are
being prepared before submission. [Protocol](docs/celeba128-wide-20261114/README.md).

Checked 13 November 2026, 23:52 Shanghai time.

## Completed CelebA128 comparison

Both seed 0 native JiT-B/16 models finished 100 epochs and their FID50k
measurements. **Native SwiGLU wins this comparison: 35.91785 versus 43.90767.**
K8 FID is 7.98981 points higher (+22.24%), despite using 32.61% fewer parameters.
Its steady training epoch is 1.75 times as long. The current K8 design has no
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

## Code and next-run settings

The [next-step audit](docs/celeba128-20261112/next-step-audit.md) recommends
frozen K8 term/response diagnostics, then a proposed K1 versus narrow SwiGLU
comparison. These controls would test the synthetic K1 transfer and separate
order from parameter budget. K8 remains the selected reference. No new
training is submitted. The current wrapper's completed output names and old
readiness binding must be replaced with a fresh registered run before launch.

The active code has [FID every 20 completed epochs](docs/celeba128-20261112/periodic-fid.md)
and complete numbered checkpoints 20/40/60/80/100. Regression checks verify
that evaluation preserves training RNG and parameter updates. Older 20/40/60/80
full checkpoints do not exist for the completed pair. New training budgets
include all five FID50k evaluations; provisional K8/native limits are 14h/8h.

- Development: `/scratch/zz5070/ODE_FFN/ODE-FFN`.
- Frozen model source: `/scratch/zz5070/ODE_FFN/runtimes/celeba128-20261112`, commit 65864ad.
- Environment: `/scratch/zz5070/ODE_FFN/envs/jit-celeba-cu128`.
- Training outputs: `/scratch/zz5070/ODE_FFN/runs/celeba/20261112-b16-{ffn}-s0`.
- Evaluation outputs: `/scratch/zz5070/ODE_FFN/runs/celeba/evaluations/20261113-b16-{ffn}-s0-ckpt100-ema9999`.

Reports 1/2 and all raw training artifacts remain unchanged. Evidence is indexed
in [docs/README.md](docs/README.md); current research decisions are in [notes.md](notes.md).
