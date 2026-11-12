# CelebA128: 100 epochs, one allocation

Checked 12 November 2026, 16:15 Shanghai time.

Both arms are submitted with 100 epochs, one six-hour priority allocation,
six CPUs and four data workers. Estimated training is 4–5 hours per arm,
including data/compute overlap and checkpoint/startup overhead. The estimate
uses measured GPU updates and real-image data probes, not completed epochs.

| Arm | Job | State | Reason |
|---|---|---|---|
| swiglu | 22822267 | PENDING | QOSMaxGRESPerUser |
| isru_d | 22822268 | PENDING | QOSMaxGRESPerUser |

No planned two-hour stop or automatic requeue remains. Actual Slurm checks
confirm TimeLimit 06:00:00, Requeue 0 and NumCPUs 6. Model seed 0, batch 256,
LR 5e-5, noise .5 and the native SwiGLU/ISRU-D K8 definitions remain fixed.
The previous 200-epoch jobs were cancelled while pending, with 0 elapsed
seconds; no training progress was discarded. [Superseded launch](launch-200epoch-superseded.json).

JiT's published ImageNet schedules are 200/600 epochs; AsymFlow rank controls
use 160; related CelebA weighting/parameterization experiments use 300.
Our 100 epochs/63,500 updates are a smaller initial budget. Evaluate FID before
deciding whether to extend. The [budget note](budget-20261112.md) provides
primary sources, exact arithmetic and the measured I/O/GPU components.

Full-model readiness 22813320 passed on H100 at the same frozen source and
batch. Reuse that receipt instead of spending another GPU probe. The source
runtime remains 65864ad; the revised wrapper is e40da5f and Slurm stores it
at submission. [GPU receipt](cuda-readiness.json).

Monitor PID 2590713 polls every 60 seconds at
`/scratch/zz5070/ODE_FFN/operations/celeba128-20261112/monitor.json`.
Each job records GPU utilization/memory/power every 30 seconds and retains
atomic per-epoch model/optimizer/two-EMA/RNG checkpoints and P snapshots.
Checkpoints support recovery from exceptional failure; they do not trigger
periodic restarts. Finalized reports and old 64px runs are untouched.

[Configuration](configuration.json), [launch commands](launch.json),
[verification](budget-verification.json), [earlier status](status-before-budget-revision.md).
