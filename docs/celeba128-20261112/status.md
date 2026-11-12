# CelebA128 submission status

Checked 12 November 2026, 15:18 Shanghai time. This is a timestamped submission snapshot, not a live dashboard.

The revised pair is submitted on torch, with a full-model GPU readiness gate.
CPU data/resume, author compatibility and an independent documented CPU
invocation pass. The GPU profile is running on an H100 at gh009; neither training arm has
started, and memory, throughput, samples and FID are not yet available.

| Job | ID | State | Reason |
|---|---|---|---|
| GPU readiness | 22813320 | COMPLETED |  |
| swiglu | 22813322 | PENDING | Dependency |
| isru_d | 22813323 | PENDING | Dependency |

Both training jobs depend on successful readiness 22813320. Its receipt must
also match model source 65864ad and batch 256 before the launcher proceeds.
The shared recipe is native JiT-B/16, CelebA128, batch 256, author-scaled
LR 5e-5, noise std .5 and seed 0. Native SwiGLU and ISRU-D K8 keep the same
AdamW/time/EMA/loss. The target is 200 epochs, with checkpointed two-hour
priority allocations and at most 20 self-requeues. All six GPU constraints
remain joined by OR; the account's actual partition permissions still apply.

## Corrected monitoring and preserved first attempt

Readiness 22810354 stopped after 4 seconds on gh011 before Python model execution:
NVML could not find the selected device. Dependencies 22810863/22810864 were
cancelled without training. These are preserved in [attempt1](launch-attempt1.json).

[Diagnostic 22812110](gpu-visibility.json) completed 0:0 on gh009 in 17 seconds:
Slurm global GPU ID 1 differs from visible NVML/CUDA index 0. CUDA forward and
backward pass. The corrected active wrappers select the GPU UUID reported by
CUDA, including NVIDIA's GPU- prefix. The exact collector block passes its
[isolated regression](gpu-uuid-regression.json). The corrected readiness CSV now contains UUID-bound GPU utilization, memory
and power records on gh009; the full-model profile is still running.

Training/model source is frozen at 65864ad in
`/scratch/zz5070/ODE_FFN/runtimes/celeba128-20261112`.
The corrected Slurm wrappers are from b0a4967 and are stored by Slurm on
submission. Old 64px runs and finalized Report1/2 are untouched.

## Live records

The read-only monitor PID 2156682 polls every 60 seconds and stops once all
three jobs terminate. Its output is
`/scratch/zz5070/ODE_FFN/operations/celeba128-20261112/monitor.json`.
Training GPU utilization, memory and power are logged every 30 seconds in the
run directories; readiness logs every 5 seconds. Per-allocation telemetry,
epoch records, full checkpoints and P-history snapshots retain progress
across requeues. Nonfinite loss, missing checkpoints, no epoch progress or a
failed collector stops the job rather than starting over.

See [protocol](protocol.md), [configuration](configuration.json),
[actual launch commands](launch.json) and [verification](verification.json).
