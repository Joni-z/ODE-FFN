# Fixed-checkpoint diagnostics: execution status

The original submission was Slurm job **20405164**. It was cancelled while pending after a successful numerical GPU preflight on host 145. The standalone evaluation subsequently completed **66/66 cells**, with all workers and the runner exiting successfully. See the [migration record](migration-145.md) and [completion record](completion-145.json). Full metric interpretation awaits archive collection and a local artifact audit.

The details below describe the original Slurm submission and its monitor; that cancelled job is no longer the active execution route.

- Frozen evaluation source: commit `1dd6a86daed8521bea71c5faf3df12107cff7d07`.
- Frozen runtime: `/scratch/zz5070/ODE_FFN/experiments/stage_20261020_frozen_order/runtime`.
- Outputs: `/scratch/zz5070/ODE_FFN/runs/synthetic/stage_20261020_frozen_order`.
- [Checkpoint and source manifest](checkpoint-manifest.json): 18 existing seed-0 checkpoints and 17 frozen files, verified before submission.
- [Submission receipt](submission.json): job ID and detached local monitor process.
- Expected evaluations: 60 ODE interventions and six unchanged SiLU/SwiGLU controls, plus native precision/reproduction checks. No new training.

## GPU use

The job requests one GPU, six CPU cores, and 24 GB of host memory, with a two-hour time limit. Its feature expression allows A100, L40S, H100, H200, or RTX6000; actual eligibility also depends on the account and partitions.

The first GPU computation measures aggregate throughput for one and two simultaneous workers on the same card. Two workers are selected only if the measured gain is at least 10%. The comparison uses the expensive matrix-exponential component, so it is not a claim about end-to-end training throughput. The three datasets then share that allocation at the selected concurrency.

`throughput.json` records both measurements and the decision. `monitoring/gpu_usage.csv` records utilization, memory, and power every 10 seconds. CPU-only matrix controls, figure generation, and report writing do not require the GPU allocation. The job exits after evaluation and integrity checks.

## Inspect live status

On torch:

```bash
/scratch/zz5070/ODE_FFN/envs/synthetic/bin/python \
  /scratch/zz5070/ODE_FFN/experiments/stage_20261020_frozen_order/runtime/scripts/monitor_diagnostics.py \
  probe --job 20405164 \
  --output /scratch/zz5070/ODE_FFN/runs/synthetic/stage_20261020_frozen_order
```

The detached monitor is running on the Mac mini, not the MBP or the GPU node. Its current status is written to:

```text
/Users/mr.z/ODE-FFN-report-work/docs/research-2026-10-20-followup/monitoring-local/latest.json
```

It checks every 60 seconds, retries connection failures, and records a warning after 15 minutes of running without a new cell. A warning alone does not establish a hang. Monitoring requires that the local machine remain running and able to reach torch. The monitor does not resubmit jobs. The GPU-side telemetry runs independently inside the Slurm allocation.

Completion requires all 66 finite cells, successful worker exits, intact saved samples and input hashes, and Slurm `COMPLETED` with exit code `0:0`. Read `summary.json` and the monitor together. Do not infer completion from a disappeared process or an empty queue.

## CPU control completed

[Controlled-operator results](controlled-operators.json) contain 255 evaluations across 17 matrices, three hidden durations, and five evaluation modes. In the three-dimensional chain family, the eigenvalues remain -0.25 and the largest singular value remains 1. At hidden duration 1, the exponential's largest singular value varies from approximately 1.225 to 1.397 as the chain geometry changes.

This demonstrates that eigenvalues and the largest singular value alone do not determine finite-time amplification. It is an illustrative mathematical example, not evidence that a particular constraint improves flow matching. The matrix-exponential and fixed-checkpoint comparisons remain separate tests; see the [protocol](protocol.md) for their interpretation.
