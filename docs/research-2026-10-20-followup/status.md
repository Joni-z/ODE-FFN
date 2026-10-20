# Follow-up stage: complete and audited

The scientific analysis is complete. [Report 1](../../reports/report1/report1.pdf)
and the [analysis bundle](analysis-report.md) incorporate the results.

| Stage | Completed evidence | Execution |
|---|---|---|
| Frozen order | 66 finite cells, 66 sample hashes, 18 native checks | 145, one GPU, 114.5 seconds |
| Matrix controls | 17 matrices × 3 durations × 5 modes = 255 evaluations | CPU |
| Ring8 solver check | 48 cells, 48 sample hashes; all corresponding Heun32 arrays reproduced exactly | 145, one GPU, 279.4 seconds |

All training checkpoints use seed 0 and update 5,000. No new training was added.
Source, data and checkpoint hashes were checked before and after GPU evaluation.
The original torch Slurm job **20405164** was cancelled while pending with zero
runtime before the standalone run. There is no pending duplicate experiment.

## Results

Low-order exact substitutions worsen generation in all six K1/K2 target pairs.
K8 is near-exponential on these evaluations, but Ring8 frozen K4/K8 substitutions
still change SW2 by about 3%; the direction persists at 256 Heun steps.
Controlled matrices show that eigenvalues and spectral norm do not determine
finite-time gain. See the complete matrices and qualified interpretation in
`analysis-report.md`, not selected best cells.

## Resources and provenance

One RTX 6000 Ada was used. The initial short concurrency benchmark selected one
worker because two improved measured throughput by only 8%, below the predefined
10% threshold. It is not a robust speed benchmark.

The frozen-order trace has 11 ten-second observations: mean utilization 57.4%,
median 82%, range 0–89%, including startup and phase boundaries. The longer solver
check has 28 observations: mean 83.9%, median 94%, range 0–94%; maximum sampled
GPU memory was 832 MiB. These are sampled observations, not exact time integrals.
Both GPU computations have finished; plotting and report work run locally on CPU.

- `collection-audit.json`: local verification of the full first-stage archive.
- `solver-check-audit.json`: local verification of all finer-resolution arrays.
- `manifest-145.json`: relocated frozen manifest, with original hashes retained.
- `gpu-usage-145.csv`, `solver-check-gpu-usage.csv`: full telemetry traces.
- `migration-145.md`: original queue, migration and connection history.
- `protocol.md`, `solver-check-protocol.md`: prespecified and post hoc scopes.

Raw output directories on 145:

```text
/data1/zz5070/ODE_FFN/frozen_order_20261020/runs/synthetic/stage_20261020_frozen_order_145
/data1/zz5070/ODE_FFN/frozen_order_20261020/runs/synthetic/stage_20261020_solver_check_145
```

Local full archives are outside the repository under
`/Users/mr.z/ode-workspace-prep/frozen-order-145-results` and
`/Users/mr.z/ode-workspace-prep/solver-check-145-results`.
