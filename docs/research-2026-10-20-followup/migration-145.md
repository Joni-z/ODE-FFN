# Standalone GPU migration and completion

The user requested checking host `145` as an alternative to the queued torch job. The evaluation inputs have been copied to:

```text
/data1/zz5070/ODE_FFN/frozen_order_20261020
```

The scientific runtime, 18 checkpoint/configuration pairs, and prepared data are preserved. The target manifest changes absolute paths and records the SHA-256 of the original torch manifest. Source, checkpoint, configuration, and data hashes passed verification on 145; all four CPU intervention tests passed.

The existing environment is Python 3.12.11, PyTorch 2.8.0+cu128, NumPy 1.26.4, and SciPy 1.17.1. This differs from the torch environment. A [numerical GPU preflight](gpu-preflight-145.json) reproduced the Gaussian K2 checkpoint's SW2: archived 0.06743679195642471, native on 145 0.06743678450584412. The same-order float64 action returned the same SW2, and an exact-action forward pass was finite. All 18 native reproduction checks and same-order precision checks remained required during the stage. These controls do not make this a cross-hardware speed comparison.

## Resource and execution state

Host 145 has eight RTX 6000 Ada GPUs with 48 GB each. Initially every GPU had an active process. The user first requested waiting for an empty GPU, then supplied authorization from the process owner and their stop helper. Only GPU 0 was reclaimed through that helper. When access was later restored, two new processes occupied the same GPU; their owner was verified before stopping them through the authorized helper. No processes on other GPUs were stopped by this workflow.

The standalone stage uses only one GPU, a two-hour runtime limit, the existing one-versus-two-worker throughput comparison, and ten-second GPU telemetry. It preserves all 66 evaluation cells and seed 0.

Evaluation **completed successfully** from 09:17:38.867 to 09:19:33.348 UTC on 20 October, approximately 114.5 seconds. The runner exited 0, all three dataset workers exited 0, all 66 cells were present, and the remote summary reported completion. The frozen launcher verifies input/source hashes before and after execution and verifies all 66 saved sample hashes before setting its completion flag. See [completion-145.json](completion-145.json).

The short exact-action microbenchmark measured 26,170.5 examples/s with one worker and 28,273.7 with two, an 8.0% gain. The preset 10% rule selected one worker. This sub-second benchmark is a routing measurement, not a robust training-throughput result. GPU utilization was sampled every ten seconds; observed samples reached 87–89%, while other samples were lower. The full trace is now in `gpu-usage-145.csv`; mean sampled utilization was 57.4% and median 82%, including startup and phase boundaries.

The original `scontrol hold` attempt had failed, so the automatic hold-based migration was stopped. Access was later recovered through the MBP's existing SSH configuration. After the GPU numerical preflight passed, the still-pending torch job was cancelled and its zero runtime verified before launching the full standalone stage. Thus the original Slurm job and the standalone stage did not execute simultaneously. For this standalone run, successful runner/worker exit and the frozen launcher's artifact checks replace the protocol's Slurm-exit requirement; scientific settings were unchanged.

The completion monitor captured successful exit before the gateway became unavailable. Collection subsequently completed: all 66 sample hashes, 18 native reproductions and manifest bindings passed the local audit. See `collection-audit.json` and `analysis-report.md`. The detached collector finished successfully.

The torch queue also briefly reported `QOSGrpGRES`, which identifies an aggregate generic-resource limit in its QoS. A later `Reason=None` does not negate that observation; Slurm documents that reasons can temporarily revert to None while awaiting another scheduling pass. See the [official reason-code reference](https://slurm.schedmd.com/job_reason_codes.html).

## Controller

`scripts/migrate_when_idle.py` provides read-only probing, detached standalone execution with an idle-GPU check, a completion monitor, an optional SSH gateway, and a hold-based migration path for schedulers that accept that operation. Six local mocked routing tests pass, including rejection of an occupied GPU despite low utilization, preventing a target launch when the original job is running, preventing any target launch after a rejected hold, and preserving command arguments through the gateway. No GPU or Slurm resources are used by those tests.

The target config is [migration-145-config.json](migration-145-config.json). The remote controller was refreshed before launch. The local monitor directory is ignored by Git. `scripts/collect_standalone_results.py` performs collection and the local artifact audit; its destination is `/Users/mr.z/ode-workspace-prep/frozen-order-145-results`.

For storage figures and the scope of the account audit, see [storage.md](storage.md).

## Focused solver follow-up

After reading the full results, a separately documented post hoc Ring8 check completed 48 cells at 32/64/128/256 Heun steps. Only GPU 0 was used; its current yp2841 owner was verified and PID 1177227 was stopped through the previously authorized helper. No other GPU jobs were stopped. The script and protocol were copied before execution, leaving the original frozen runtime unchanged. It ran from 11:43:59.587 to 11:48:38.994 UTC (279.4 seconds), with one worker, ten-second telemetry and a 30-minute timeout. All 48 array hashes passed locally; the 12 Heun32 score/array reproductions were exact. See `solver-check-audit.json` and `solver-check-submission.json`.
