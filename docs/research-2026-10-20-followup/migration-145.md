# Standalone GPU migration preparation

The user requested checking host `145` as an alternative to the queued torch job. The evaluation inputs have been copied to:

```text
/data1/zz5070/ODE_FFN/frozen_order_20261020
```

The scientific runtime, 18 checkpoint/configuration pairs, and prepared data are preserved. The target manifest changes absolute paths and records the SHA-256 of the original torch manifest. Source, checkpoint, configuration, and data hashes passed verification on 145; all four CPU intervention tests passed.

The existing environment is Python 3.12.11, PyTorch 2.8.0+cu128, NumPy 1.26.4, and SciPy 1.17.1. This differs from the torch environment. Native checkpoint reproduction and same-order float64 precision checks remain mandatory; the migration is not itself evidence of numerical agreement or a cross-hardware speed comparison.

## Resource and execution state

Host 145 has eight RTX 6000 Ada GPUs with 48 GB each. Initially every GPU had an active process. The user first requested waiting for an empty GPU, then supplied authorization from the process owner and their stop helper. Exactly one owner process, on GPU 0, was stopped through that helper. The immediate check showed the GPU's memory falling to 2 MiB. Other processes were not stopped by this workflow.

The standalone stage uses only one GPU, a two-hour runtime limit, the existing one-versus-two-worker throughput comparison, and ten-second GPU telemetry. It preserves all 66 evaluation cells and seed 0.

At the latest preparation checkpoint, **GPU evaluation has not started**. The coordinator could not hold the original Slurm job: `scontrol hold 20405164` returned an unspecified error while the job remained pending. The coordinator was stopped without starting the target evaluation. Subsequently the SSH connection to 145 ended and a new connection returned `Permission denied`.

Torch job `20405164` has not been cancelled. Restore SSH access, recheck GPU availability, and run the numerical GPU preflight before selecting the final execution route. Do not launch duplicate evaluations. The Slurm job must be reconciled before the full standalone launch; the original frozen runtime must not be edited.

The torch queue also briefly reported `QOSGrpGRES`, which identifies an aggregate generic-resource limit in its QoS. A later `Reason=None` does not negate that observation; Slurm documents that reasons can temporarily revert to None while awaiting another scheduling pass. See the [official reason-code reference](https://slurm.schedmd.com/job_reason_codes.html).

## Controller

`scripts/migrate_when_idle.py` provides read-only probing, detached standalone execution with an idle-GPU check, a completion monitor, and a hold-based migration path for schedulers that accept that operation. Five local mocked routing tests pass, including rejection of an occupied GPU despite low utilization, preventing a target launch when the original job is running, and preventing any target launch after a rejected hold. No GPU or Slurm resources are used by those tests.

The target config is [migration-145-config.json](migration-145-config.json). The local monitor directory is ignored by Git. The remote controller copy predates the subsequent error-reporting and monitor-mode fixes and must be refreshed when SSH access is restored.

For storage figures and the scope of the account audit, see [storage.md](storage.md).
