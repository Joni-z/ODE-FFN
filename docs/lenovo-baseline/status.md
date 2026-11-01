# Lenovo baseline status

Snapshot: 2026-11-01T12:50:23.240928+00:00. Re-query before treating it as live.

The native SwiGLU baseline is **running** on Lenovo's RTX 4060 Laptop. Source is
frozen at `3b07a570f0742c7e4c85cbadde475a93716656b6` in
`D:/Research/ODE-FFN/runtimes/jit_cifar10_swiglu_20261101`.
The supervisor calls the upstream main_jit.py; it does not implement training.

- JiT-B/2: 129,881,868 parameters; unconditional CIFAR-10; seed 0.
- Batch 8, 200 epochs, 6,250 updates/epoch; LR 1e-4 after 5 warmup epochs.
- Confirmed progress:epoch 0, 560/6,250 updates, with finite losses.
- Steady throughput about 0.19 s/update after worker startup.
- GPU 99%, 6,325 MiB resident, 61 C, 74.76 W at the saved observation.
- Exact command, process IDs and hashes:[launch-snapshot.json](launch-snapshot.json).
- Windows task:ODEFFN_JiT_CIFAR10_SwiGLU_20261101.
- Outputs:D:/Research/ODE-FFN/runs/jit_cifar10_swiglu_20261101.
- train.log:native loss/LR/progress; output/:TensorBoard and checkpoints;
  status.json/monitor.jsonl:30-second process/GPU observations;
  process-exit.json:written only when the training child exits.

A full latest checkpoint is scheduled at each epoch boundary. This snapshot
precedes the first checkpoint. First FID5k/sample evaluation: after epoch 20.
Training-only estimate at 0.19 s/update: 20 minutes/epoch, 6.6 hours to epoch 20,
and 66 hours for 200 epochs. Evaluation adds time. These are estimates, and may
change with power, temperature, sleep or competing workloads.

No new FID exists. The acceptance gate remains **FID50k below 50** on a clearly
identified checkpoint, with inspected samples. FID5k progress curves and
readiness do not establish ISRU effectiveness. See [protocol.md](protocol.md).

## Inspect without restarting

```powershell
Get-Content D:/Research/ODE-FFN/runs/jit_cifar10_swiglu_20261101/status.json
Get-Content D:/Research/ODE-FFN/runs/jit_cifar10_swiglu_20261101/train.log -Tail 8
Get-ScheduledTask -TaskName ODEFFN_JiT_CIFAR10_SwiGLU_20261101
```

The task uses the signed-in jonat session and prevents idle sleep while active.
Logout, closing the machine, reboot or power loss can interrupt training.
There is no automatic restart. Inspect status/exit/checkpoint before explicit
recovery; do not rerun the monitor against its existing output directory.
The scheduler has no 72-hour limit; the native 200-epoch budget ends training.

Old Windows edits/runs, Mac experiments and both report folders were preserved.
Report1 remains final. Active-checkout updates cannot change this detached run.
