# Lenovo baseline status

**Paused at the user's request on 1 November 2026, 22:09 Asia/Shanghai.**
Four epochs (25,000 updates) are preserved in a verified full checkpoint.
The original training process tree was stopped after saving; profiling updates
were discarded. All six measurements are complete; the fastest tested setup
still projects to60hours for200epochs. The long run remains paused.
See [pause.json](pause.json) and [performance.md](performance.md).
The observation below predates this pause.

Snapshot: 2026-11-01T13:46:59.718850+00:00. Re-query before treating it as live.

At the earlier snapshot, the native SwiGLU baseline was **running** on Lenovo's RTX 4060 Laptop. Source is
frozen at `3b07a570f0742c7e4c85cbadde475a93716656b6` in
`D:/Research/ODE-FFN/runtimes/jit_cifar10_swiglu_20261101`.
The supervisor calls the upstream main_jit.py; it does not implement training.

- JiT-B/2: 129,881,868 parameters; unconditional CIFAR-10; seed 0.
- Batch 8, 200 epochs, 6,250 updates/epoch; LR 1e-4 after 5 warmup epochs.
- Confirmed progress: epoch index 2 (third epoch), 5,540/6,250 updates in the
  monitor log tail; about 18,041 total updates. Recent loss 0.1559.
- Median throughput over the last 100 printed rows: 0.1926 s/update, about
  41.5 images/s at batch 8. Completed epochs took 20:18 and 20:14.
- GPU 99%, 6,346 MiB resident, 63 C, 79.90 W at the saved observation.
- Exact command, process IDs and hashes:[launch-snapshot.json](launch-snapshot.json).
- Windows task:ODEFFN_JiT_CIFAR10_SwiGLU_20261101.
- Outputs:D:/Research/ODE-FFN/runs/jit_cifar10_swiglu_20261101.
- train.log:native loss/LR/progress; output/:TensorBoard and checkpoints;
  status.json/monitor.jsonl:30-second process/GPU observations;
  process-exit.json:written only when the training child exits.

The latest full checkpoint exists (2,596,444,635 bytes) and is saved at each
epoch boundary. No error or FID line was present in the copied log.
First FID5k/sample evaluation: after 20 completed epochs, approximately 5.7
training hours after this snapshot. At 0.1926 s/update, 200 epochs require about
67 training hours in total; about 66 remain. Evaluation adds time. These are estimates, and may
change with power, temperature, sleep or competing workloads.

No new FID exists. The acceptance gate remains **FID50k below 50** on a clearly
identified checkpoint, with inspected samples. FID5k progress curves and
readiness do not establish ISRU effectiveness. See [protocol.md](protocol.md).

GPU display: see [gpu-monitor.md](gpu-monitor.md) for the verified `ssh lenovo`
then `nvitop` workflow.

See [the runtime audit](runtime-audit.md): the 67-hour estimate is supported by
throughput, but the local batch/LR/200-epoch CIFAR recipe remains unvalidated.
No budget change or speedup was applied.

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
