# Lenovo Windows workspace

Workspace: `D:\Research\ODE-FFN` on `lenovo`, user `jonat`.

| Directory | Contents |
|---|---|
| `repo` | Current project source, Git history and the reviewed uncommitted continuation |
| `envs\synthetic` | Isolated Python 3.12 CUDA environment |
| `datasets\synthetic\v1` | Exact original dataset files, verified against their manifest |
| `runs\clock_20261024` | Focused experiment outputs, checkpoints and monitoring |
| `tools` | Managed Python and portable Git |
| `archives` | Transport archives and the Git bundle |
| `cache` | Installation cache and temporary files |
| `logs` | Launcher output and environment records |

The environment is for the unconditional synthetic ODELayer study. The native Windows path does not depend on Slurm, WSL, shell scripts or NCCL. See [protocol.md](protocol.md) for the scientific question, fixed settings, staged decision and metrics.

The first six-cell study is complete and audited. See [findings.md](findings.md), [comparison.csv](results/comparison.csv) and [audit.json](results/audit.json). The runner selected two CPU workers after measuring higher throughput than two CUDA workers for this small model. The CUDA environment remains ready for larger workloads.

Open a PowerShell terminal on Lenovo:

```powershell
$base = 'D:\Research\ODE-FFN'
$env:PATH = "$base\tools\git\cmd;" + $env:PATH
$env:OMP_NUM_THREADS = '1'
Set-Location "$base\repo"
& "$base\envs\synthetic\Scripts\python.exe" -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
Get-Content "$base\runs\clock_20261024\status.json"
Get-Content "$base\runs\clock_20261024\monitor.json"
```

The study supervisor writes `status.json`, `monitor.json`, `telemetry.jsonl`, `benchmark.json`, per-run logs, `decision.json` and `results.json`. Monitor records include GPU utilization/memory/temperature/power and the latest recorded training steps. Sampling and validation can temporarily lower training utilization.

`status.json` is authoritative for completion. `monitor.json` is the last periodic sample and can precede the final completed state.

The deployed background run is the on-demand Windows scheduled task `ODEFFN_clock_20261024`, using the signed-in `jonat` session. It survives an SSH command ending. Check its state with `Get-ScheduledTask -TaskName ODEFFN_clock_20261024`. The supervisor temporarily prevents idle sleep while it runs. Logging out or rebooting interrupts the job; snapshots are retained for explicit recovery.

For a **new** output directory, the foreground runner command is:

```powershell
& "$base\envs\synthetic\Scripts\python.exe" scripts/run_clock_study.py --data-root "$base\datasets\synthetic\v1" --output "$base\runs\clock_NEW"
```

Do not run that command a second time against the active output directory. The runner refuses to overwrite outputs. Each training cell saves optimizer/RNG states, configuration and source/dataset hashes. If interrupted, inspect its status and logs before resuming individual cells with the trainer's explicit `--resume` option.

For a new background task, use `powershell -ExecutionPolicy Bypass -File scripts/start_clock_study.ps1 -OutputName clock_NEW` from the repository. The bypass applies to this process only. Existing tasks/output directories are not overwritten. Stop the current study with `Stop-ScheduledTask -TaskName ODEFFN_clock_20261024`; do not kill unrelated Python processes.

Only synthetic data is deployed; ImageNet data and historical image checkpoints are not required for this experiment. Portable Git is project-local; the system Python and existing user environments are preserved.
