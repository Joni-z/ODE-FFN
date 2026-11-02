# Execution record

Numerical code snapshot: **295dd3e** (GitHub main). Formal runtime checkouts stay pinned
while documentation updates may advance main. Model seed 0; fixed grid in the
registered protocol. Raw artifacts and environments remain outside Git.

## Readiness

A fresh agent executed all four documented native witnesses. See `verification/`:

| Host/device | Scalar AdamW | Scalar Muon | PD AdamW | Wide SwiGLU Muon |
|---|---:|---:|---:|---:|
| 145 CPU | 14.50 ms | 19.87 ms | 14.04 ms | 18.85 ms |
| 145 CUDA | 8.52 ms | 12.21 ms | 8.37 ms | 8.46 ms |
| Lenovo CPU | 7.14 ms | 9.54 ms | 7.46 ms | 8.84 ms |
| Lenovo CUDA | 7.57 ms | 11.00 ms | 7.28 ms | 7.49 ms |

Each timing uses 15 warm-up and 120 measured updates, one CPU thread. Polynomial
error is at most 2.50e-16. Original forward, common initialization, full gate
derivatives, exact official optimizer updates and physical checkpoint resume all
passed. 145 kernel witnesses were at 4b15145; the numerical source is unchanged
at 295dd3e, which adds invocation provenance. Lenovo witnesses use 295dd3e.

Both hosts then passed three 8-update native-trainer smoke runs: scalar Muon,
PD AdamW and SwiGLU AdamW, on Product16 for 145 and Ring8 for Lenovo. Each produced
initial/final metrics, update spectra, two checkpoints and ADAPTER_COMPLETE.
These are execution checks, not model-quality results.

Lenovo's direct GitHub clone stalled. Only that newly created clone process was
stopped. Its incomplete folder is preserved as `repo-incomplete-fetch`. A fresh
clone of the existing committed local upstream base plus a published incremental
Git bundle reconstructed the exact source commit; origin points to GitHub.
No historical worktree was reset. Windows line endings were restored from exact
Git blobs and its index refreshed with `git add --renormalize` (empty staged diff).
Raw-byte source checks then passed; the verifier was not weakened.

## 145 launch

Use free GPU5 UUID GPU-ce799976-7805-c539-9913-3bbadcdbadbf, two workers.
22 cells: Product16 and anisotropic Gaussian. Other users' GPUs are untouched.

```bash
ssh 145 'cd /data2/zz5070/ODE_FFN/muon-pd-20261102/repo && CUDA_VISIBLE_DEVICES=GPU-ce799976-7805-c539-9913-3bbadcdbadbf OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data2/zz5070/miniconda3/envs/py312/bin/python studies/muon_pd_20261102/run_queue.py --host 145 --device cuda --data-root /data2/zz5070/ODE_FFN/muon-pd-20261102/data --output /data2/zz5070/ODE_FFN/muon-pd-20261102/output --workers 2 --detach'
```

145 output: `/data2/zz5070/ODE_FFN/muon-pd-20261102/output`.
Data are copies of the registered archived banks; manifest and individual file
hashes are checked before every cell. Approximately 20 GiB were available before
the study; expected study output is below 2 GiB.

## Lenovo launch

Use RTX4060 GPU0, two workers. All 14 Ring8 cells remain on CUDA. Two-worker
1000-update placement took 17.14s with CUDA versus19.29s with CPU, including
startup and evaluation. GPU sampling was also faster (0.125s versus1.188s for
the scalar-Muon endpoint). This timing comparison, not model quality, selected
CUDA. It is a short workload estimate and does not guarantee an identical gain
throughout the formal study. Raw timing summary is in verification/.

```bash
ssh lenovo 'set OMP_NUM_THREADS=1&& set OPENBLAS_NUM_THREADS=1&& set MKL_NUM_THREADS=1&& set CUDA_VISIBLE_DEVICES=0&& set PATH=D:\Research\ODE-FFN\tools\git\cmd;%PATH%&& D:\Research\ODE-FFN\envs\synthetic\Scripts\python.exe D:\Research\ODE-FFN\muon-pd-20261102\repo\studies\muon_pd_20261102\run_queue.py --host lenovo --device cuda --data-root D:/Research/ODE-FFN/datasets/synthetic --output D:/Research/ODE-FFN/muon-pd-20261102/output --workers 2 --detach'
```

Output: `D:/Research/ODE-FFN/muon-pd-20261102/output`. Archived datasets are
read in place with exact manifest/file-hash validation. Existing runs remain
untouched. The shorter placement runs live in separate placement-cpu/cuda folders
and do not replace any formal cell.

## Monitoring

Each detached controller writes `output/monitor/status.json` and an append-only
`history.jsonl` every 30 seconds, including child PIDs, most recent steps/metrics,
exits, pending cells, GPU memory/utilization/power and the GPU binding. Per-cell
stdout is retained. Failed cells are not automatically retried; a failure stops
dispatching new cells while existing children finish. The lock remains as an
execution record. ADAPTER_COMPLETE requires both training and prescribed final
test sampling; the inherited COMPLETE flag alone is insufficient.

Watch commands:

```bash
ssh 145 'cat /data2/zz5070/ODE_FFN/muon-pd-20261102/output/monitor/status.json'
ssh lenovo 'type D:\Research\ODE-FFN\muon-pd-20261102\output\monitor\status.json'
```

No quality conclusion is available at registration or readiness.

## Windows background-process correction

The first Lenovo detached-controller attempt returned PID28548 but exited with
the SSH session before creating a queue lock, child training process or status
file. It is not a training trajectory. The controller now sets Windows
CREATE_BREAKAWAY_FROM_JOB in addition to DETACHED_PROCESS and a new process
group. A separate child witness completed its five-second delayed file write
after the launching SSH command had returned, establishing persistence. This
changes process lifetime only; model, optimizer, data and numerical kernels are
unchanged. Lenovo uses the corrected controller commit recorded with launch
verification;145 keeps its existing running snapshot.

## Confirmed running state

Independent fresh-SSH observations separated by60.34s (145) and90.22s (Lenovo)
confirmed increasing steps and live CUDA processes after launch connections
closed. See `verification/ssh-persistence.json` and timestamped `launch.json`.

- 145: snapshot295dd3e, controller1156083 (parent1), GPU5;22 registered cells.
- Lenovo: snapshotae4560e, actual controller25028 under venv launcher21296,
  GPU0;14 registered cells. The extra Windows venv process layer is expected.
- At2026-11-02 07:21UTC,145 had2 completed cells and2 running; Lenovo had2
  running. Both controllers reported failed=false. GPU snapshots were91% and90%
  utilization, with1189MiB and792MiB used. Utilization is a point observation.
- No trained-model conclusion is made from the first ODE-only endpoints while
  the corresponding ordinary controls remain pending. The fixed grid does not
  launch unregistered follow-ups.

The root image trainer and both report PDFs are unchanged. Report1 SHA256
165d46476362f006db6124f0035bcb6964cb163e509f52eb49adbf4de5839ecf;
Report2 SHA25682c97fbcb1a073043a8fb4258a1c499468d464a421437750fe92214b97b67648.
