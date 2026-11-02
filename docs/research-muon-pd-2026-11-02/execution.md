# Execution record

Code snapshot: **295dd3e** (GitHub main). Formal runtime checkouts stay pinned
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

## Lenovo placement pending

14 Ring8 cells will stay on one selected device for within-task pairing. A
two-worker, 1000-update CPU/CUDA placement witness runs in separate directories.
Only elapsed time is consulted; its field/sample metrics are not selection
evidence and it does not replace any formal 20k cell. Record the selected device
and launch command here after timing completes.

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
