# Wave2 execution record

Thirty registered seed-0 cells: fifteen Product16 cells on145 and fifteen Ring8
cells onLenovo. Each runs20,000 updates with batch512. Reuse all audited first-wave
controls. Two workers share one GPU on each host; no other GPU or image job is used.

## Frozen runtime

Both new worktrees use `ec5d2b14e127726917d6f000c289593ea3614f51`.
The original first-wave worktrees and outputs remain intact. GitHub main may
advance for analysis/docs; neither live numerical worktree will be updated.

| Host | New study root | Python | GPU |
|---|---|---|---|
|145|`/data2/zz5070/ODE_FFN/muon-pd-wave2-20261102`|`/data2/zz5070/miniconda3/envs/py312/bin/python`, torch2.8.0+cu128|RTX6000Ada, GPU5, `GPU-ce799976-7805-c539-9913-3bbadcdbadbf`|
|Lenovo|`D:/Research/ODE-FFN/muon-pd-wave2-20261102`|`D:/Research/ODE-FFN/envs/synthetic/Scripts/python.exe`, torch2.8.0+cu126|RTX4060Laptop, GPU0|

`repo/` holds code; `output/` holds formal results and `output/monitor/` holds
queue status, history, PIDs, resource samples and console logs. Raw checkpoints,
data and execution logs are excluded from Git. Training data are the existing
hash-verified banks, not newly generated data.

## Readiness

CPU witnesses passed on both hosts. The initial CUDA native runs found a real
inherited restore error: RNG bytes mapped to CUDA could not be restored. The
minimal `.cpu()` correction is recorded in `reference.patch` and its hash manifest.
Failed smoke output is preserved; corrected witnesses use `native-smoke-rngfix`.
See [exact validation commands](validation.md). Both native CUDA witnesses passed at the frozen runtime commit, including
actual8-to16 checkpoint continuation. The [145 receipt](readiness/145-cuda.json)
and [Lenovo receipt](readiness/lenovo-cuda.json) retain the exact check results.

## Registered launch commands

145:

```bash
ssh 145 'CUDA_VISIBLE_DEVICES=GPU-ce799976-7805-c539-9913-3bbadcdbadbf OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data2/zz5070/miniconda3/envs/py312/bin/python /data2/zz5070/ODE_FFN/muon-pd-wave2-20261102/repo/studies/muon_pd_20261102/run_queue.py --host 145 --device cuda --data-root /data2/zz5070/ODE_FFN/muon-pd-20261102/data --output /data2/zz5070/ODE_FFN/muon-pd-wave2-20261102/output --workers 2 --detach --grid /data2/zz5070/ODE_FFN/muon-pd-wave2-20261102/repo/studies/muon_pd_20261102/wave2/grid.json --runner /data2/zz5070/ODE_FFN/muon-pd-wave2-20261102/repo/studies/muon_pd_20261102/wave2/train_wave2.py --experiment-id muon_pd_wave2_20261102'
```

Lenovo:

```bash
ssh lenovo 'set OMP_NUM_THREADS=1&& set OPENBLAS_NUM_THREADS=1&& set MKL_NUM_THREADS=1&& set CUDA_VISIBLE_DEVICES=0&& set PATH=D:\Research\ODE-FFN\tools\git\cmd;%PATH%&& D:\Research\ODE-FFN\envs\synthetic\Scripts\python.exe D:\Research\ODE-FFN\muon-pd-wave2-20261102\repo\studies\muon_pd_20261102\run_queue.py --host lenovo --device cuda --data-root D:/Research/ODE-FFN/datasets/synthetic --output D:/Research/ODE-FFN/muon-pd-wave2-20261102/output --workers 2 --detach --grid D:/Research/ODE-FFN/muon-pd-wave2-20261102/repo/studies/muon_pd_20261102/wave2/grid.json --runner D:/Research/ODE-FFN/muon-pd-wave2-20261102/repo/studies/muon_pd_20261102/wave2/train_wave2.py --experiment-id muon_pd_wave2_20261102'
```

The controller survives SSH exit, records a heartbeat every30s and polls child
exits every2s. A failed cell stops subsequent dispatch. No automatic retry or
unregistered sweep is enabled. After dispatch, verify PIDs and increasing metrics
through separate SSH connections; a printed launch PID alone is insufficient.

## Completed status

Both queues completed15/15 with exit0 and no pending cells. All30 final checkpoints passed sample replay and independent SW2 recalculation. See [findings](results/findings.md). The original launch snapshot below is historical; neither queue is still training.

## Launch snapshot

Both formal queues are running. Separate SSH observations confirmed increasing
steps after the launching connections exited. Both report `failed=false` and
retain the frozen runtime commit. Before dispatch both worktrees were clean;
145 had15GiB free on/data2 and the selected GPUs had no compute jobs.

| Host | Controller | Two observed training progressions | GPU at second check |
|---|---|---|---|
|145|1881221, parent1|Both1,500 to10,500 updates over120.7s|GPU5,92%,1,195MiB|
|Lenovo|440, launcher16520|Both1,500 to9,500 updates over120.2s|GPU0,90% heartbeat/94% direct query,816MiB|

145 CUDA trainer PIDs are1881222/1881223. Lenovo CUDA PIDs are16108/13180,
under venv wrappers17112/10464. Each host had two active cells and thirteen
pending at that observation. This snapshot demonstrates execution, not a new
quality result. Full receipts: [145](readiness/launch-145.json) and
[Lenovo](readiness/launch-lenovo.json).

Next: let the registered queues complete, then replay endpoints and compare all
three intervention blocks with their audited first-wave controls. Do not update
a running numerical checkout or add another sweep automatically.
