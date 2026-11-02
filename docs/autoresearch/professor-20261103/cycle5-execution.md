# Cycle5 execution

Frozen runtime f81663abae37db0efb3d022b0d9ec259a08a1f64. Earlier76 autoresearch trajectories and12 parity cases are complete/replayed. CPU and fresh documented CUDA witnesses passed on both hosts, including actual8-to16 continuation.14 new trajectories: ordinary bilinear64 references first, then absK1 candidates; two workers per host. Retain all SwiGLU and constant-FFN controls.

## 145

```bash
ssh 145 'CUDA_VISIBLE_DEVICES=GPU-5d6078be-3b77-0dfd-0e6b-9cb486e62ff4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data2/zz5070/miniconda3/envs/py312/bin/python /data2/zz5070/ODE_FFN/professor-autoresearch-cycle5-20261103/repo/studies/muon_pd_20261102/run_queue.py --host 145 --device cuda --data-root /data2/zz5070/ODE_FFN/muon-pd-20261102/data --output /data2/zz5070/ODE_FFN/professor-autoresearch-cycle5-20261103/output --workers 2 --detach --grid /data2/zz5070/ODE_FFN/professor-autoresearch-cycle5-20261103/repo/studies/muon_pd_20261102/autoresearch_20261103/cycle5/grid.json --runner /data2/zz5070/ODE_FFN/professor-autoresearch-cycle5-20261103/repo/studies/muon_pd_20261102/autoresearch_20261103/cycle5/trial3.py --experiment-id professor_autoresearch_cycle5_20261103'
```

## lenovo

```bash
ssh lenovo 'set CUDA_VISIBLE_DEVICES=0&& set OMP_NUM_THREADS=1&& set OPENBLAS_NUM_THREADS=1&& set MKL_NUM_THREADS=1&& set PATH=D:\Research\ODE-FFN\tools\git\cmd;%PATH%&& D:/Research/ODE-FFN/envs/synthetic/Scripts/python.exe D:/Research/ODE-FFN/professor-autoresearch-cycle5-20261103/repo/studies/muon_pd_20261102/run_queue.py --host lenovo --device cuda --data-root D:/Research/ODE-FFN/datasets/synthetic --output D:/Research/ODE-FFN/professor-autoresearch-cycle5-20261103/output --workers 2 --detach --grid D:/Research/ODE-FFN/professor-autoresearch-cycle5-20261103/repo/studies/muon_pd_20261102/autoresearch_20261103/cycle5/grid.json --runner D:/Research/ODE-FFN/professor-autoresearch-cycle5-20261103/repo/studies/muon_pd_20261102/autoresearch_20261103/cycle5/trial3.py --experiment-id professor_autoresearch_cycle5_20261103'
```

Status: live and independently verified after SSH exit. Two snapshots60s apart show145both bilinear Adam jobs5500->18500/19000 and Lenovo5500->18000/17500.145GPU0utilization91%/1175MiB, Lenovo90%/796MiB.145controller2048269/CUDA2048270,2048271; Lenovo launcher23540/controller4112/actualCUDA9744,15312. Both numerical worktrees stay clean at f81663a, no failures. Receipts in readiness/cycle5/. Persistent monitor: output/monitor/status.json and history.jsonl. No silent retries; failures stop new dispatch. Numerical worktrees remain immutable.
