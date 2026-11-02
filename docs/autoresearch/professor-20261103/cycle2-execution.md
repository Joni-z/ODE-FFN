# Cycle2 execution

Frozen runtime d26ba5ace85f2617f2d614305f32ef6e225bc2fe. Separate worktrees and outputs preserve completed cycle1. All18 cycle1 checkpoint replays and six anchor reruns passed before registration. CPU and fresh documented CUDA witnesses passed for the three cycle2 candidates, including actual8-to16 continuation.

## Exact dispatch

### 145

```bash
ssh 145 'CUDA_VISIBLE_DEVICES=GPU-5d6078be-3b77-0dfd-0e6b-9cb486e62ff4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data2/zz5070/miniconda3/envs/py312/bin/python /data2/zz5070/ODE_FFN/professor-autoresearch-cycle2-20261103/repo/studies/muon_pd_20261102/run_queue.py --host 145 --device cuda --data-root /data2/zz5070/ODE_FFN/muon-pd-20261102/data --output /data2/zz5070/ODE_FFN/professor-autoresearch-cycle2-20261103/output --workers 2 --detach --grid /data2/zz5070/ODE_FFN/professor-autoresearch-cycle2-20261103/repo/studies/muon_pd_20261102/autoresearch_20261103/cycle2/grid.json --runner /data2/zz5070/ODE_FFN/professor-autoresearch-cycle2-20261103/repo/studies/muon_pd_20261102/autoresearch_20261103/cycle2/trial2.py --experiment-id professor_autoresearch_cycle2_20261103'
```

### lenovo

```bash
ssh lenovo 'set CUDA_VISIBLE_DEVICES=0&& set OMP_NUM_THREADS=1&& set OPENBLAS_NUM_THREADS=1&& set MKL_NUM_THREADS=1&& set PATH=D:\Research\ODE-FFN\tools\git\cmd;%PATH%&& D:/Research/ODE-FFN/envs/synthetic/Scripts/python.exe D:/Research/ODE-FFN/professor-autoresearch-cycle2-20261103/repo/studies/muon_pd_20261102/run_queue.py --host lenovo --device cuda --data-root D:/Research/ODE-FFN/datasets/synthetic --output D:/Research/ODE-FFN/professor-autoresearch-cycle2-20261103/output --workers 2 --detach --grid D:/Research/ODE-FFN/professor-autoresearch-cycle2-20261103/repo/studies/muon_pd_20261102/autoresearch_20261103/cycle2/grid.json --runner D:/Research/ODE-FFN/professor-autoresearch-cycle2-20261103/repo/studies/muon_pd_20261102/autoresearch_20261103/cycle2/trial2.py --experiment-id professor_autoresearch_cycle2_20261103'
```

Status: dispatched.145controller1524877 and Lenovo launcher27908; first post-SSH observations show increasing steps with source d26ba5a, no failures, GPU92%/90%. Two separated post-SSH observations passed: over60s145Adam/Muon6500->13500 and4500->9000; Lenovo5500->12000 and3500->8000. Real controllers1524877/28456; no failures, source remains d26ba5a. Receipts in readiness/cycle2/. Monitor:output/monitor/status.json,history.jsonl, per-cell console logs. Failure stops new dispatch; no automatic retry.
