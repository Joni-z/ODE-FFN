# Cycle1 execution

Runtime01fe2df8624d4e20c8f24cd765fee526b51960f3, separate immutable worktrees. CPU and fresh documented CUDA witnesses passed, including actual8-to16 native resume for both candidates. Existing synthetic environments reused. Two workers share each selected GPU; all other jobs are preserved.

## Exact dispatch commands

### 145

```bash
ssh 145 'CUDA_VISIBLE_DEVICES=GPU-5d6078be-3b77-0dfd-0e6b-9cb486e62ff4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data2/zz5070/miniconda3/envs/py312/bin/python /data2/zz5070/ODE_FFN/professor-autoresearch-20261103/repo/studies/muon_pd_20261102/run_queue.py --host 145 --device cuda --data-root /data2/zz5070/ODE_FFN/muon-pd-20261102/data --output /data2/zz5070/ODE_FFN/professor-autoresearch-20261103/output --workers 2 --detach --grid /data2/zz5070/ODE_FFN/professor-autoresearch-20261103/repo/studies/muon_pd_20261102/autoresearch_20261103/grid.json --runner /data2/zz5070/ODE_FFN/professor-autoresearch-20261103/repo/studies/muon_pd_20261102/autoresearch_20261103/trial.py --experiment-id professor_autoresearch_20261103'
```

### lenovo

```bash
ssh lenovo 'set CUDA_VISIBLE_DEVICES=0&& set OMP_NUM_THREADS=1&& set OPENBLAS_NUM_THREADS=1&& set MKL_NUM_THREADS=1&& set PATH=D:\Research\ODE-FFN\tools\git\cmd;%PATH%&& D:/Research/ODE-FFN/envs/synthetic/Scripts/python.exe D:/Research/ODE-FFN/professor-autoresearch-20261103/repo/studies/muon_pd_20261102/run_queue.py --host lenovo --device cuda --data-root D:/Research/ODE-FFN/datasets/synthetic --output D:/Research/ODE-FFN/professor-autoresearch-20261103/output --workers 2 --detach --grid D:/Research/ODE-FFN/professor-autoresearch-20261103/repo/studies/muon_pd_20261102/autoresearch_20261103/grid.json --runner D:/Research/ODE-FFN/professor-autoresearch-20261103/repo/studies/muon_pd_20261102/autoresearch_20261103/trial.py --experiment-id professor_autoresearch_20261103'
```

Status: running, post-SSH continuation verified.145 controller1429857, CUDA1429858/1429859; Lenovo controller30072 under launcher29944, CUDA18820/3108. Over60s the PD/scalar anchors advanced2500->9000 and2000->6500 on145,2500->8500 and1500->5500 onLenovo. GPU92%/91%; no failures. Source remains01fe2df. Monitor is output/monitor/{status.json,history.jsonl}; failures stop dispatch, never silently retry.

## Completed cycle1

Both hosts completed9/9 with no failures. All18 checkpoints replayed; six anchors are exact reruns. See cycle1-results/findings.md. The earlier live launch snapshot is historical. Cycle2 uses a separate worktree/output and registration.
