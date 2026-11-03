# Cycle10 execution commands

Runtime is7e394e296665a96d734316c26626506257da66da. Use separate cycle10
worktrees. Launch only after both fresh native witnesses pass and GPU0 is
free. No package change.

```bash
ssh 145 'CUDA_VISIBLE_DEVICES=GPU-5d6078be-3b77-0dfd-0e6b-9cb486e62ff4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data2/zz5070/miniconda3/envs/py312/bin/python /data2/zz5070/ODE_FFN/professor-autoresearch-cycle10-20261103/repo/studies/muon_pd_20261102/run_queue.py --host 145 --device cuda --data-root /data2/zz5070/ODE_FFN/muon-pd-20261102/data --output /data2/zz5070/ODE_FFN/professor-autoresearch-cycle10-20261103/output --workers 2 --detach --grid /data2/zz5070/ODE_FFN/professor-autoresearch-cycle10-20261103/repo/studies/muon_pd_20261102/autoresearch_20261103/cycle10/grid.json --runner /data2/zz5070/ODE_FFN/professor-autoresearch-cycle10-20261103/repo/studies/muon_pd_20261102/autoresearch_20261103/cycle10/trial10.py --experiment-id professor_autoresearch_cycle10_20261103'
```

```bash
ssh lenovo 'set CUDA_VISIBLE_DEVICES=0&& set OMP_NUM_THREADS=1&& set OPENBLAS_NUM_THREADS=1&& set MKL_NUM_THREADS=1&& set PATH=D:\Research\ODE-FFN\tools\git\cmd;%PATH%&& D:/Research/ODE-FFN/envs/synthetic/Scripts/python.exe D:/Research/ODE-FFN/professor-autoresearch-cycle10-20261103/repo/studies/muon_pd_20261102/run_queue.py --host lenovo --device cuda --data-root D:/Research/ODE-FFN/datasets/synthetic --output D:/Research/ODE-FFN/professor-autoresearch-cycle10-20261103/output --workers 2 --detach --grid D:/Research/ODE-FFN/professor-autoresearch-cycle10-20261103/repo/studies/muon_pd_20261102/autoresearch_20261103/cycle10/grid.json --runner D:/Research/ODE-FFN/professor-autoresearch-cycle10-20261103/repo/studies/muon_pd_20261102/autoresearch_20261103/cycle10/trial10.py --experiment-id professor_autoresearch_cycle10_20261103'
```

Each host runs six registered cells, two workers on one free GPU0. Raw and
Swi64 references launch first; radial/static candidates and the two wider
ordinary references follow. The existing monitor persists status and GPU
utilization/memory/power every30s. Confirm increasing steps after SSH exits.
Do not retry a failed cell silently.

Before interpretation, replay all twelve final20k checkpoints with the native
evaluator and cycle10/trial10.py, and compare all eight native reference
states/saved arrays with their archived parents using cycle10/anchors.py.

Post-SSH90s witness:145native raw/Swi64 reached9500/18500,93%GPU and1183MiB;
Lenovo8000/8500,89%GPU and808MiB. Controllers2777039and21920 persist after
SSH exits; no failures. These are resource/readiness observations, not quality
results. Final interpretation awaits the full registered comparison.

Closed: both queues6/6complete, no pending/running cells or failures. Native
endpoint replay and four exact final20k anchor checks pass perhost; all saved
generated arrays reproduce exactly. Final results and full source/receipt
bindings are in [cycle10-results](cycle10-results/findings.md). GPU0returned
to its idle baseline on both hosts. No new training queue is active.
