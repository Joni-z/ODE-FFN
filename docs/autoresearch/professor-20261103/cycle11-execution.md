# Cycle11 execution commands

Runtime e493b054e5e7726f91b61c852f0df9ad07f69941 is frozen. Use separate cycle11
worktrees and require both fresh documented CUDA passes before launch.

```bash
ssh 145 'CUDA_VISIBLE_DEVICES=GPU-5d6078be-3b77-0dfd-0e6b-9cb486e62ff4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data2/zz5070/miniconda3/envs/py312/bin/python /data2/zz5070/ODE_FFN/professor-autoresearch-cycle11-20261103/repo/studies/muon_pd_20261102/run_queue.py --host 145 --device cuda --data-root /data2/zz5070/ODE_FFN/muon-pd-20261102/data --output /data2/zz5070/ODE_FFN/professor-autoresearch-cycle11-20261103/output --workers 2 --detach --grid /data2/zz5070/ODE_FFN/professor-autoresearch-cycle11-20261103/repo/studies/muon_pd_20261102/autoresearch_20261103/cycle11/grid.json --runner /data2/zz5070/ODE_FFN/professor-autoresearch-cycle11-20261103/repo/studies/muon_pd_20261102/autoresearch_20261103/cycle11/trial11.py --experiment-id professor_autoresearch_cycle11_20261103'
```

```bash
ssh lenovo 'set CUDA_VISIBLE_DEVICES=0&& set OMP_NUM_THREADS=1&& set OPENBLAS_NUM_THREADS=1&& set MKL_NUM_THREADS=1&& set PATH=D:\Research\ODE-FFN\tools\git\cmd;%PATH%&& D:/Research/ODE-FFN/envs/synthetic/Scripts/python.exe D:/Research/ODE-FFN/professor-autoresearch-cycle11-20261103/repo/studies/muon_pd_20261102/run_queue.py --host lenovo --device cuda --data-root D:/Research/ODE-FFN/datasets/synthetic --output D:/Research/ODE-FFN/professor-autoresearch-cycle11-20261103/output --workers 2 --detach --grid D:/Research/ODE-FFN/professor-autoresearch-cycle11-20261103/repo/studies/muon_pd_20261102/autoresearch_20261103/cycle11/grid.json --runner D:/Research/ODE-FFN/professor-autoresearch-cycle11-20261103/repo/studies/muon_pd_20261102/autoresearch_20261103/cycle11/trial11.py --experiment-id professor_autoresearch_cycle11_20261103'
```

Each host runs five registered cells: two native references first, then the
three existing optimizer/rate settings of the missing controller/output corner.
Two workers/GPU0, one CPU thread/process, persistent30s GPU/step/PID monitoring.
Verify increasing steps after SSH exits. Finish all ten, replay with explicit
cycle11 experiment id/endpoint20000/count5 and exact-check two archived20k
references/host. Full factorial interpretation uses all eighteen hash-bound
archived corners; retain negative outcomes and the ordinary frontier.

Post-SSH persistence verified beyond300s: both hosts progressed from native
references into candidates, with no failures.145controller2896557 and Lenovo
controller15300 (launcher23272) were active at that historical observation. Native logs/30s monitor retain
worker PIDs, steps and resource samples; these are not quality results.

Closed: both queues5/5complete, no pending/running/failed cells. All ten final
20k native replays pass, generated arrays are exact, and all four native
reference states/samples match archived parents. GPUs are released. The complete
24-corner comparison is in [cycle11-results](cycle11-results/findings.md).
