# Cycle8 execution commands

Formal runtime is f6bfefc in separate cycle8-v2 worktrees. Earlier v1 readiness
failure is preserved and no formal20k job used that grid. Launch only after
both fresh v2 native witnesses pass and GPU0 is free. No package change.

```bash
ssh 145 'CUDA_VISIBLE_DEVICES=GPU-5d6078be-3b77-0dfd-0e6b-9cb486e62ff4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data2/zz5070/miniconda3/envs/py312/bin/python /data2/zz5070/ODE_FFN/professor-autoresearch-cycle8-v2-20261103/repo/studies/muon_pd_20261102/run_queue.py --host 145 --device cuda --data-root /data2/zz5070/ODE_FFN/muon-pd-20261102/data --output /data2/zz5070/ODE_FFN/professor-autoresearch-cycle8-v2-20261103/output --workers 2 --detach --grid /data2/zz5070/ODE_FFN/professor-autoresearch-cycle8-v2-20261103/repo/studies/muon_pd_20261102/autoresearch_20261103/cycle8/grid.json --runner /data2/zz5070/ODE_FFN/professor-autoresearch-cycle8-v2-20261103/repo/studies/muon_pd_20261102/autoresearch_20261103/cycle8/trial8.py --experiment-id professor_autoresearch_cycle8_20261103'
```

```bash
ssh lenovo 'set CUDA_VISIBLE_DEVICES=0&& set OMP_NUM_THREADS=1&& set OPENBLAS_NUM_THREADS=1&& set MKL_NUM_THREADS=1&& set PATH=D:\Research\ODE-FFN\tools\git\cmd;%PATH%&& D:/Research/ODE-FFN/envs/synthetic/Scripts/python.exe D:/Research/ODE-FFN/professor-autoresearch-cycle8-v2-20261103/repo/studies/muon_pd_20261102/run_queue.py --host lenovo --device cuda --data-root D:/Research/ODE-FFN/datasets/synthetic --output D:/Research/ODE-FFN/professor-autoresearch-cycle8-v2-20261103/output --workers 2 --detach --grid D:/Research/ODE-FFN/professor-autoresearch-cycle8-v2-20261103/repo/studies/muon_pd_20261102/autoresearch_20261103/cycle8/grid.json --runner D:/Research/ODE-FFN/professor-autoresearch-cycle8-v2-20261103/repo/studies/muon_pd_20261102/autoresearch_20261103/cycle8/trial8.py --experiment-id professor_autoresearch_cycle8_20261103'
```

Each host has six cells, two workers. Native scalar/raw anchors precede shape
and scalar-contraction arms. Existing monitor persists status.json/history.jsonl
every30s with PIDs, completed/pending cells and GPU utilization/memory/power.
Inspect post-SSH progress rather than treating launcher success as persistence.
Final state/update checks use cycle8/audit_states.py; native final replay uses
audit_completed.py with variant-module cycle8/trial8.py and endpoint-step20000.

Post-SSH persistence confirmed over60s:145native scalar/raw advanced to4k/6k
with94%GPU utilization and1191MiB; Lenovo3.5k/5.5k with90%and812MiB.
145controller2489535/children2489555,2489556; Lenovo launcher13808, persistent
controller24900/children30064,23348. Source is immutable f6bfefc. No failures.
GPU activity is a resource-use measure, not a throughput or model-quality claim.

Closed: both queues completed6/6, no failures or pending cells. Native six-cell
endpoint replays and CPU state/update checks pass on both hosts. Four exact
archived20k anchors and twelve exact generated-array replays. Quality results
are in cycle8-results/findings.md; the two training GPUs are released.
