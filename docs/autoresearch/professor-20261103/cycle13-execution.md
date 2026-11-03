# Cycle13 execution

Use the published cycle13 runtime commit recorded in readiness receipts, separate
professor-autoresearch-cycle13-20261103 worktrees, unchanged packages and freeGPU0.
Require both fresh documented readiness passes before these commands.

```bash
ssh 145 'CUDA_VISIBLE_DEVICES=GPU-5d6078be-3b77-0dfd-0e6b-9cb486e62ff4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data2/zz5070/miniconda3/envs/py312/bin/python /data2/zz5070/ODE_FFN/professor-autoresearch-cycle13-20261103/repo/studies/muon_pd_20261102/run_queue.py --host 145 --device cuda --data-root /data2/zz5070/ODE_FFN/muon-pd-20261102/data --output /data2/zz5070/ODE_FFN/professor-autoresearch-cycle13-20261103/output --workers 2 --detach --grid /data2/zz5070/ODE_FFN/professor-autoresearch-cycle13-20261103/repo/studies/muon_pd_20261102/autoresearch_20261103/cycle13/grid.json --runner /data2/zz5070/ODE_FFN/professor-autoresearch-cycle13-20261103/repo/studies/muon_pd_20261102/autoresearch_20261103/cycle13/trial13.py --experiment-id professor_autoresearch_cycle13_20261103'
```

```bash
ssh lenovo 'set CUDA_VISIBLE_DEVICES=0&& set OMP_NUM_THREADS=1&& set OPENBLAS_NUM_THREADS=1&& set MKL_NUM_THREADS=1&& set PATH=D:\Research\ODE-FFN\tools\git\cmd;%PATH%&& D:/Research/ODE-FFN/envs/synthetic/Scripts/python.exe D:/Research/ODE-FFN/professor-autoresearch-cycle13-20261103/repo/studies/muon_pd_20261102/run_queue.py --host lenovo --device cuda --data-root D:/Research/ODE-FFN/datasets/synthetic --output D:/Research/ODE-FFN/professor-autoresearch-cycle13-20261103/output --workers 2 --detach --grid D:/Research/ODE-FFN/professor-autoresearch-cycle13-20261103/repo/studies/muon_pd_20261102/autoresearch_20261103/cycle13/grid.json --runner D:/Research/ODE-FFN/professor-autoresearch-cycle13-20261103/repo/studies/muon_pd_20261102/autoresearch_20261103/cycle13/trial13.py --experiment-id professor_autoresearch_cycle13_20261103'
```

Each host has nine cells, two one-thread workers. Constant-D1/Muon and selected
Swi64 start first, then ISRU-D/free ISRU, AdamW comparisons and wider references.
The native monitor logs PID, steps, memory, utilization and failures every30s.
Verify increasing steps after SSH exits. Do not restart failed cells silently.

After completion, run audit_completed.py with --variant-module
autoresearch_20261103/cycle13/trial13.py --experiment-id
professor_autoresearch_cycle13_20261103 --expected-count9 --endpoint-step20000.
Run cycle13/anchors.py with all three original-output roots; require five exact
20k numerical states/saved arrays per host. Full controls and both metrics must
remain visible. No further method selection from incomplete results.
