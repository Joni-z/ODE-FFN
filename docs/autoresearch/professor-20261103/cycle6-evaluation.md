# Cycle6 endpoint replay and registered final examples

Each host may be checked after all its4formal cells complete and GPU0 is free.
Do not interpret a winner until both hosts/all8are complete and replayed. Use
the frozen b33c837 source/model builders. Both full prefix receipts must already
have4exact matches/no pending. No model, checkpoint or metric selection here.

## Native80k replay

```bash
ssh 145 'CUDA_VISIBLE_DEVICES=GPU-5d6078be-3b77-0dfd-0e6b-9cb486e62ff4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data2/zz5070/miniconda3/envs/py312/bin/python /data2/zz5070/ODE_FFN/professor-autoresearch-cycle6-20261103/repo/studies/muon_pd_20261102/audit_completed.py --source /data2/zz5070/ODE_FFN/professor-autoresearch-cycle6-20261103/repo/studies/muon_pd_20261102 --variant-module autoresearch_20261103/cycle6/trial6.py --experiment-id professor_autoresearch_cycle6_20261103 --endpoint-step 80000 --output-root /data2/zz5070/ODE_FFN/professor-autoresearch-cycle6-20261103/output --expected-count 4 --save /data2/zz5070/ODE_FFN/professor-autoresearch-cycle6-20261103/audit/replay.json'
```

```bash
ssh lenovo 'set CUDA_VISIBLE_DEVICES=0&& set OMP_NUM_THREADS=1&& set OPENBLAS_NUM_THREADS=1&& D:/Research/ODE-FFN/envs/synthetic/Scripts/python.exe D:/Research/ODE-FFN/professor-autoresearch-cycle6-20261103/repo/studies/muon_pd_20261102/audit_completed.py --source D:/Research/ODE-FFN/professor-autoresearch-cycle6-20261103/repo/studies/muon_pd_20261102 --variant-module autoresearch_20261103/cycle6/trial6.py --experiment-id professor_autoresearch_cycle6_20261103 --endpoint-step 80000 --output-root D:/Research/ODE-FFN/professor-autoresearch-cycle6-20261103/output --expected-count 4 --save D:/Research/ODE-FFN/professor-autoresearch-cycle6-20261103/audit/replay.json'
```

## Read-only final examples

Copy the published cycle6/sample_readouts.py outside the frozen worktree to each
cycle6 root before these commands. CPU load checks the saved checkpoint/config/
manifest binding and original target arrays. No new sample generation.

```bash
ssh 145 'CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data2/zz5070/miniconda3/envs/py312/bin/python /data2/zz5070/ODE_FFN/professor-autoresearch-cycle6-20261103/sample_readouts.py --output-root /data2/zz5070/ODE_FFN/professor-autoresearch-cycle6-20261103/output --audit /data2/zz5070/ODE_FFN/professor-autoresearch-cycle6-20261103/audit/replay.json --save /data2/zz5070/ODE_FFN/professor-autoresearch-cycle6-20261103/audit/sample-readouts.json'
```

```bash
ssh lenovo 'set CUDA_VISIBLE_DEVICES=&& set OMP_NUM_THREADS=1&& set OPENBLAS_NUM_THREADS=1&& D:/Research/ODE-FFN/envs/synthetic/Scripts/python.exe D:/Research/ODE-FFN/professor-autoresearch-cycle6-20261103/sample_readouts.py --output-root D:/Research/ODE-FFN/professor-autoresearch-cycle6-20261103/output --audit D:/Research/ODE-FFN/professor-autoresearch-cycle6-20261103/audit/replay.json --save D:/Research/ODE-FFN/professor-autoresearch-cycle6-20261103/audit/sample-readouts.json'
```
