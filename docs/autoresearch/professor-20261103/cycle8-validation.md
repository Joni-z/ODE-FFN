# Cycle8 documented native CUDA readiness

Warm reuse, unchanged package spec. Before commands verify GPU0 is free and
source matches the published cycle8 commit. Each command checks the native
no-op trajectory and formula, then exercises all six registered host cells:
actual trainer8->16 resume crossing the step10 relaxation boundary must exactly
match uninterrupted16. The two native anchors also match their parent8-step
states. It creates a new readiness directory; do not silently rerun an existing
directory or launch formal20k cells on failure.

```bash
ssh 145 'CUDA_VISIBLE_DEVICES=GPU-5d6078be-3b77-0dfd-0e6b-9cb486e62ff4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data2/zz5070/miniconda3/envs/py312/bin/python /data2/zz5070/ODE_FFN/professor-autoresearch-cycle8-v2-20261103/repo/studies/muon_pd_20261102/autoresearch_20261103/cycle8/check8.py --device cuda --output /data2/zz5070/ODE_FFN/professor-autoresearch-cycle8-v2-20261103/audit/native-witness.json --data-root /data2/zz5070/ODE_FFN/muon-pd-20261102/data --dataset rotated_product_bimodal_16d --smoke-output /data2/zz5070/ODE_FFN/professor-autoresearch-cycle8-v2-20261103/preflight/documented-v2'
```

```bash
ssh lenovo 'set CUDA_VISIBLE_DEVICES=0&& set OMP_NUM_THREADS=1&& set OPENBLAS_NUM_THREADS=1&& set MKL_NUM_THREADS=1&& D:/Research/ODE-FFN/envs/synthetic/Scripts/python.exe D:/Research/ODE-FFN/professor-autoresearch-cycle8-v2-20261103/repo/studies/muon_pd_20261102/autoresearch_20261103/cycle8/check8.py --device cuda --output D:/Research/ODE-FFN/professor-autoresearch-cycle8-v2-20261103/audit/native-witness.json --data-root D:/Research/ODE-FFN/datasets/synthetic --dataset gmm_ring8_2d --smoke-output D:/Research/ODE-FFN/professor-autoresearch-cycle8-v2-20261103/preflight/documented-v2'
```

Pass: exit0, passed=true, two kernel/formula families and six native rows per
host, each native_resume16_exact_continuous16=true. Archive exact source commit,
receipt and command logs. No formal launch before the fresh documented pass.

Revision2: the first145readiness attempt at4d381c6 stopped before its first
8-step cell because Product16 was mislabeled v1 instead of its archived
product16_v1 cache. Logs and the original worktree are preserved. The grid
now copies each parent data version, and the seeded check verifies all12
parent/cache identities before model updates. New runtime uses a separate
cycle8-v2 worktree and fresh documented-v2 directories; no formal20k runs
used the incorrect grid.
