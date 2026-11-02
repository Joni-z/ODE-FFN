# Cycle6 native witness commands

Execute once per host after checking freeGPU0 and clean deployed source. No formal queues in this witness. Preserve failures and use a distinct output directory if a diagnosed correction is necessary.

## 145

```bash
ssh 145 'CUDA_VISIBLE_DEVICES=GPU-5d6078be-3b77-0dfd-0e6b-9cb486e62ff4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data2/zz5070/miniconda3/envs/py312/bin/python /data2/zz5070/ODE_FFN/professor-autoresearch-cycle6-20261103/repo/studies/muon_pd_20261102/autoresearch_20261103/cycle6/check6.py --device cuda --dataset rotated_product_bimodal_16d --data-root /data2/zz5070/ODE_FFN/muon-pd-20261102/data --smoke-output /data2/zz5070/ODE_FFN/professor-autoresearch-cycle6-20261103/preflight/native-smoke --output /data2/zz5070/ODE_FFN/professor-autoresearch-cycle6-20261103/preflight/cuda.json'
```

## lenovo

```bash
ssh lenovo 'set CUDA_VISIBLE_DEVICES=0&& set OMP_NUM_THREADS=1&& set OPENBLAS_NUM_THREADS=1&& set MKL_NUM_THREADS=1&& D:/Research/ODE-FFN/envs/synthetic/Scripts/python.exe D:/Research/ODE-FFN/professor-autoresearch-cycle6-20261103/repo/studies/muon_pd_20261102/autoresearch_20261103/cycle6/check6.py --device cuda --dataset gmm_ring8_2d --data-root D:/Research/ODE-FFN/datasets/synthetic --smoke-output D:/Research/ODE-FFN/professor-autoresearch-cycle6-20261103/preflight/native-smoke --output D:/Research/ODE-FFN/professor-autoresearch-cycle6-20261103/preflight/cuda.json'
```

The witness compares complete old-budget/new-budget numerical checkpoints at8
updates for all4selected families, then performs actual native8-to16 resumes.
The formal80k runs have separate identities; the original20k checkpoints remain
untouched. CPU constructor/source check does not stand in for this CUDA pass.
