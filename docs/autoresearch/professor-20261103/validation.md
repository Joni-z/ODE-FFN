# Native witness commands

A fresh agent executes each CUDA command once after deployment. Verify GPU0 is free first. Preserve failed directories; do not retry into the same smoke-output. Code/data paths are explicit. No package install, formal queue or source change is part of this witness.

## 145

```bash
ssh 145 'CUDA_VISIBLE_DEVICES=GPU-5d6078be-3b77-0dfd-0e6b-9cb486e62ff4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data2/zz5070/miniconda3/envs/py312/bin/python /data2/zz5070/ODE_FFN/professor-autoresearch-20261103/repo/studies/muon_pd_20261102/autoresearch_20261103/check.py --device cuda --data-root /data2/zz5070/ODE_FFN/muon-pd-20261102/data --smoke-output /data2/zz5070/ODE_FFN/professor-autoresearch-20261103/preflight/native-smoke --output /data2/zz5070/ODE_FFN/professor-autoresearch-20261103/preflight/cuda.json'
```

## lenovo

```bash
ssh lenovo 'set CUDA_VISIBLE_DEVICES=0&& set OMP_NUM_THREADS=1&& set OPENBLAS_NUM_THREADS=1&& set MKL_NUM_THREADS=1&& D:/Research/ODE-FFN/envs/synthetic/Scripts/python.exe D:/Research/ODE-FFN/professor-autoresearch-20261103/repo/studies/muon_pd_20261102/autoresearch_20261103/check.py --device cuda --data-root D:/Research/ODE-FFN/datasets/synthetic --smoke-output D:/Research/ODE-FFN/professor-autoresearch-20261103/preflight/native-smoke --output D:/Research/ODE-FFN/professor-autoresearch-20261103/preflight/cuda.json'
```
