# Cycle2 native documented witnesses

Execute each once on free GPUs after separate worktree deployment. Preserve failed output; do not retry into the same smoke path. Tests include zero/signed input algebra, controller-rank matching, full derivatives, six optimizer roundtrips, and actual8-to16 native continuation for all three candidates. No formal queues in this witness.

## 145

```bash
ssh 145 'CUDA_VISIBLE_DEVICES=GPU-5d6078be-3b77-0dfd-0e6b-9cb486e62ff4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data2/zz5070/miniconda3/envs/py312/bin/python /data2/zz5070/ODE_FFN/professor-autoresearch-cycle2-20261103/repo/studies/muon_pd_20261102/autoresearch_20261103/cycle2/check2.py --device cuda --data-root /data2/zz5070/ODE_FFN/muon-pd-20261102/data --smoke-output /data2/zz5070/ODE_FFN/professor-autoresearch-cycle2-20261103/preflight/native-smoke --output /data2/zz5070/ODE_FFN/professor-autoresearch-cycle2-20261103/preflight/cuda.json'
```

## lenovo

```bash
ssh lenovo 'set CUDA_VISIBLE_DEVICES=0&& set OMP_NUM_THREADS=1&& set OPENBLAS_NUM_THREADS=1&& set MKL_NUM_THREADS=1&& D:/Research/ODE-FFN/envs/synthetic/Scripts/python.exe D:/Research/ODE-FFN/professor-autoresearch-cycle2-20261103/repo/studies/muon_pd_20261102/autoresearch_20261103/cycle2/check2.py --device cuda --data-root D:/Research/ODE-FFN/datasets/synthetic --smoke-output D:/Research/ODE-FFN/professor-autoresearch-cycle2-20261103/preflight/native-smoke --output D:/Research/ODE-FFN/professor-autoresearch-cycle2-20261103/preflight/cuda.json'
```
