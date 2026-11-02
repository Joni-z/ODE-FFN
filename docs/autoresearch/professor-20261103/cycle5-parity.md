# Registered post-replay parity commands

Run only after all14 formal cells and their independent replay pass, on freeGPU0.
Use the published symmetry_case.py copied outside the frozen worktree to the
cycle5 root; record its SHA256 in the resulting receipt. Its new optional mode
selects exactly one final checkpoint per new family using the existing final
validation error, with all7 cells in the selection-pool receipt. The retained
old-output arguments preserve the existing CLI; cycle5 mode evaluates only the
two new families. Neither test sampling nor parity values enter selection.

## 145

```bash
ssh 145 'CUDA_VISIBLE_DEVICES=GPU-5d6078be-3b77-0dfd-0e6b-9cb486e62ff4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data2/zz5070/miniconda3/envs/py312/bin/python /data2/zz5070/ODE_FFN/professor-autoresearch-cycle5-20261103/symmetry_case.py --source /data2/zz5070/ODE_FFN/professor-autoresearch-cycle5-20261103/repo/studies/muon_pd_20261102 --old-output /data2/zz5070/ODE_FFN/muon-pd-20261102/output --cycle2-output /data2/zz5070/ODE_FFN/professor-autoresearch-cycle2-20261103/output --cycle3-output /data2/zz5070/ODE_FFN/professor-autoresearch-cycle3-20261103/output --cycle5-output /data2/zz5070/ODE_FFN/professor-autoresearch-cycle5-20261103/output --dataset rotated_product_bimodal_16d --save /data2/zz5070/ODE_FFN/professor-autoresearch-cycle5-20261103/audit/parity.json'
```

## Lenovo

```bash
ssh lenovo 'set CUDA_VISIBLE_DEVICES=0&& set OMP_NUM_THREADS=1&& set OPENBLAS_NUM_THREADS=1&& D:/Research/ODE-FFN/envs/synthetic/Scripts/python.exe D:/Research/ODE-FFN/professor-autoresearch-cycle5-20261103/symmetry_case.py --source D:/Research/ODE-FFN/professor-autoresearch-cycle5-20261103/repo/studies/muon_pd_20261102 --old-output D:/Research/ODE-FFN/muon-pd-20261102/output --cycle2-output D:/Research/ODE-FFN/professor-autoresearch-cycle2-20261103/output --cycle3-output D:/Research/ODE-FFN/professor-autoresearch-cycle3-20261103/output --cycle5-output D:/Research/ODE-FFN/professor-autoresearch-cycle5-20261103/output --dataset gmm_ring8_2d --save D:/Research/ODE-FFN/professor-autoresearch-cycle5-20261103/audit/parity.json'
```
