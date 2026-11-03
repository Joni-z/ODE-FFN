# Cycle10 native derivative and actual-entry readiness

Warm reuse, unchanged packages, trainer and official optimizer. After deployment
verify source cleanliness and freeGPU0, then execute each command once. The
seeded check tests both new D functions, full layer Jacobians, raw direction/
parity, normalized derivative bound, zero-anchor Hessians, unit-RMS static match
and paired common tensors/buffers/RNG. All12parent data versions must match.

Each host then runs its six actual native entries:8->16 continuation must match
uninterrupted16 for all six, and all four native references must match their
original8-step numerical states. Keep the new output directory unique and
preserve failure logs; no formal20k training before both fresh passes succeed.

```bash
ssh 145 'CUDA_VISIBLE_DEVICES=GPU-5d6078be-3b77-0dfd-0e6b-9cb486e62ff4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data2/zz5070/miniconda3/envs/py312/bin/python /data2/zz5070/ODE_FFN/professor-autoresearch-cycle10-20261103/repo/studies/muon_pd_20261102/autoresearch_20261103/cycle10/check10.py --device cuda --output /data2/zz5070/ODE_FFN/professor-autoresearch-cycle10-20261103/audit/native-witness.json --data-root /data2/zz5070/ODE_FFN/muon-pd-20261102/data --dataset rotated_product_bimodal_16d --smoke-output /data2/zz5070/ODE_FFN/professor-autoresearch-cycle10-20261103/preflight/documented'
```

```bash
ssh lenovo 'set CUDA_VISIBLE_DEVICES=0&& set OMP_NUM_THREADS=1&& set OPENBLAS_NUM_THREADS=1&& set MKL_NUM_THREADS=1&& D:/Research/ODE-FFN/envs/synthetic/Scripts/python.exe D:/Research/ODE-FFN/professor-autoresearch-cycle10-20261103/repo/studies/muon_pd_20261102/autoresearch_20261103/cycle10/check10.py --device cuda --output D:/Research/ODE-FFN/professor-autoresearch-cycle10-20261103/audit/native-witness.json --data-root D:/Research/ODE-FFN/datasets/synthetic --dataset gmm_ring8_2d --smoke-output D:/Research/ODE-FFN/professor-autoresearch-cycle10-20261103/preflight/documented'
```

Pass: exit0, passed=true, device=cuda, two new-function rows and six native rows,
all exact resume comparisons and four parent8 anchors. Record source/grid hashes.
These checks are implementation readiness, not a positive experimental result.
