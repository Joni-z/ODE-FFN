# Cycle13 native readiness

Warm reuse of the unchanged synthetic environments. Check source cleanliness
and freeGPU0 on both hosts, then run each command exactly once. Keep all logs.
No formal20k launch before a fresh agent reports both passed=true.

The witness checks exact copied definitions/provenance,18grid parent records,
common tensors/buffers/RNG, complete ISRU layer and ordinary-control Jacobians,
bounded odd features, origin Hessian/third derivative, native oracle formula and
all optimizer groups. Each host then executes nine real native entries:
8->16 must match uninterrupted16 exactly; all five native references must
match their original8-step states. Progress persists after every completed cell.

```bash
ssh 145 'CUDA_VISIBLE_DEVICES=GPU-5d6078be-3b77-0dfd-0e6b-9cb486e62ff4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data2/zz5070/miniconda3/envs/py312/bin/python /data2/zz5070/ODE_FFN/professor-autoresearch-cycle13-20261103/repo/studies/muon_pd_20261102/autoresearch_20261103/cycle13/check13.py --device cuda --output /data2/zz5070/ODE_FFN/professor-autoresearch-cycle13-20261103/audit/native-witness.json --data-root /data2/zz5070/ODE_FFN/muon-pd-20261102/data --dataset rotated_product_bimodal_16d --smoke-output /data2/zz5070/ODE_FFN/professor-autoresearch-cycle13-20261103/preflight/documented'
```

```bash
ssh lenovo 'set CUDA_VISIBLE_DEVICES=0&& set OMP_NUM_THREADS=1&& set OPENBLAS_NUM_THREADS=1&& set MKL_NUM_THREADS=1&& D:/Research/ODE-FFN/envs/synthetic/Scripts/python.exe D:/Research/ODE-FFN/professor-autoresearch-cycle13-20261103/repo/studies/muon_pd_20261102/autoresearch_20261103/cycle13/check13.py --device cuda --output D:/Research/ODE-FFN/professor-autoresearch-cycle13-20261103/audit/native-witness.json --data-root D:/Research/ODE-FFN/datasets/synthetic --dataset gmm_ring8_2d --smoke-output D:/Research/ODE-FFN/professor-autoresearch-cycle13-20261103/preflight/documented'
```

Pass: exit0, passed=true, device=cuda, twelve kernel/model rows, nine native rows with exact resume states, five parent8 anchors. No repair or retry by the witness agent. Lenovo Git is D:/Research/ODE-FFN/tools/git/cmd/git.exe. Readiness is not quality evidence.

Both commands passed verbatim once at clean5dd1e1644ab4eb69bee3e1cd1b44f4cfc0af6dcf: nine exact continuations and five parent8 anchors perhost. No divergence/repair/retry. [Fresh receipt](readiness/cycle13/documented-witness.json).
