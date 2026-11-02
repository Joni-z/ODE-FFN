# Cycle7 documented CUDA witness

Warm reuse of the existing synthetic environments, with no new packages or
training code. Run after deployment and a fresh free-GPU0 check. This checks
the full-rank, rank-deficient, already-isotropic and zero matrix cases, the
shape/scale norm identities, and exact k=0 native model outputs for both families.
The archived model/trainer and official Muon source hashes must still verify.

```bash
ssh 145 'CUDA_VISIBLE_DEVICES=GPU-5d6078be-3b77-0dfd-0e6b-9cb486e62ff4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data2/zz5070/miniconda3/envs/py312/bin/python /data2/zz5070/ODE_FFN/professor-autoresearch-cycle7-20261103/repo/studies/muon_pd_20261102/autoresearch_20261103/cycle7/frozen_shape.py --source /data2/zz5070/ODE_FFN/professor-autoresearch-cycle7-20261103/repo/studies/muon_pd_20261102 --device cuda --self-test --save /data2/zz5070/ODE_FFN/professor-autoresearch-cycle7-20261103/audit/native-witness.json'
```

```bash
ssh lenovo 'set CUDA_VISIBLE_DEVICES=0&& set OMP_NUM_THREADS=1&& set OPENBLAS_NUM_THREADS=1&& D:/Research/ODE-FFN/envs/synthetic/Scripts/python.exe D:/Research/ODE-FFN/professor-autoresearch-cycle7-20261103/repo/studies/muon_pd_20261102/autoresearch_20261103/cycle7/frozen_shape.py --source D:/Research/ODE-FFN/professor-autoresearch-cycle7-20261103/repo/studies/muon_pd_20261102 --device cuda --self-test --save D:/Research/ODE-FFN/professor-autoresearch-cycle7-20261103/audit/native-witness.json'
```

Pass: exit0 and a JSON receipt with passed=true, self_test=true, device=cuda.
No formal checkpoint is changed or trained. Formal untouched anchors additionally
replay all six original checkpoints before interpreting their interventions.
