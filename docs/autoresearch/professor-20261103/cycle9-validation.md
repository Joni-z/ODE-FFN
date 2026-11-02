# Cycle9 native CUDA identity witness

No new packages, model, training or data. After deploying the frozen source,
verify GPU0 is free, then run both commands once. Each checks 12 native layers
from two seeded families, exact complete-prediction equality with/without hooks,
float64 moment/finite-series identities, native-output reconstruction and the
symmetric energy-accounting formula. The14-case cohort and its hash are checked.
This is a diagnostic readiness check, not a learning result.

```bash
ssh 145 'CUDA_VISIBLE_DEVICES=GPU-5d6078be-3b77-0dfd-0e6b-9cb486e62ff4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data2/zz5070/miniconda3/envs/py312/bin/python /data2/zz5070/ODE_FFN/professor-autoresearch-cycle9-20261103/repo/studies/muon_pd_20261102/autoresearch_20261103/cycle9/features.py --source /data2/zz5070/ODE_FFN/professor-autoresearch-cycle9-20261103/repo/studies/muon_pd_20261102 --device cuda --self-test --out /data2/zz5070/ODE_FFN/professor-autoresearch-cycle9-20261103/audit/native-witness.json'
```

```bash
ssh lenovo 'set CUDA_VISIBLE_DEVICES=0&& set OMP_NUM_THREADS=1&& set OPENBLAS_NUM_THREADS=1&& D:/Research/ODE-FFN/envs/synthetic/Scripts/python.exe D:/Research/ODE-FFN/professor-autoresearch-cycle9-20261103/repo/studies/muon_pd_20261102/autoresearch_20261103/cycle9/features.py --source D:/Research/ODE-FFN/professor-autoresearch-cycle9-20261103/repo/studies/muon_pd_20261102 --device cuda --self-test --out D:/Research/ODE-FFN/professor-autoresearch-cycle9-20261103/audit/native-witness.json'
```

Pass: exit0, passed=true, device=cuda, hooked_prediction_exact=true, 12 layer
records, matching source/cohort hashes on both hosts. Native reconstruction
relative RMS errors must be below5e-6; moment/series/accounting identities must
meet the source's tighter float64 bounds. Preserve and diagnose any failure;
do not rerun a changed source inside the same frozen worktree.
