# Cycle9 execution and saved-array replay

Frozen native producer:8bb1490. Both fresh CUDA witnesses passed once, with
12native layer checks per host and exact hooked-predictor equality. All fourteen
registered checkpoints were read on their original hosts; no parameters were
updated. Raw sufficient arrays remain outside Git with published SHA256 bindings.

## 145

```bash
ssh 145 'CUDA_VISIBLE_DEVICES=GPU-5d6078be-3b77-0dfd-0e6b-9cb486e62ff4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data2/zz5070/miniconda3/envs/py312/bin/python -u /data2/zz5070/ODE_FFN/professor-autoresearch-cycle9-20261103/repo/studies/muon_pd_20261102/autoresearch_20261103/cycle9/features.py --source /data2/zz5070/ODE_FFN/professor-autoresearch-cycle9-20261103/repo/studies/muon_pd_20261102 --device cuda --host 145 --cycle8-output /data2/zz5070/ODE_FFN/professor-autoresearch-cycle8-v2-20261103/output --cycle6-output /data2/zz5070/ODE_FFN/professor-autoresearch-cycle6-20261103/output --out /data2/zz5070/ODE_FFN/professor-autoresearch-cycle9-20261103/audit/features > /data2/zz5070/ODE_FFN/professor-autoresearch-cycle9-20261103/audit/formal.log 2>&1'
```

## lenovo

```bash
ssh lenovo 'set CUDA_VISIBLE_DEVICES=0&& set OMP_NUM_THREADS=1&& set OPENBLAS_NUM_THREADS=1&& D:/Research/ODE-FFN/envs/synthetic/Scripts/python.exe -u D:/Research/ODE-FFN/professor-autoresearch-cycle9-20261103/repo/studies/muon_pd_20261102/autoresearch_20261103/cycle9/features.py --source D:/Research/ODE-FFN/professor-autoresearch-cycle9-20261103/repo/studies/muon_pd_20261102 --device cuda --host lenovo --cycle8-output D:/Research/ODE-FFN/professor-autoresearch-cycle8-v2-20261103/output --cycle6-output D:/Research/ODE-FFN/professor-autoresearch-cycle6-20261103/output --out D:/Research/ODE-FFN/professor-autoresearch-cycle9-20261103/audit/features > D:/Research/ODE-FFN/professor-autoresearch-cycle9-20261103/audit/formal.log 2>&1'
```

All seven cases/six layers per host completed. GPUs returned to2MiB on145 and
415MiB desktop usage onLenovo. The raw mirror is
`/Users/mr.z/ode-workspace-prep/muon-pd-20261102/nov03-cycle9-evidence/{145,lenovo}`.

```bash
OPENBLAS_NUM_THREADS=1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/autoresearch_20261103/cycle9/replay.py --evidence /Users/mr.z/ode-workspace-prep/muon-pd-20261102/nov03-cycle9-evidence --out docs/autoresearch/professor-20261103/cycle9-results
```

Independent NumPy einsum/direct-sample calculations verify moments, polynomial
energy, spectra and all60matched-layer accounting identities. The auxiliary
feature-amplitude times directional-gain factorization was added before reading
these diagnostic outcomes; it selects no model and replaces no quality metric.
Source/array/cohort digests are retained. This replay does not re-generate the
native hidden states or prove a causal optimization mechanism.
