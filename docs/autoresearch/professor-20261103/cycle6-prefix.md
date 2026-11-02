# Cycle6 numerical-prefix checks

Wait until the relevant child metrics have advanced beyond20500 so the20k file
is closed. Partial mode checks only ready cells and explicitly lists pending
ones; it must not be reported as all-eight completion. Run CPU-only alongside
training. Hash-bound original checkpoints/configs/manifests remain read-only.

## Early first-pair check

```bash
ssh 145 'CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data2/zz5070/miniconda3/envs/py312/bin/python /data2/zz5070/ODE_FFN/professor-autoresearch-cycle6-20261103/repo/studies/muon_pd_20261102/autoresearch_20261103/cycle6/prefix.py --host 145 --output-root /data2/zz5070/ODE_FFN/professor-autoresearch-cycle6-20261103/output --old-output /data2/zz5070/ODE_FFN/muon-pd-20261102/output --cycle3-output /data2/zz5070/ODE_FFN/professor-autoresearch-cycle3-20261103/output --save /data2/zz5070/ODE_FFN/professor-autoresearch-cycle6-20261103/audit/prefix-early.json --only-ready'
```

```bash
ssh lenovo 'set CUDA_VISIBLE_DEVICES=&& set OMP_NUM_THREADS=1&& set OPENBLAS_NUM_THREADS=1&& D:/Research/ODE-FFN/envs/synthetic/Scripts/python.exe D:/Research/ODE-FFN/professor-autoresearch-cycle6-20261103/repo/studies/muon_pd_20261102/autoresearch_20261103/cycle6/prefix.py --host lenovo --output-root D:/Research/ODE-FFN/professor-autoresearch-cycle6-20261103/output --old-output D:/Research/ODE-FFN/muon-pd-20261102/output --cycle3-output D:/Research/ODE-FFN/professor-autoresearch-cycle3-20261103/output --save D:/Research/ODE-FFN/professor-autoresearch-cycle6-20261103/audit/prefix-early.json --only-ready'
```

Each receipt must contain exact numerical matches for its first Swi64 and rawK1
pair. Later use the same commands without --only-ready and save prefix.json,
after all4selected cells perhost pass20500. That final check must have4rows and
no pending cells. Compare model, optimizer, nonpersistent buffers, data/global
RNG states exactly; different run identities/config hashes are deliberate and
remain recorded. A failure is not a reason to reset or replace a parent run.
