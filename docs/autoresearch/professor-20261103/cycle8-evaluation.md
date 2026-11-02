# Cycle8 endpoint verification

Run after the host has completed all six cells and its GPU0 is free. Use the
frozen f6bfefc native evaluator. The additional published state/sample helpers
are outside the frozen worktree and run on CPU. Early anchor receipts have
partial=true and do not substitute for the final six-cell state receipts.

## 145

```bash
ssh 145 'CUDA_VISIBLE_DEVICES=GPU-5d6078be-3b77-0dfd-0e6b-9cb486e62ff4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data2/zz5070/miniconda3/envs/py312/bin/python /data2/zz5070/ODE_FFN/professor-autoresearch-cycle8-v2-20261103/repo/studies/muon_pd_20261102/audit_completed.py --source /data2/zz5070/ODE_FFN/professor-autoresearch-cycle8-v2-20261103/repo/studies/muon_pd_20261102 --variant-module autoresearch_20261103/cycle8/trial8.py --experiment-id professor_autoresearch_cycle8_20261103 --endpoint-step 20000 --output-root /data2/zz5070/ODE_FFN/professor-autoresearch-cycle8-v2-20261103/output --expected-count 6 --save /data2/zz5070/ODE_FFN/professor-autoresearch-cycle8-v2-20261103/audit/replay.json'
```

```bash
ssh 145 'CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data2/zz5070/miniconda3/envs/py312/bin/python /data2/zz5070/ODE_FFN/professor-autoresearch-cycle8-v2-20261103/audit_states.py --source /data2/zz5070/ODE_FFN/professor-autoresearch-cycle8-v2-20261103/repo/studies/muon_pd_20261102 --output-root /data2/zz5070/ODE_FFN/professor-autoresearch-cycle8-v2-20261103/output --old-output /data2/zz5070/ODE_FFN/muon-pd-20261102/output --cycle3-output /data2/zz5070/ODE_FFN/professor-autoresearch-cycle3-20261103/output --host 145 --save /data2/zz5070/ODE_FFN/professor-autoresearch-cycle8-v2-20261103/audit/states.json'
```

```bash
ssh 145 'CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data2/zz5070/miniconda3/envs/py312/bin/python /data2/zz5070/ODE_FFN/professor-autoresearch-cycle8-v2-20261103/sample_readouts.py --output-root /data2/zz5070/ODE_FFN/professor-autoresearch-cycle8-v2-20261103/output --audit /data2/zz5070/ODE_FFN/professor-autoresearch-cycle8-v2-20261103/audit/replay.json --save /data2/zz5070/ODE_FFN/professor-autoresearch-cycle8-v2-20261103/audit/sample-readouts.json --endpoint-step 20000 --expected-count 6 --experiment-id professor_autoresearch_cycle8_20261103'
```

## lenovo

```bash
ssh lenovo 'set CUDA_VISIBLE_DEVICES=0&& set OMP_NUM_THREADS=1&& set OPENBLAS_NUM_THREADS=1&& D:/Research/ODE-FFN/envs/synthetic/Scripts/python.exe D:/Research/ODE-FFN/professor-autoresearch-cycle8-v2-20261103/repo/studies/muon_pd_20261102/audit_completed.py --source D:/Research/ODE-FFN/professor-autoresearch-cycle8-v2-20261103/repo/studies/muon_pd_20261102 --variant-module autoresearch_20261103/cycle8/trial8.py --experiment-id professor_autoresearch_cycle8_20261103 --endpoint-step 20000 --output-root D:/Research/ODE-FFN/professor-autoresearch-cycle8-v2-20261103/output --expected-count 6 --save D:/Research/ODE-FFN/professor-autoresearch-cycle8-v2-20261103/audit/replay.json'
```

```bash
ssh lenovo 'set CUDA_VISIBLE_DEVICES=&& set OMP_NUM_THREADS=1&& set OPENBLAS_NUM_THREADS=1&& D:/Research/ODE-FFN/envs/synthetic/Scripts/python.exe D:/Research/ODE-FFN/professor-autoresearch-cycle8-v2-20261103/audit_states.py --source D:/Research/ODE-FFN/professor-autoresearch-cycle8-v2-20261103/repo/studies/muon_pd_20261102 --output-root D:/Research/ODE-FFN/professor-autoresearch-cycle8-v2-20261103/output --old-output D:/Research/ODE-FFN/muon-pd-20261102/output --cycle3-output D:/Research/ODE-FFN/professor-autoresearch-cycle3-20261103/output --host lenovo --save D:/Research/ODE-FFN/professor-autoresearch-cycle8-v2-20261103/audit/states.json'
```

```bash
ssh lenovo 'set CUDA_VISIBLE_DEVICES=&& set OMP_NUM_THREADS=1&& set OPENBLAS_NUM_THREADS=1&& D:/Research/ODE-FFN/envs/synthetic/Scripts/python.exe D:/Research/ODE-FFN/professor-autoresearch-cycle8-v2-20261103/sample_readouts.py --output-root D:/Research/ODE-FFN/professor-autoresearch-cycle8-v2-20261103/output --audit D:/Research/ODE-FFN/professor-autoresearch-cycle8-v2-20261103/audit/replay.json --save D:/Research/ODE-FFN/professor-autoresearch-cycle8-v2-20261103/audit/sample-readouts.json --endpoint-step 20000 --expected-count 6 --experiment-id professor_autoresearch_cycle8_20261103'
```

After downloading all receipts, curate all12 endpoints once, then run the
all-arm render helper. Require all four exact archived20k model/optimizer/RNG
anchors, all saved-array replays, no pending cases, and final logged matrix
spectra matching the checkpoint. Test metrics never select a checkpoint.
