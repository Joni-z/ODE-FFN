# Cycle6 execution

Frozen runtime b33c837f2c3892f4504e04a1f23754dd1a2cd278. Budget-only80k confirmation,8trajectories: rawK1 and native Swi64/128/192 on each task. No model/loss/data/rate changes. Fresh documented CUDA checks pass on both hosts for all4old/new8-update numerical prefixes and actual8-to16 continuation. Source worktrees remain immutable.

## 145

```bash
ssh 145 'CUDA_VISIBLE_DEVICES=GPU-5d6078be-3b77-0dfd-0e6b-9cb486e62ff4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data2/zz5070/miniconda3/envs/py312/bin/python /data2/zz5070/ODE_FFN/professor-autoresearch-cycle6-20261103/repo/studies/muon_pd_20261102/run_queue.py --host 145 --device cuda --data-root /data2/zz5070/ODE_FFN/muon-pd-20261102/data --output /data2/zz5070/ODE_FFN/professor-autoresearch-cycle6-20261103/output --workers 2 --detach --grid /data2/zz5070/ODE_FFN/professor-autoresearch-cycle6-20261103/repo/studies/muon_pd_20261102/autoresearch_20261103/cycle6/grid.json --runner /data2/zz5070/ODE_FFN/professor-autoresearch-cycle6-20261103/repo/studies/muon_pd_20261102/autoresearch_20261103/cycle6/trial6.py --experiment-id professor_autoresearch_cycle6_20261103'
```

## lenovo

```bash
ssh lenovo 'set CUDA_VISIBLE_DEVICES=0&& set OMP_NUM_THREADS=1&& set OPENBLAS_NUM_THREADS=1&& set MKL_NUM_THREADS=1&& set PATH=D:\Research\ODE-FFN\tools\git\cmd;%PATH%&& D:/Research/ODE-FFN/envs/synthetic/Scripts/python.exe D:/Research/ODE-FFN/professor-autoresearch-cycle6-20261103/repo/studies/muon_pd_20261102/run_queue.py --host lenovo --device cuda --data-root D:/Research/ODE-FFN/datasets/synthetic --output D:/Research/ODE-FFN/professor-autoresearch-cycle6-20261103/output --workers 2 --detach --grid D:/Research/ODE-FFN/professor-autoresearch-cycle6-20261103/repo/studies/muon_pd_20261102/autoresearch_20261103/cycle6/grid.json --runner D:/Research/ODE-FFN/professor-autoresearch-cycle6-20261103/repo/studies/muon_pd_20261102/autoresearch_20261103/cycle6/trial6.py --experiment-id professor_autoresearch_cycle6_20261103'
```

Status: live and independently verified after SSH exit. Over30s145Swi64 advanced12000->18500 and raw6000->9000; Lenovo5500->8500 and5000->8000.145GPU0utilization93%/1183MiB, Lenovo91%/808MiB.145controller2238359/CUDA2238360,2238361; Lenovo launcher16300/controller27876/actualCUDA23348,10584. Both source worktrees stay clean at b33c837, no failures. Next verify exact archived20k numerical prefixes after each loop passes20500. Persistent monitor:output/monitor/status.json and history.jsonl. Failed cells stop dispatch; no silent retries.80k endpoints remain unfinished.

Early20k prefix audit PASSED for the first Swi64/raw pair on both hosts:
all model/optimizer/buffer/RNG numerical states match their archived parents
exactly. Four of8prefixes verified; each host still lists pending Swi128/192.
Receipts:readiness/cycle6/{145-prefix-early,lenovo-prefix-early}.json. This is
comparability evidence, not an80k quality result.

Full20k prefix milestone PASSED:8/8 model/optimizer/buffer/RNG states exactly
match original parents; no pending cells in either final receipt. Both commands
ran once after their four selected children passed20500. Bindings/hashes pass;
source remains b33c837. Receipts:readiness/cycle6/{145-prefix,lenovo-prefix}.json.
Remaining task: finish and replay80k, not another prefix or architecture search.

Closed: all8reached80k, passed exact20k states and native endpoint replay. The
launch commands above now name trial6.py, matching the original monitor process
commands; an earlier documentation typo said trial3.py. Runtime/code/results
were unaffected. See cycle6-results/findings.md.
