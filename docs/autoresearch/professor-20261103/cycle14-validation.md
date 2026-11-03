# Cycle14 frozen-checkpoint readiness

Warm reuse; no installation or training. Run each command below once, verbatim,
after checking the clean worktree and the specified GPU is free. Preserve every
log and stop on divergence. Do not repair, retry or launch the formal queue in
the witness. Windows Git is D:/Research/ODE-FFN/tools/git/cmd/git.exe.

The source is the GitHub-published cycle14 commit in each separate
professor-validation14-v2-20261104/repo worktree. The caller records its exact HEAD.
Expected: exit0, passed=true, device=cuda on both hosts, eight seeded restore/kernel fixtures,
15 physical checkpoint bindings, five native4096sample final anchors and
15 fresh64sample/Heun128 finite dispatches per host. Every parameter and runtime
buffer stays unchanged. No optimizer is constructed. This is readiness evidence.

Original-host CUDA is required on both hosts.145GPU0 binding, UUID
GPU-5d6078be-3b77-0dfd-0e6b-9cb486e62ff4. Use separate v2 worktrees and outputs.
Recheck capacity before each invocation; if occupied, leave that host pending.
Both hosts require bitwise native-anchor equality. All checkpoint/model/bank
settings stay fixed. Attempt1 CPU readiness on145 failed its registered array
roundoff guard before formal execution; its logs and runtime remain preserved.
The Lenovo attempt1 CUDA pass is retained but does not replace the new pass.

```bash
ssh 145 'CUDA_VISIBLE_DEVICES=GPU-5d6078be-3b77-0dfd-0e6b-9cb486e62ff4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data2/zz5070/miniconda3/envs/py312/bin/python /data2/zz5070/ODE_FFN/professor-validation14-v2-20261104/repo/studies/muon_pd_20261102/autoresearch_20261103/cycle14/audit.py --device cuda --host 145 --witness --data-root /data2/zz5070/ODE_FFN/muon-pd-20261102/data --output /data2/zz5070/ODE_FFN/professor-validation14-v2-20261104/preflight/documented'
```

```bash
ssh lenovo 'set CUDA_VISIBLE_DEVICES=0&& set OMP_NUM_THREADS=1&& set OPENBLAS_NUM_THREADS=1&& set MKL_NUM_THREADS=1&& D:/Research/ODE-FFN/envs/synthetic/Scripts/python.exe D:/Research/ODE-FFN/professor-validation14-v2-20261104/repo/studies/muon_pd_20261102/autoresearch_20261103/cycle14/audit.py --device cuda --host lenovo --witness --data-root D:/Research/ODE-FFN/datasets/synthetic --output D:/Research/ODE-FFN/professor-validation14-v2-20261104/preflight/documented'
```

After a fresh independent witness passes a host command, recheck that GPU and launch
the unchanged native run_queue.py with two workers, cycle14/grid.json,
cycle14/audit.py, experiment id professor_validation14_20261104, and a new output
folder. One host may proceed while the other is pending; partial analysis must
name the missing cohort. Persist30second monitoring and confirm progress after SSH exits.
ADAPTER_COMPLETE denotes a completed frozen evaluation here, not a trained fit.
