# Cycle14 frozen-checkpoint readiness

Warm reuse; no installation or training. Run each command below once, verbatim,
after checking the clean worktree and Lenovo GPU0 is free. Preserve every
log and stop on divergence. Do not repair, retry or launch the formal queue in
the witness. Windows Git is D:/Research/ODE-FFN/tools/git/cmd/git.exe.

The source is the GitHub-published cycle14 commit in each separate
professor-validation14-20261104/repo worktree. The caller records its exact HEAD.
Expected: exit0, passed=true, device=cuda on Lenovo and device=cpu on145, eight seeded restore/kernel fixtures,
15 physical checkpoint bindings, five native4096sample final anchors and
15 fresh64sample/Heun128 finite dispatches per host. Every parameter and runtime
buffer stays unchanged. No optimizer is constructed. This is readiness evidence.

145 now has all eight GPUs occupied; use two one-thread CPU workers there.
Lenovo uses its sole4060. No other task is preempted. Lenovo anchors must be
bitwise exact.145 CPU replay uses rtol1e-5/atol1e-6 against archived CUDA
coordinates, plus absolute2e-6 scalar SW2/field guards. Retain actual array
errors and exactness flags; CPU roundoff is not called bitwise equality. This
execution-only adaptation was registered before any new sampling outcome.

```bash
ssh 145 'CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data2/zz5070/miniconda3/envs/py312/bin/python /data2/zz5070/ODE_FFN/professor-validation14-20261104/repo/studies/muon_pd_20261102/autoresearch_20261103/cycle14/audit.py --device cpu --host 145 --witness --data-root /data2/zz5070/ODE_FFN/muon-pd-20261102/data --output /data2/zz5070/ODE_FFN/professor-validation14-20261104/preflight/documented'
```

```bash
ssh lenovo 'set CUDA_VISIBLE_DEVICES=0&& set OMP_NUM_THREADS=1&& set OPENBLAS_NUM_THREADS=1&& set MKL_NUM_THREADS=1&& D:/Research/ODE-FFN/envs/synthetic/Scripts/python.exe D:/Research/ODE-FFN/professor-validation14-20261104/repo/studies/muon_pd_20261102/autoresearch_20261103/cycle14/audit.py --device cuda --host lenovo --witness --data-root D:/Research/ODE-FFN/datasets/synthetic --output D:/Research/ODE-FFN/professor-validation14-20261104/preflight/documented'
```

After a fresh independent witness passes both commands, recheck the Lenovo GPU and launch
the unchanged native run_queue.py with two workers, cycle14/grid.json,
cycle14/audit.py, experiment id professor_validation14_20261104, and a new output
folder. Persist30second monitoring and confirm progress after SSH exits.
ADAPTER_COMPLETE denotes a completed frozen evaluation here, not a trained fit.
