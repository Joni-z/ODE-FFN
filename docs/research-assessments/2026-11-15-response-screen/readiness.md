# Mac mini CPU documented checks

Use the existing environment; do not install packages. No GPU/SSH operation.
Data are already byte-verified local synthetic caches. Assets were generated
before formal training. Run the blocks in order, stop on a failure, and report
divergence without repairing files or deleting outputs. The first block must
pass before running the second. No full20k training is launched here.

```bash
cd /Users/mr.z/ode-workspace-prep/response-screen-20261115/repo
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/response_screen_20261115/check_screen.py --data-root /Users/mr.z/ode-workspace-prep/data/synthetic --assets /Users/mr.z/ode-workspace-prep/response-screen-20261115/assets --output /Users/mr.z/ode-workspace-prep/response-screen-20261115/preflight/documented-v1
```

Expected: kernel16 cases;16 exact native8->16 continuations/restorations;
the actual erf-K8 selection evaluator witness; PASS.json passed=true.
Softsign diagonal nonsmoothness at zero is explicitly excluded from the
smooth gradcheck premise. No other family's zero is skipped.

```bash
cd /Users/mr.z/ode-workspace-prep/response-screen-20261115/repo
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/response_screen_20261115/benchmark.py --data-root /Users/mr.z/ode-workspace-prep/data/synthetic --output /Users/mr.z/ode-workspace-prep/response-screen-20261115/preflight/benchmark-v1
```

Expected: benchmark PASS.json with two-worker throughput ratio and selected
concurrency. One and two workers each run native200step preflights in separate
folders, with no updates transferred into formal training. Report both wall
times. Source worktree must remain clean at its original HEAD.
