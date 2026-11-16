# Exact learned-field transport readiness invocation

Warm response-screen-cpu@04f11f75, unchanged packages. A fresh agent reads the
compute contract and provider ledger, then executes BOTH commands verbatim from:

/Users/mr.z/ode-workspace-prep/coefficient-20261116/target-aware-transport/repo

The frozen archive has no.git; source-freeze.json in its parent binds all files.
Confirm readiness absent before the first command. Do not repair code, arguments,
weights or packages, retrain, regenerate banks, overwrite outputs or restart a
live process on observation timeout. Poll each original handle to actual exit.

```sh
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/target_transport_20261117/check_math.py --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/target-aware-transport/readiness/math.json
```

```sh
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/target_transport_20261117/run_study.py --cases docs/research-assessments/2026-11-16-ode-mechanism/cross-solver-cases.json --data-root /Users/mr.z/ode-workspace-prep/data/synthetic --assets /Users/mr.z/ode-workspace-prep/coefficient-20261116/target-aware-transport/assets --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/target-aware-transport/readiness/actual --workers 4 --witness
```

Require math passed=true and TARGET_TRANSPORT_MATH_PASS; eight actual native
and eight independent exits0, TERMINAL passed=true,all_numerically_qualified=true.
Each witness keeps one bank/64sources with the actual Heun128 and all four
lambdas, exact original native Heun32 anchor, exact augmented FP64 primal/path,
learned JVP and whole-sampler/target-functional finite differences under the
registered gates. The NumPy verifier reconstructs each retained outcome.
This checks execution/numerics; no full-quality conclusion from64sources.

Own only readiness/command-exits.json (and an optional short witness note).
Record exact commands, actual terminal exits, divergences and source commit.
Report failed gates without changing/removing the case. Full dispatch is a
separate parent action after this readiness passes and the source is frozen.
