# Exact full cached-output verification after input-binding repair

Use unchanged response-screen-cpu@04f11f75. A fresh agent reads the compute
contract/provider ledger and executes the one command below verbatim from:

/Users/mr.z/ode-workspace-prep/coefficient-20261116/rk4-control/cross-solver/verification-repo

First confirm verification-v2 absent. This audits the eight ALREADY completed
full native evaluations. Do not generate samples, retrain, resubmit, repair
source/arguments/packages or overwrite old verification outputs. Run to actual
terminal exit, polling the same handle if observation yields. The archive has
no .git. A generator queue's original overall exit 1 is expected and retained;
all eight native-generation child exits must independently be 0.

```sh
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/coefficient_20261116/verify_cached_cross_solver.py --root /Users/mr.z/ode-workspace-prep/coefficient-20261116/rk4-control/cross-solver --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/rk4-control/cross-solver/verification-v2
```

Require PASS.json passed=true, cases8, native_actual_exits eight0, actual_child_exits
sixteen 0, original_samples_and_outputs_unchanged=true. Every precision reference
must bind its actual split, checkpoint, solver, asset and field_x/time hashes;
every per-output verifier must have witness=false and all eight 4096-source banks.
The original precision threshold remains 3e-5 absolute/relative for all 4,096 field
predictions; independent field/sample/case metric error must remain below 1e-10.
Original native 20k Heun32 anchors must still be exact before the saved substitution.

Own only readiness-v2/command-exits.json and an optional short witness note.
Record commands:[{command,actual_terminal_exit}],divergences:[],and the new
source-freeze-v2.json. Write the durable receipt promptly. The command preserves
all sixteen child exits and original-output hashes itself. This is also the
complete cached-cohort verification; no further sampling dispatch is authorized.
Report all divergences and preserve any failed point without narrowing the cohort.
