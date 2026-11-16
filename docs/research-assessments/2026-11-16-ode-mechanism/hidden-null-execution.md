# Exact dimension-matched intervention witness

Use warm response-screen-cpu@04f11f75 with one CPU thread and no package
installation. A fresh agent runs the following four commands verbatim,
sequentially, from:

`/Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-null/repo`

First confirm all four readiness output directories are absent. Do not alter
code, arguments, frames, thresholds or existing outputs. Preserve actual
terminal exits and document/reality divergence. A rejected hypothesis is an
experimental result, not a failed execution.

```sh
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/target_geometry_20261116/hidden_null_probe.py --study /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order --frames /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-null/frames --data-root /Users/mr.z/ode-workspace-prep/data/synthetic --dataset gmm_ring8_2d --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-null/readiness/ring --witness

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 studies/muon_pd_20261102/target_geometry_20261116/verify_hidden_null.py --input /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-null/readiness/ring --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-null/readiness/ring-verified

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/target_geometry_20261116/hidden_null_probe.py --study /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order --frames /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-null/frames --data-root /Users/mr.z/ode-workspace-prep/data/synthetic --dataset rotated_product_bimodal_16d --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-null/readiness/product --witness

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 studies/muon_pd_20261102/target_geometry_20261116/verify_hidden_null.py --input /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-null/readiness/product --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-null/readiness/product-verified
```

Each native COMPLETE must pass witness=true, cases3, inputs_per_case80,
layers_per_case1 and directions64. It uses every direction in the first layer
of all three models for that dataset. Each independent verifier must pass,
reconstruct all bound frames and candidates, and bind its native COMPLETE.
Intervals and both hypothesis outcomes can have either sign.

Save readiness/command-exits.json with commands:[{command,actual_terminal_exit}],
divergences:[...],and the source freeze. No fixes or formal dispatch by the
fresh witness. Read and re-poll original running handles to actual terminal
exits; a timeout does not authorize a restart.

After a clean source-bound pass, the executor runs both native commands without
--witness into formal/ring and formal/product, one CPU thread per dataset.
Formal results must have witness=false and layers_per_case6, keeping all
80 inputs and64 directions per layer. Run the corresponding verifiers into
formal/ring-verified and formal/product-verified. Retain every case, contrast,
frame and direction regardless of the outcome. No training, sampling extension,
image-code change or finalized-report change is performed by these commands.
