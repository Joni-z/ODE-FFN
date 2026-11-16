# Exact hidden solver-bias witness

Use the unchanged response-screen-cpu@04f11f75 environment with one CPU thread,
no packages or GPU. A fresh agent executes these four commands verbatim and
sequentially from /Users/mr.z/ode-workspace-prep/coefficient-20261116/solver-bias/repo.
First verify the four readiness output directories are absent. Do not modify
code, arguments, coefficients, tolerances, previous endpoints or output files.
Record actual terminal exits and document/reality divergence. Re-poll original
live handles, never restart on an observation timeout.

```sh
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/target_geometry_20261116/solver_bias_probe.py --energy /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-energy --surrogate /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-surrogate --dataset gmm_ring8_2d --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/solver-bias/readiness/ring --witness

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/target_geometry_20261116/verify_solver_bias.py --input /Users/mr.z/ode-workspace-prep/coefficient-20261116/solver-bias/readiness/ring --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/solver-bias/readiness/ring-verified

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/target_geometry_20261116/solver_bias_probe.py --energy /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-energy --surrogate /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-surrogate --dataset rotated_product_bimodal_16d --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/solver-bias/readiness/product --witness

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/target_geometry_20261116/verify_solver_bias.py --input /Users/mr.z/ode-workspace-prep/coefficient-20261116/solver-bias/readiness/product --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/solver-bias/readiness/product-verified
```

Both native COMPLETE files must pass witness=true,cases3,inputs_per_case80,
layers_per_case1,and levels4/8/16/32/64/128/256. Each independent verifier must
pass and bind its native COMPLETE, retain every state and reconstruct source
arrays, then check both finite-perturbation scales. Numerical qualification
may fail without an execution failure; report every unresolved point and all
hypothesis signs. A first-layer witness does not establish all-layer results.

Save readiness/command-exits.json with commands:[{command,actual_terminal_exit}],
divergences:[...],and the source freeze. The fresh witness performs no source
fix, package change or formal dispatch. Author guard-edit outputs are preserved
separately; only this frozen invocation can certify the documented path.

After a clean fresh pass, the executor runs native commands without --witness
into formal/ring and formal/product, one CPU thread each. Formal COMPLETE must
retain all six layers and all80 inputs per checkpoint. Run both independent
verifiers into formal/ring-verified and formal/product-verified. Preserve every
numeric failure, actual exit, layer, bank and resolution. This is not training
or generation evaluation. Final reports and the native image trainer are fixed.
