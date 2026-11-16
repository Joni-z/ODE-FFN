# Exact native-reference qualification invocation

Use the unchanged warm response-screen-cpu@04f11f75environment, no installation
or accelerator. A fresh agent runs these four readiness commands verbatim and
sequentially from:

`/Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-surrogate/repo`

Confirm the four designated readiness output directories do not exist before
starting. Do not alter code, arguments, numerical gates or existing outputs.
Report every actual terminal exit, and any document/reality divergence.

```sh
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/target_geometry_20261116/hidden_surrogate_probe.py --study /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order --energy /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-energy --readout /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-readout --data-root /Users/mr.z/ode-workspace-prep/data/synthetic --dataset gmm_ring8_2d --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-surrogate/readiness/ring --witness

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 studies/muon_pd_20261102/target_geometry_20261116/verify_hidden_surrogate.py --input /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-surrogate/readiness/ring --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-surrogate/readiness/ring-verified

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/target_geometry_20261116/hidden_surrogate_probe.py --study /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order --energy /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-energy --readout /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-readout --data-root /Users/mr.z/ode-workspace-prep/data/synthetic --dataset rotated_product_bimodal_16d --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-surrogate/readiness/product --witness

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 studies/muon_pd_20261102/target_geometry_20261116/verify_hidden_surrogate.py --input /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-surrogate/readiness/product --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-surrogate/readiness/product-verified
```

Each native COMPLETE must pass with witness=true,cases3,inputs_per_case80,
layers_per_case1,and levels16/32/64/128/256. Each verifier must pass and bind
its native COMPLETE. Qualified levels may be empty: failed approximation
gates are valid scientific observations, not execution failures. The first
layer witness does not establish formal all-layer qualification.

Save an agent-owned command-exits.json under readiness with exact commands,
actual terminal exits and divergence notes. After the clean documented pass,
the executor runs the native commands without --witness into formal/ring and
formal/product, one CPUthread each, retaining all six layers. Run the same
independent verifier against each formal result into formal/ring-verified
and formal/product-verified. Record logs/actual exits; qualify a shared level
only from the intersection across both complete independent summaries.

No generation samples or training are launched by these commands. Any later
generation intervention requires a separate fixed protocol and on-trajectory
resolution checks. Reports1/2and all native image source stay untouched.
