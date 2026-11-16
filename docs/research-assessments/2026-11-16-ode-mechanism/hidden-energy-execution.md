# Exact hidden-energy invocation

Use warm response-screen-cpu@04f11f75. No packages, CUDA or MPS. Run these five
readiness commands verbatim and sequentially from the frozen directory:
`/Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-energy/repo`.
First check that all five output directories are absent. The fresh
agent-follows-doc pass must not fix code, overwrite outputs or launch formal work.

```sh
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/target_geometry_20261116/hidden_energy_probe.py --study /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order --data-root /Users/mr.z/ode-workspace-prep/data/synthetic --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-energy/readiness/native-check --check-only

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/target_geometry_20261116/hidden_energy_probe.py --study /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order --data-root /Users/mr.z/ode-workspace-prep/data/synthetic --dataset gmm_ring8_2d --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-energy/readiness/ring --witness

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 studies/muon_pd_20261102/target_geometry_20261116/verify_hidden_energy.py --input /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-energy/readiness/ring --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-energy/readiness/ring-verified

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/target_geometry_20261116/hidden_energy_probe.py --study /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order --data-root /Users/mr.z/ode-workspace-prep/data/synthetic --dataset rotated_product_bimodal_16d --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-energy/readiness/product --witness

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 studies/muon_pd_20261102/target_geometry_20261116/verify_hidden_energy.py --input /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-energy/readiness/product --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-energy/readiness/product-verified
```

The native check COMPLETE must pass with check_only=true,cases=6and
points_per_layer=80. It checks all36layers and four midpoint resolutions
against native Torch outputs. Both actual witness COMPLETE markers must have
passed=true,witness=true,cases=3,points_per_layer=8. Each independent COMPLETE
must pass and bind its native summary. Numerical qualification is reported
separately; no unresolved reference state may be dropped or rescued.

After this pass the executor creates a source-bound readiness record and runs
the two native dataset commands without --witness into hidden-energy/formal/ring
and hidden-energy/formal/product. Each keeps three models, six layers/model and
all80registered input states/layer. They may run concurrently at one CPUthread
each. Save logs and actual process exits. After each actual exit0, run the same
independent verifier into formal/ring-verified or formal/product-verified.
This is a bounded frozen-weight measurement, not new training or generation.
