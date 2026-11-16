# Exact single-layer output-intervention invocation

Use warm response-screen-cpu@04f11f75, without packages, CUDA or MPS. Run the
following five readiness commands verbatim and sequentially from:
`/Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-readout/repo`.
First confirm all five designated output directories are absent. No fixes,
improvisation, overwrites or formal dispatch in the fresh agent-follows-doc pass.

```sh
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/target_geometry_20261116/hidden_readout_probe.py --study /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order --energy /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-energy --data-root /Users/mr.z/ode-workspace-prep/data/synthetic --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-readout/readiness/native-check --check-only

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/target_geometry_20261116/hidden_readout_probe.py --study /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order --energy /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-energy --data-root /Users/mr.z/ode-workspace-prep/data/synthetic --dataset gmm_ring8_2d --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-readout/readiness/ring --witness

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 studies/muon_pd_20261102/target_geometry_20261116/verify_hidden_readout.py --input /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-readout/readiness/ring --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-readout/readiness/ring-verified

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/target_geometry_20261116/hidden_readout_probe.py --study /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order --energy /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-energy --data-root /Users/mr.z/ode-workspace-prep/data/synthetic --dataset rotated_product_bimodal_16d --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-readout/readiness/product --witness

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 studies/muon_pd_20261102/target_geometry_20261116/verify_hidden_readout.py --input /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-readout/readiness/product --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-readout/readiness/product-verified
```

The native check must pass with cases6,inputs_per_case80,layers_per_case6,
check_only=true. Every layer preserves the native identity output and qualifies
the ray potential. It measures no nonidentity full-output effects. Witness
commands retain all80inputs but change only the first local layer in each of
three models; COMPLETE must pass with witness=true,layers_per_case1. All five
modes are measured there. Both independent verifiers must pass and bind the
native outputs. No formal six-layer result is implied by the witness.

After the clean source-bound readiness pass, the executor runs the two native
dataset commands without --witness into hidden-readout/formal/ring and
hidden-readout/formal/product, one CPUthread each. Each formal dataset keeps
all three models, six separate layer interventions and every80input. Save logs
and actual exits; then run the corresponding verifier into formal/ring-verified
and formal/product-verified. Checkpoint data and previous energy arrays remain
read only. This is not training or a sampling-performance comparison.
