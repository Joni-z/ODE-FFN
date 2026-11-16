# Exact all-bank confirmation execution

Warm CPU environment response-screen-cpu@04f11f75; unchanged packages and thread
settings. Fixed source directory:
`/Users/mr.z/ode-workspace-prep/coefficient-20261116/target-geometry/repo-confirmation`.
Run all four witness commands below verbatim in that directory, sequentially.
Outputs must be absent; never fix or overwrite a failed documented invocation.

```sh
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/target_geometry_20261116/frozen_probe.py --study /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order --data-root /Users/mr.z/ode-workspace-prep/data/synthetic --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/target-geometry/confirmation-readiness/ring --confirmation-dataset gmm_ring8_2d --witness

/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 studies/muon_pd_20261102/target_geometry_20261116/verify_probe.py --input /Users/mr.z/ode-workspace-prep/coefficient-20261116/target-geometry/confirmation-readiness/ring --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/target-geometry/confirmation-readiness/ring-verified

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/target_geometry_20261116/frozen_probe.py --study /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order --data-root /Users/mr.z/ode-workspace-prep/data/synthetic --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/target-geometry/confirmation-readiness/product --confirmation-dataset rotated_product_bimodal_16d --witness

/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 studies/muon_pd_20261102/target_geometry_20261116/verify_probe.py --input /Users/mr.z/ode-workspace-prep/coefficient-20261116/target-geometry/confirmation-readiness/product --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/target-geometry/confirmation-readiness/product-verified
```

Each native/verified COMPLETE must pass with witness=true, two cases and two
time rows. Each raw file has32pairs:4from every one of eight confirmation banks,
time.5. Do not treat these witness scores as the full confirmation.

After the fresh pass, the executor runs the two native commands above with
`--witness` removed and outputs `.../target-geometry/confirmation/ring` and
`.../target-geometry/confirmation/product`. They may run concurrently at one
CPUthread each. Expect two cases/ten time rows per task,32768pairs per time,
five times and banks0-7all present. Run the separate verifier into
`confirmation/ring-verified` and `confirmation/product-verified` after each
native process exits0. Preserve logs and exact actual exit receipts.
