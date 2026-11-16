# Exact transported-error calibration commands

Warm CPU environment response-screen-cpu@04f11f75 from .aris/compute/macmini.md.
No packages, CUDA or MPS. Run the five witness commands verbatim and sequentially
from the frozen source directory:
`/Users/mr.z/ode-workspace-prep/coefficient-20261116/trajectory-response/repo`.
Check that outputs are absent first. Do not fix, overwrite or run the full pilot
in the fresh agent-follows-doc pass.

```sh
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/target_geometry_20261116/check_trajectory.py --study /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order --data-root /Users/mr.z/ode-workspace-prep/data/synthetic --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/trajectory-response/readiness/check

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/target_geometry_20261116/trajectory_probe.py --study /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order --data-root /Users/mr.z/ode-workspace-prep/data/synthetic --dataset gmm_ring8_2d --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/trajectory-response/readiness/ring --witness

/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 studies/muon_pd_20261102/target_geometry_20261116/verify_trajectory.py --input /Users/mr.z/ode-workspace-prep/coefficient-20261116/trajectory-response/readiness/ring --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/trajectory-response/readiness/ring-verified

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/target_geometry_20261116/trajectory_probe.py --study /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order --data-root /Users/mr.z/ode-workspace-prep/data/synthetic --dataset rotated_product_bimodal_16d --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/trajectory-response/readiness/product --witness

/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 studies/muon_pd_20261102/target_geometry_20261116/verify_trajectory.py --input /Users/mr.z/ode-workspace-prep/coefficient-20261116/trajectory-response/readiness/product --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/trajectory-response/readiness/product-verified
```

Expected check PASS: three exact Gaussian examples; analytic native-oracle JVPs
at seven times per task, including both endpoints; all twelve actual checkpoints
pass native FP32sample replay, FP64conversion, lambda1identity, augmented-oracle
identity and finite differences of the native discrete flow. Witness native
COMPLETE: passed=true,witness=true,six cases/eight sources/eight banks. Verified
COMPLETE must pass with witness=true and six reconstructed cases.
All registered blend values and adaptive resolution levels are retained.
Numerical qualification of a small witness does not qualify the full pilot.

After the fresh pass, the executor runs the two trajectory_probe.py commands
with `--witness` removed and outputs
`.../trajectory-response/pilot/ring` and `.../trajectory-response/pilot/product`.
They may run concurrently at one CPUthread each. Each task retains six models,
256sources and all eight banks, stopping numerical refinement only at the
registered pass or1024step cap. Save logs and process identities. After each
native exit0, run its verifier into `pilot/ring-verified` or
`pilot/product-verified`. A COMPLETE with passed=true means the observation
finished; scientific qualification is separately reported as all_qualified.
