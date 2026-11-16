# Exact refined-sensitivity commands

Use warm response-screen-cpu@04f11f75and no package changes. Run the five
commands below verbatim and sequentially from:
`/Users/mr.z/ode-workspace-prep/coefficient-20261116/refined-trajectory/repo`.
Check that their five output directories are absent first. The fresh
agent-follows-doc pass must not fix code, overwrite outputs or launch formal work.

```sh
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/target_geometry_20261116/check_native_time.py --study /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order --pilot /Users/mr.z/ode-workspace-prep/coefficient-20261116/trajectory-response/pilot --data-root /Users/mr.z/ode-workspace-prep/data/synthetic --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/refined-trajectory/readiness/check

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/target_geometry_20261116/refined_trajectory_probe.py --study /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order --pilot /Users/mr.z/ode-workspace-prep/coefficient-20261116/trajectory-response/pilot --native /Users/mr.z/ode-workspace-prep/coefficient-20261116/native-time-resolution/verified --data-root /Users/mr.z/ode-workspace-prep/data/synthetic --case gmm_ring8_2d__palindrome_operator_midpoint4__adamw_0.0003 --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/refined-trajectory/readiness/ring --witness

/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 studies/muon_pd_20261102/target_geometry_20261116/verify_refined_trajectory.py --input /Users/mr.z/ode-workspace-prep/coefficient-20261116/refined-trajectory/readiness/ring --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/refined-trajectory/readiness/ring-verified

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/target_geometry_20261116/refined_trajectory_probe.py --study /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order --pilot /Users/mr.z/ode-workspace-prep/coefficient-20261116/trajectory-response/pilot --native /Users/mr.z/ode-workspace-prep/coefficient-20261116/native-time-resolution/verified --data-root /Users/mr.z/ode-workspace-prep/data/synthetic --case rotated_product_bimodal_16d__palindrome_operator_midpoint4__adamw_0.0003 --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/refined-trajectory/readiness/product --witness

/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 studies/muon_pd_20261102/target_geometry_20261116/verify_refined_trajectory.py --input /Users/mr.z/ode-workspace-prep/coefficient-20261116/refined-trajectory/readiness/product --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/refined-trajectory/readiness/product-verified
```

The first command must produce PASS.jsonwith twelve exact cached native/packed
endpoint-and-path checks and Gaussian/non-dyadic controls. Each native witness
COMPLETE must have passed=true,witness=true,eight sources/eight banks; each
NumPyCOMPLETE must pass and bind its native summary. Scientific qualification
is separate: an unresolved8192row is preserved and is not an execution failure.
Where native fine-grid arrays exist, packed lambda1endpoints and all nine
recorded states must be exactly identical. No weights/buffers may change.

After a clean fresh pass, the executor creates a source-bound readiness.json,
then launches the existing two-worker monitored queue from the same freeze:

```sh
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/target_geometry_20261116/native_time_queue.py --study /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order --pilot /Users/mr.z/ode-workspace-prep/coefficient-20261116/trajectory-response/pilot --refined-native /Users/mr.z/ode-workspace-prep/coefficient-20261116/native-time-resolution/verified --data-root /Users/mr.z/ode-workspace-prep/data/synthetic --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/refined-trajectory/formal --readiness /Users/mr.z/ode-workspace-prep/coefficient-20261116/refined-trajectory/readiness/READY.json --detach
```

Each of twelve cases uses all256original sources and2048/4096, then8192only
if needed. Check the actual queue/child PIDs, retain logs and never restart
because a monitoring window expires. When terminal, all twelve actual child
exit codes and complete markers are required before the independent collector:

```sh
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 studies/muon_pd_20261102/target_geometry_20261116/collect_native_time.py --input /Users/mr.z/ode-workspace-prep/coefficient-20261116/refined-trajectory/formal --study /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/refined-trajectory/verified --refined --wait
```
