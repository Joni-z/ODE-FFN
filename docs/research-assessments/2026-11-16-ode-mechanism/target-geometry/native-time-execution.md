# Exact native-time precision commands

Warm environment response-screen-cpu@04f11f75 from .aris/compute/macmini.md.
No package changes, CUDA or MPS. Fresh source directory:
`/Users/mr.z/ode-workspace-prep/coefficient-20261116/native-time-resolution/repo`.

Run the following five witness commands verbatim and sequentially in that
directory. Check outputs absent first. Do not fix, overwrite or dispatch the
full queue during this fresh pass.

```sh
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/target_geometry_20261116/check_native_time.py --study /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order --pilot /Users/mr.z/ode-workspace-prep/coefficient-20261116/trajectory-response/pilot --data-root /Users/mr.z/ode-workspace-prep/data/synthetic --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/native-time-resolution/readiness/check

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/target_geometry_20261116/native_time_probe.py --study /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order --pilot /Users/mr.z/ode-workspace-prep/coefficient-20261116/trajectory-response/pilot --data-root /Users/mr.z/ode-workspace-prep/data/synthetic --case gmm_ring8_2d__palindrome_operator_midpoint4__adamw_0.0003 --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/native-time-resolution/readiness/ring --witness

/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 studies/muon_pd_20261102/target_geometry_20261116/verify_native_time.py --input /Users/mr.z/ode-workspace-prep/coefficient-20261116/native-time-resolution/readiness/ring --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/native-time-resolution/readiness/ring-verified

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/target_geometry_20261116/native_time_probe.py --study /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order --pilot /Users/mr.z/ode-workspace-prep/coefficient-20261116/trajectory-response/pilot --data-root /Users/mr.z/ode-workspace-prep/data/synthetic --case rotated_product_bimodal_16d__palindrome_operator_midpoint4__adamw_0.0003 --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/native-time-resolution/readiness/product --witness

/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 studies/muon_pd_20261102/target_geometry_20261116/verify_native_time.py --input /Users/mr.z/ode-workspace-prep/coefficient-20261116/native-time-resolution/readiness/product --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/native-time-resolution/readiness/product-verified
```

Expected check PASS: twelve complete cached native/packed endpoint and nine-time
trace checks on all256sources per model, two non-dyadic Gaussian/native recording
checks and an exact Gaussian packed control. Report measured packing throughput;
it is an execution result, not a model improvement.

Each native witness COMPLETE must pass with witness=true, eight sources/eight
banks and the specified case ID. It reuses the bound128/1024 endpoints and
actually evaluates1000/1032/2048/4096, plus8192only if the fine gate needs it.
Each independent verifier must pass and reproduce that witness's qualification.
A small witness cannot qualify the later256-source population.

After the fresh pass, the executor binds all five checks into readiness/READY.json
with source hashes. Then run the following in the same frozen directory:

```sh
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/target_geometry_20261116/native_time_queue.py --study /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order --pilot /Users/mr.z/ode-workspace-prep/coefficient-20261116/trajectory-response/pilot --data-root /Users/mr.z/ode-workspace-prep/data/synthetic --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/native-time-resolution/formal --readiness /Users/mr.z/ode-workspace-prep/coefficient-20261116/native-time-resolution/readiness/READY.json --detach
```

This starts at most two one-thread CPUworkers across all twelve cases. Verify
queue/child PIDs and actual status.json progress after launch. Each case retains
all256sources. Numerical failure is recorded separately from command failure;
the queue never retries, overwrites or omits a case. Independently verify all
twelve completed case folders after the queue's terminal receipt. No GPUused.
