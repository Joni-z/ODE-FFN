# Exact unchanged-gate sampling-extension witness

Use warm response-screen-cpu@04f11f75, one CPU thread, no package installation.
A fresh agent executes the following five commands verbatim and sequentially
from /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-sampling-extension/repo.
First verify all five readiness destinations are absent. Do not change code,
arguments, numerical gates, data, reused arrays or prior results. Record actual
terminal exits and document/reality divergence; re-poll original live handles.

```sh
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/target_geometry_20261116/hidden_sampling_probe.py --study /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order --surrogate /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-surrogate --data-root /Users/mr.z/ode-workspace-prep/data/synthetic --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-sampling-extension/readiness/native-check --check-only --extend1024-from /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-sampling

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/target_geometry_20261116/hidden_sampling_probe.py --study /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order --surrogate /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-surrogate --data-root /Users/mr.z/ode-workspace-prep/data/synthetic --dataset gmm_ring8_2d --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-sampling-extension/readiness/ring --witness --extend1024-from /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-sampling

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 studies/muon_pd_20261102/target_geometry_20261116/verify_hidden_sampling.py --input /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-sampling-extension/readiness/ring --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-sampling-extension/readiness/ring-verified

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/target_geometry_20261116/hidden_sampling_probe.py --study /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order --surrogate /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-surrogate --data-root /Users/mr.z/ode-workspace-prep/data/synthetic --dataset rotated_product_bimodal_16d --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-sampling-extension/readiness/product --witness --extend1024-from /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-sampling

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 studies/muon_pd_20261102/target_geometry_20261116/verify_hidden_sampling.py --input /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-sampling-extension/readiness/product --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-sampling-extension/readiness/product-verified
```

The first COMPLETE must pass check_only=true,cases6: all six saved80-input
candidate/output replays remain exact. It does not observe new1024-step sampling.
Each actual witness must pass witness=true,cases3,per_bank1,inner_steps512/1024
and Heun128/129. It retains22arms:18focal and2native per ordinary control;
14are source-bound reused native/512 arms and8are new1024 arms. Both numerical
and directional outcomes may fail without an execution failure. Each independent
verifier must pass and bind its native COMPLETE and all copied-source hashes.

Save readiness/command-exits.json with commands:[{command,actual_terminal_exit}],
divergences:[...],and the source freeze. No fixes or formal dispatch by the fresh
witness. The unchanged-gate protocol is hidden-sampling-extension-protocol.md.

After a clean fresh pass, the executor runs native dataset commands without
--witness into formal/ring and formal/product. Each formal task has5cases and
per_bank8, all three focal plus two ordinary controls. Both datasets together
retain116arms:68copied exactly from the original formal pilot and48new1024
arms. Run both independent verifiers into formal/ring-verified and
formal/product-verified. Do not overwrite or reinterpret the original256
failures. Keep two one-thread workers and save actual exit receipts/logs.
