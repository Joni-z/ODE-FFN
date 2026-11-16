# Exact first-layer sampling witness

Warm response-screen-cpu@04f11f75, one CPUthread, no packages or accelerator.
A fresh agent follows these five commands verbatim and sequentially from:

`/Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-sampling/repo`

First confirm all five readiness output directories below are absent. Do not
fix code, change arguments/gates or overwrite any result. Record actual terminal
exits and document/reality divergences. Numerical hypothesis failures are valid
observations; a witness does not have to produce favorable sampling scores.

```sh
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/target_geometry_20261116/hidden_sampling_probe.py --study /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order --surrogate /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-surrogate --data-root /Users/mr.z/ode-workspace-prep/data/synthetic --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-sampling/readiness/native-check --check-only

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/target_geometry_20261116/hidden_sampling_probe.py --study /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order --surrogate /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-surrogate --data-root /Users/mr.z/ode-workspace-prep/data/synthetic --dataset gmm_ring8_2d --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-sampling/readiness/ring --witness

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 studies/muon_pd_20261102/target_geometry_20261116/verify_hidden_sampling.py --input /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-sampling/readiness/ring --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-sampling/readiness/ring-verified

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/target_geometry_20261116/hidden_sampling_probe.py --study /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order --surrogate /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-surrogate --data-root /Users/mr.z/ode-workspace-prep/data/synthetic --dataset rotated_product_bimodal_16d --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-sampling/readiness/product --witness

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 studies/muon_pd_20261102/target_geometry_20261116/verify_hidden_sampling.py --input /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-sampling/readiness/product --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/hidden-sampling/readiness/product-verified
```

The first COMPLETE must pass check_only=true,cases6,first_layer=blocks.0.ffn.ode1;
it exactly reconstructs every80input first-layer candidate and full output.
Each actual sampler witness must pass witness=true,cases3,per_bank1,and use
inner256/512and outer128/129. It contains one shared-Pmodel's18arms and two
ordinary controls' two native arms each, with all eight original input banks.
Each independent verifier must pass and bind its native COMPLETE. Its numerical
and hypothesis outcomes may fail without execution failure; retain them all.

Save readiness/command-exits.json with commands:[{command,actual_terminal_exit}],
divergences:[...],and the source freeze. Do not launch the formal pilot.

After that clean pass, the executor runs the native dataset commands without
--witness into formal/ring and formal/product, one worker per task. Formal
COMPLETE must have witness=false,cases5,per_bank8: all three focal models plus
both ordinary controls,64sources each. Preserve all18focal arms, both grids,
both numerical levels and all DOPsnapshots, even when a gate fails. Run the
corresponding verifiers into formal/ring-verified and formal/product-verified.
Record logs, original tool handles and actual terminal exits. This remains a
bounded descriptive pilot, not a full32768sample or image-performance result.
