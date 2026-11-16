# Exact temporal-alias counterexample check

This is the complete twelve-checkpoint source audit, not a training run.
Reuse warm response-screen-cpu@04f11f75. In the frozen directory
`/Users/mr.z/ode-workspace-prep/coefficient-20261116/trajectory-response/alias-repo`,
check both output paths absent and run these two commands verbatim sequentially.
Do not install, fix, overwrite or start any other experiment.

```sh
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/target_geometry_20261116/time_alias_check.py --study /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order --data-root /Users/mr.z/ode-workspace-prep/data/synthetic --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/trajectory-response/time-alias/native

/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 studies/muon_pd_20261102/target_geometry_20261116/verify_time_alias.py --input /Users/mr.z/ode-workspace-prep/coefficient-20261116/trajectory-response/time-alias/native --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/trajectory-response/time-alias/verified
```

Both COMPLETE files must pass with twelve cases. Native rows bind1025grid
predictions, all original continuous selection-field pairs,16source endpoints
and the original128-pair/five-time Jacobians for each checkpoint. Grid/output
agreement is checked in both FP32andFP64with separately registered tolerances.
All original parameters/buffers must be restored after the single-frequency
substitution. Raw arrays and independently recomputed summaries must agree.

These commands cover the full registered counterexample, so their fresh pass
is retained as its canonical audit; do not repeat it as a separate formal run.
It shows what a sampling grid can distinguish, not whether changing a feature
improves an architecture or explains all numerical error.
