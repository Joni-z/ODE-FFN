# Exact matched-solver training readiness

Use warm response-screen-cpu@04f11f75, one CPU thread, no package installation
or accelerator. A fresh agent runs the following command verbatim from:

`/Users/mr.z/ode-workspace-prep/coefficient-20261116/rk4-control/repo`

First confirm readiness/check does not exist. Do not change source, arguments,
recipe, data or outputs. Run to an actual terminal exit; re-poll the same handle
if the observation yields. No fixes or full training dispatch by the witness.

```sh
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/coefficient_20261116/rk4_check.py --data-root /Users/mr.z/ode-workspace-prep/data/synthetic --assets /Users/mr.z/ode-workspace-prep/coefficient-20261116/sharing-assets --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/rk4-control/readiness/check
```

Require PASS.json passed=true, native_continuations2, real_evaluator_witnesses2.
Both kernel dimensions must have exact initialized parameters/buffers/RNG,
an independent RK4 formula check, complete input/projection gradient checks,
fourth-order refinement and matrix_applications[8,8]. Each dataset's native
8->16 continuation must match continuous16, and the training data-generator
state at8 must match the unchanged midpoint control. The local initialized
maps need not match: the solver is the intended difference. No quality claim
is made from16-step checks.

Save readiness/command-exits.json with commands:[{command,actual_terminal_exit}],
divergences:[...],and source freeze. Report actual parameter counts, errors,
child continuation/evaluator results and any document/reality divergence.

After this clean frozen pass, the executor may run the two registered native
20k commands, omitting --steps, under the same source and CPU environment:

```sh
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/coefficient_20261116/rk4_trial.py --cell gmm_ring8_2d__state_rk4_2__adamw_0.0003 --data-root /Users/mr.z/ode-workspace-prep/data/synthetic --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/rk4-control/training --device cpu

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/coefficient_20261116/rk4_trial.py --cell rotated_product_bimodal_16d__state_rk4_2__adamw_0.0003 --data-root /Users/mr.z/ode-workspace-prep/data/synthetic --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/rk4-control/training --device cpu
```

Use two one-thread workers and preserve actual exits, native metrics,
update_spectra.jsonl, all milestone checkpoints and ADAPTER_COMPLETE. Monitor
live identities and progress; do not restart on timeout. A20k completion and
standard final test remain required before confirmation evaluation. No image
training, baseline replacement, remote job sharing or report edit is authorized
by this readiness command. The explicit next evaluation stages are in the
registered protocol and use only rk4_evaluate.py's builder callback.
