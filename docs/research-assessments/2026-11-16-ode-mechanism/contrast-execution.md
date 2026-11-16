# Mean-preserving stage release: native execution

The [registered control](sharing-optimization-control.md) changes only the
coordinates of the four effective stage matrices. Keep the ISRU response,
four midpoint steps, eight matrix applications, seed 0, batch 512, 20k updates,
AdamW 0.0003, weight decay 0 and native gradient clipping 1.

The new model has the same function class and parameter count as the anchored
release: 130,434 parameters on Ring8 and 132,240 on Product16. It adds 73,728 to
the shared model. Root image training, its native baseline and Reports 1/2 are
unchanged. Both new trajectories use Mac CPU, one thread each, matching the
existing references' host/backend. Do not allocate a GPU for these commands.

## Source and environment

Publish the verified source, then create the detached worktree at
`/Users/mr.z/ode-workspace-prep/coefficient-20261116/mean-preserving/repo`.
Reuse `/Users/mr.z/ode-workspace-prep/.venv-test/bin/python`; package/spec
response-screen-cpu@04f11f75 is unchanged. The native source hash checks remain
mandatory. No dependencies are installed or upgraded.

## Fresh documented witness

A fresh agent executes this once verbatim after the frozen worktree exists:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python -u /Users/mr.z/ode-workspace-prep/coefficient-20261116/mean-preserving/repo/studies/muon_pd_20261102/coefficient_20261116/contrast_check.py --data-root /Users/mr.z/ode-workspace-prep/data/synthetic --assets /Users/mr.z/ode-workspace-prep/coefficient-20261116/sharing-assets --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/mean-preserving/doc-witness
```

Require exit 0 and PASS.json with two kernel cases, two exact native 8-to-16
continuations and two actual native evaluator witnesses. Kernels require:
identical initial outputs/input gradients; common-gradient agreement at
atol 1e-14 / rtol 1e-10; active contrasts; arbitrary nonzero-stage coordinate
conversion; gradcheck; a controlled common-optimizer update; and mean(W)=P
after the real native optimizer with global clipping. Native clip factors
are recorded, not assumed equal. The witness uses a nonzero readout probe
to avoid trivial zero gradients, but does not alter formal initialization.

## Bind the original references before training

After that witness passes, run once:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python /Users/mr.z/ode-workspace-prep/coefficient-20261116/mean-preserving/repo/studies/muon_pd_20261102/coefficient_20261116/contrast_prepare.py --root /Users/mr.z/ode-workspace-prep/coefficient-20261116 --study /Users/mr.z/ode-workspace-prep/coefficient-20261116/mean-preserving --witness /Users/mr.z/ode-workspace-prep/coefficient-20261116/mean-preserving/doc-witness
```

It binds all eight existing checkpoints/configs/evaluations and checks both
new native step 0 checkpoints against the original shared formal initial state
and data RNG. Reuse the already bound eight 4,096-sample banks. Historical
selection/confirmation filenames stay unchanged; this is a matched control,
not a fresh model-selection confirmation. The primary checkpoint is 20k.

## Launch two native trainings with the existing monitor

Run once after readiness and anchor binding:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python -u /Users/mr.z/ode-workspace-prep/coefficient-20261116/mean-preserving/repo/studies/muon_pd_20261102/run_queue.py --host macmini --device cpu --data-root /Users/mr.z/ode-workspace-prep/data/synthetic --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/mean-preserving/output --workers 2 --grid /Users/mr.z/ode-workspace-prep/coefficient-20261116/mean-preserving/repo/studies/muon_pd_20261102/coefficient_20261116/contrast-grid.json --runner /Users/mr.z/ode-workspace-prep/coefficient-20261116/mean-preserving/repo/studies/muon_pd_20261102/coefficient_20261116/contrast_trial.py --experiment-id mean_preserving_stages_20261116 --detach
```

Record actual queue/worker PIDs and verify live processes plus increasing
steps in output/monitor/status.json. Never duplicate a live queue after a
timeout. Check terminal exit 0 for both workers, native final samples and
complete 0/15k/20k checkpoints.

Start `contrast_finish.py --study <the study above> --data-root <the same
cache> --numpy-python /Library/Frameworks/Python.framework/Versions/3.12/bin/python3
--queue-pid <actual queue PID>` as a separate monitored process. It waits for
the actual queue, checks formal recipe/initialization/data-RNG pairing,
reuses the eight hash-bound completed evaluations through symlinks, and runs
only the four new evaluation invocations. The existing NumPy collector then
reconstructs all ten methods/tasks on both evaluation splits. It stops on
failure, preserves logs and never retries training automatically.

## Completion and decisions

Execution completion is not a result: independently recompute every new
bank/pooled SW2 and field MSE, show both tasks and all ordinary controls, and
compare training fit, stage action and measured cost. No response, optimizer,
checkpoint or data stream may be selected after seeing these scores.
If the mean-preserving release closes the old gap, revise the sharing story
toward optimizer coordinates. If it does not, the sharing constraint survives
this particular coordinate control. Keep the separate density/SW2 distinction.
