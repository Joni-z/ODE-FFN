# Frozen-weight measurements for the matched order control

Status: observer routing is implemented and step16native checkpoint restoration
passes for all four new families/tasks. Full export and observer-entry validation
wait for all four20k trajectories and their independent common-bank evaluation.
Do not treat the step16builder check as a completed population observation.

Use the unchanged Mac CPU environment and one-thread settings from
[the training invocation](execution.md). The training runtime stays frozen at
67b4664. A separate published observer snapshot will bind the following code.
No additional training, response search or density run is authorized here.

## Prerequisite

`/Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order/analysis/COMPLETE.json`
must exist and pass for four new trajectories plus eight fixed references.
All following output paths must be absent. From
`/Users/mr.z/ODE-FFN-upstream`, run these commands verbatim in sequence:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/coefficient_20261116/contrast_portable.py --study /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order --data-root /Users/mr.z/ode-workspace-prep/data/synthetic --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order/portable --schedule-order --sampling-only
```

Require four portable cases and only `sampling-models`, never a density bundle.
The exporter must replay the complete CPU field and bank0exactly, preserve
nonpersistent buffers, copy/checksum all eight bank outputs and bind every
imported local model source. Its actual portable checkpoint restore must match
the native restore. Source mismatch must fail rather than rewrite a binding.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/coefficient_20261116/sharing_probe.py --root /Users/mr.z/ode-workspace-prep/coefficient-20261116 --data-root /Users/mr.z/ode-workspace-prep/data/synthetic --schedule-study /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order --arrays-output /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order/activity-arrays --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order/activity.json
```

Require six cases: shared P and the two schedules on both tasks. Retain actual
entrances/operators, local brackets at five fixed times, complete field and
frozen mean/reverse/base interventions, training curves and native gradients.
The B-gradient group is reported separately. These are mechanism observations,
not substitute retraining. The palindrome reverse map should be identical;
the leading bracket cancellation need not remove its higher-order stage action.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 studies/muon_pd_20261102/coefficient_20261116/verify_schedule.py --probe /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order/activity.json --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order/activity-verified.json
```

Require independent NumPy reconstruction of all six models,180layer/time
records and five horizons per record. Do not run the old family-specific
`analyze_schedule.py` plotter on these new families.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/coefficient_20261116/contrast_cost.py --study /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order --data-root /Users/mr.z/ode-workspace-prep/data/synthetic --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order/cost.json
```

Require twelve same-process cases. Shared P and both schedules must have equal
dense MACs; the two schedules must have equal parameter counts. Report actual
training/evaluation time alongside these CPU timings, which are not GPU speed.

## Subsequent numerical observations

Before any GPU use, check145physical4-7for memory below500MiB and no compute
PID, bind one GPU UUID per process, and run the actual `gpu_resolution.py
--witness` on a frozen published observer tree. Never use0-3or share/stop a
foreign job. Recheck ownership while running; preserve any interrupted attempt.
Document the exact CUDA invocation and obtain a fresh agent-follows-doc witness.

The existing full-bank entry checks five registered configurations per model:
(inner4,Heun128/256/512), (inner8,Heun128), (inner16,Heun128). Every configuration
keeps all eight4096-source banks. Inner refinement preserves the four stage
boundaries. These are one-axis sweeps, not a full Cartesian grid or joint-limit
proof. Use the existing shared-P/ordinary results only with their same bound
arrays and numerical settings. Independently recompute pooled and paired bank
metrics from the retained arrays before interpreting order effects.
