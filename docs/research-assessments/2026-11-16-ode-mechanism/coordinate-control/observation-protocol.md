# New-coordinate checkpoints: solver and density observations

This is the registered secondary measurement of the two completed
mean-preserving models. It adds no training, response function, optimizer
setting or image experiment. Keep the two fixed 20k checkpoints, seed 0,
native prediction, loss and sampling mathematics. Reports 1/2 stay finalized.

## Questions and decision rules

1. Does the shared-versus-new SW2 comparison survive more accurate numerical
   integration? Evaluate internal midpoint substeps 4/8/16 at outer Heun128,
   and outer Heun128/256/512 at internal4, on all eight original 4,096-source
   sharing banks. Retain every bank and all point estimates. Inner and outer
   sweeps are separate interventions, not a full factorial grid. Report
   endpoint shifts, paired scores and dense work. A changed ordering weakens
   the corresponding finite-step sharing claim. Do not select a favorable
   numerical setting as a new model.
2. Does the new release change the joint-density discrepancy or component
   widths relative to shared and anchored P? Reuse the original four P2
   banks of 256 initial points, both original and projected fields. Copy the
   arrays byte-for-byte from the completed reference population; regenerate
   only new-model CPU reference outputs and derivatives. Preserve all old
   sixteen model/field observations and their qualified source resolutions.

The primary density contrasts are new minus shared on both tasks and fields;
new minus anchored gives four secondary contrasts. Report all eight, with
10,000 within-bank paired source resamples using RNG0 and 99.375% pointwise
percentile intervals (Bonferroni allocation for these eight displayed
contrasts). These are conditional frozen-function Monte Carlo comparisons.
The earlier four-contrast family and its intervals remain unchanged.
No time window, source subset, checkpoint or setting is selected from the
confirmation time profiles. Keep ordinary controls in the comparison.

If the new model has lower SW2 but higher KL than a reference, report the
metric ordering directly and examine the preregistered component moments.
Do not call lower projected SW2 alone better full-joint density matching.
If sharing remains better in both after this coordinate control, retain the
stationary-generator hypothesis for that task. Neither result establishes
an advantage over an ordinary recurrent computation with the same map.

## Numerical requirements

Reuse the unchanged `density_ops.py` observer and native sampler. Start all
new population paths at1024/2048steps. Retain every source. For unresolved
sources only, use4096/8192/16384with the original per-dimension chain and
local-logdet tolerances0.001, logratio-refinement0.002, relative endpoint
refinement0.001and zero nonpositive local Jacobian steps. Do not drop sources
or relax a gate. Unresolved cells are reported as such, without a density
ranking. The density estimator concerns the continuous flow and requires
numerical qualification; it is not an exact finite-Heun-map likelihood.

## Source, inputs and compute

Routing update: the initial CPU launch described below is preserved. After
fresh CUDA readiness and an actual 7.968x combined speed measurement,
Product16 moved to physical GPU7 with its exact CPU state suspended as a
fallback. Ring8 stays on CPU. This is the explicitly recorded exception to
the initial no-duplication route; the entire GPU case must pass independent
and matched-row audits before superseding the CPU observation.
[Measured handoff](observations/resource-handoff.md).

Frozen worktree, created after verified publication:
`/Users/mr.z/ode-workspace-prep/coefficient-20261116/contrast-observations/repo`.
Inputs: `contrast-observations/input/{sampling-models,density-models}`;
density data: `contrast-observations/density-data` under the same parent.
The exporter checks the full native field and bank0samples against the
completed evaluation and binds all imported project-local model sources.
The portable loader and independent audit enforce those source hashes.
The existing density entries now derive their case count from the manifest;
the original eight-case population remains supported.

Reuse response-screen-cpu@04f11f75, Python3.12.4/Torch2.10.0/NumPy2.2.6,
one thread/process, no package changes. At most two density and two numerical
workers on Mac mini. On145physical0-3are forbidden and4-7are currently
occupied, so allocate no GPU. Do not stop or share foreign tasks. A future
CUDA observation requires a separately recorded idle-card/native witness;
it must not duplicate or restart a live CPU observation.

## Fresh documented CPU witness

Run the following four commands once, verbatim, stopping on failure. Output
folders must be absent. These are kernel/native-entry witnesses, not formal
population runs. The same two models must pass full Jacobians, native
endpoints and finite differences of the complete native sampler map.

```sh
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python -u /Users/mr.z/ode-workspace-prep/coefficient-20261116/contrast-observations/repo/studies/muon_pd_20261102/density_transport_20261116/check_formal.py --root /Users/mr.z/ode-workspace-prep/coefficient-20261116/contrast-observations/input/density-models --data /Users/mr.z/ode-workspace-prep/coefficient-20261116/contrast-observations/density-data --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/contrast-observations/doc-witness --device cpu --derivative-backend coordinate

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python -u /Users/mr.z/ode-workspace-prep/coefficient-20261116/contrast-observations/repo/studies/muon_pd_20261102/density_transport_20261116/check_map_logdet.py --root /Users/mr.z/ode-workspace-prep/coefficient-20261116/contrast-observations/input/density-models --data /Users/mr.z/ode-workspace-prep/coefficient-20261116/contrast-observations/density-data --entries /Users/mr.z/ode-workspace-prep/coefficient-20261116/contrast-observations/doc-witness/derivatives --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/contrast-observations/doc-map.json

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python -u /Users/mr.z/ode-workspace-prep/coefficient-20261116/contrast-observations/repo/studies/muon_pd_20261102/coefficient_20261116/cpu_resolution.py --root /Users/mr.z/ode-workspace-prep/coefficient-20261116/contrast-observations/input/sampling-models --case gmm_ring8_2d__contrast_operator_midpoint4__adamw_0.0003 --mode outer-subset --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/contrast-observations/doc-outer-ring8

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python -u /Users/mr.z/ode-workspace-prep/coefficient-20261116/contrast-observations/repo/studies/muon_pd_20261102/coefficient_20261116/cpu_resolution.py --root /Users/mr.z/ode-workspace-prep/coefficient-20261116/contrast-observations/input/sampling-models --case rotated_product_bimodal_16d__contrast_operator_midpoint4__adamw_0.0003 --mode outer-subset --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/contrast-observations/doc-outer-product16
```

Require all exits0, two formal witnesses, eight complete-map finite-difference
checks, two derivative-model checks and four numerical rows per task.
The full-map logdet tolerance remains5e-5. The bounded outer witnesses use
512sources; they do not qualify the full32768-source comparison.

## Formal observations after readiness

Use the existing `formal_dispatch.py` with these frozen inputs, `--phase
benchmark --indices 0,1 --device cpu --workers 1 --derivative-backend coordinate
--readiness <doc-witness/COMPLETE.json>` for sequential256-source timing. Then
`--phase formal --indices 0,1 --device cpu --workers 2 --detach` with the same
readiness, separate output folder and unchanged inputs. Formal outputs must
contain32new rows, each retaining256sources, and exact native endpoints.

Use the existing `resolution_queue.py --root <sampling-models> --output
<new numerical folder> --only numerical --full-outer --inner-substeps 4 8 16`.
This creates four monitored numerical tasks, without symmetry-task additions.
Require full32768sources for each formal sweep and independent NumPy bank
and pooled score reconstruction. Preserve queue PID, live progress, terminal
worker exits, frozen commit, data/model hashes and CPU load history.
Re-poll the existing queue after an observation timeout; never launch a copy.

After the population run, independently verify all32rows, then refine only
unresolved sources through the existing adaptive entry with a separately
recorded native witness. Merge the four new numerically qualified cells with
the sixteen preserved reference cells for the paired density/width analysis.


## Adaptive-entry witness for the new checkpoints

The formal population queues remain untouched. Before any unresolved new
sources are refined, a fresh documented pass runs these two commands once,
sequentially, stopping on failure. They use the completed four-source
formal-entry witnesses as verified parents. They are not population results
and must report witness=true with both qualification flags false.

```sh
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python -u /Users/mr.z/ode-workspace-prep/coefficient-20261116/contrast-observations/repo/studies/muon_pd_20261102/density_transport_20261116/adaptive_density.py --root /Users/mr.z/ode-workspace-prep/coefficient-20261116/contrast-observations/input/density-models --data /Users/mr.z/ode-workspace-prep/coefficient-20261116/contrast-observations/density-data --case-output /Users/mr.z/ode-workspace-prep/coefficient-20261116/contrast-observations/doc-witness/formal-entry-0 --parent-commit fd7f09b0064ceab36a13f7757fd2716d002cbba6 --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/contrast-observations/adaptive-doc-ring8 --device cpu --derivative-backend coordinate --witness

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python -u /Users/mr.z/ode-workspace-prep/coefficient-20261116/contrast-observations/repo/studies/muon_pd_20261102/density_transport_20261116/adaptive_density.py --root /Users/mr.z/ode-workspace-prep/coefficient-20261116/contrast-observations/input/density-models --data /Users/mr.z/ode-workspace-prep/coefficient-20261116/contrast-observations/density-data --case-output /Users/mr.z/ode-workspace-prep/coefficient-20261116/contrast-observations/doc-witness/formal-entry-1 --parent-commit fd7f09b0064ceab36a13f7757fd2716d002cbba6 --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/contrast-observations/adaptive-doc-product16 --device cpu --derivative-backend coordinate --witness
```

Require both exits0, four native observer rows and two four-source merged
cells per model. Preserve the unchanged gates, observer and source snapshot.
The parent verifies source and model dependencies at fd7f09b; no running
engine is edited. After independent NumPy reconstruction, a completed formal
case may use the same adaptive entry with all1,024sources and its actual
completed parent. Begin only on a successful terminal parent worker;
a partial progress file is insufficient.
