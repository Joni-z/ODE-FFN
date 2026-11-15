# Density transport: registered numerical qualification

## Question

Does the remaining shared-generator sampling advantage correspond to a
smaller density error along the generated paths? The selected-symmetry
account failed across tasks. Use the exact log-density-ratio identity in
[the derivation](post-symmetry-decision.md), retaining ordinary controls.
The density identity is established continuous-normalizing-flow calculus;
the experiment tests these learned architectures, not a new identity.

## Fixed objects and samples

Use the same eight final 20k checkpoints: shared P, released stage P, Free
ISRU and native SwiGLU on Ring8/Product16. No training, optimizer change,
response change or checkpoint selection. Original and projected fields use
the same groups as P1. Every state remains float64.

Fresh evaluation stream 2026112300 (+1 for Product16), with four banks of 256
independent standard-normal sources and independent native targets per task.
The first eight sources of bank 0 are a fixed numerical pilot, not a model
ranking cohort. No group expansion is applied to those sources. Field
measurements use four independent 64-point banks at each of five prescribed
times 0.05,0.25,0.50,0.75,0.95: 1280 points per task. These inputs, the eight
checkpoint digests and fresh CPU derivative/trajectory references are bound
in density-data.json before the numerical pilot.

## Observer and tests

The pinned native generated_samples function is unchanged. A model-call
observer returns its original velocity and separately calculates the full
input Jacobian with one reverse-autograd pass per output coordinate. Exact
trace in 2/16 dimensions is used. It records:

- oracle field error, divergence error and score-aligned error;
- their signed time integrals on the actual model paths;
- the implied continuous-flow log density ratio;
- its independent oracle-log-density/velocity-divergence reconstruction;
- the log determinants of the actual Heun step Jacobians, as a numerical
  consistency diagnostic, not an assumed globally invertible discrete density.

Require analytic oracle values to match native velocity/log_density,
derivatives to match autograd and decreasing-step central differences,
and the oracle continuity residual below 1e-8. Gaussian rotational/radial
examples have equal velocity MSE but different known density effects.
Their log-ratio and actual linear-step determinants are checked separately.
For all eight learned models, compare input Jacobians with central differences,
require sample separability, and require exact native endpoint identity.

## Pilot and go/no-go rules

For each of the eight cases and both original/projected fields, run native
Heun128/256/512 on the same eight new sources. Save every per-point integral,
endpoint, field observation and exact step-Jacobian determinant.

The 512-step estimate qualifies for larger density measurement only if:

1. No nonpositive step determinant occurs on these observed paths.
2. Maximum absolute source-integral versus oracle-density chain residual
   is at most 0.001 nats per dimension.
3. Maximum absolute continuous-divergence versus actual local-step-logdet
   discrepancy is at most 0.001 nats per dimension.
4. Maximum 256->512 change in either continuous log-ratio estimate is at
   most 0.002 nats per dimension.
5. Maximum relative endpoint change is at most0.001, using Euclidean
   displacement divided by 1 + the 512-step endpoint norm.

These are numerical accuracy tolerances, not performance margins. Preserve
failures as unresolved numerical cases. A subsequent bounded 1024/2048
qualification may be registered for unresolved cases; never silently loosen
tolerances or treat a failed numerical gate as an architecture result.
The field observations can be summarized separately, but source-residual
norms do not replace the intended terminal density object.

Even qualified finite-source averages are Monte Carlo observations, not
analytical KL values. A full comparative measurement retains all ordinary
controls, common source banks and a second resolved outer step size. KL
and SW2 need not rank models identically. A time-local oracle intervention
requires its own stated prediction and confirmation banks after this stage.

## Execution

Warm Mac mini CPU environment: Python 3.12.4, Torch 2.10.0, NumPy 2.2.6,
SciPy 1.18.1; no package changes. Tiny pilot paths use four one-thread CPU
workers, preserving GPU capacity for larger batches. Native source, data,
readiness, PIDs and per-case progress are recorded. On 145, any later CUDA
work must use freshly idle physical 4-7 only; 0-3 are forbidden, and no foreign
task may be stopped or shared. Final reports and image training stay fixed.

## Fresh documented CPU invocation

After the author CPU checks and source publication, a fresh agent runs this
exact command once. It checks the two oracle laws, Gaussian examples,
all eight actual model observers, new CPU references and two real probe entries.

```sh
cd /Users/mr.z/ODE-FFN-upstream && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python -u studies/muon_pd_20261102/density_transport_20261116/check_density.py --root /Users/mr.z/ode-workspace-prep/coefficient-20261116/sharing-symmetry/portable-v2 --data /Users/mr.z/ode-workspace-prep/coefficient-20261116/density-transport/portable --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/density-transport/doc-cpu
```

Only after COMPLETE.json passes with eight models and two actual probe entries,
launch run_density_queue.py with the same root/data, its readiness path,
a new pilot-runs output directory, --workers 4 and --detach. Do not restart an
existing live queue after an observation timeout.

## Registered follow-up after the 512-step pilot

All 48 pilot rows and 16 field cells were independently checked; 6/16 cells
qualify at 512 steps. The other 10 fail ratio refinement, with two also failing
endpoint refinement. All fine-path determinant signs are positive. Do not
report pilot log-ratio averages as architecture rankings.

For only the 10 unresolved original/projected cells, compare 512->1024 under
the identical five gates. If still unresolved, compare 1024->2048. Stop
at 2048 and retain unresolved status if needed. Qualified 512 cells are retained
without extra integration. Use the same eight numerical sources throughout.
No tolerance, source, checkpoint or field changes are allowed.

A fresh agent exercises the actual refinement entry with the bounded witness
flag. The following command is run once, without a retry or code modification:

```sh
cd /Users/mr.z/ODE-FFN-upstream && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python -u studies/muon_pd_20261102/density_transport_20261116/refine_density.py --root /Users/mr.z/ode-workspace-prep/coefficient-20261116/sharing-symmetry/portable-v2 --data /Users/mr.z/ode-workspace-prep/coefficient-20261116/density-transport/portable --pilot /Users/mr.z/ode-workspace-prep/coefficient-20261116/density-transport/pilot-runs --verification /Users/mr.z/ode-workspace-prep/coefficient-20261116/density-transport/verified-pilot --index 1 --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/density-transport/refine-doc-entry --witness
```

The witness is explicitly not a resolution qualification. On success, the
same monitored CPUqueue uses --phase refinement, --pilot and --verification
with a separate refinement-runs output. No formal population KL measurement
is launched by this step.
