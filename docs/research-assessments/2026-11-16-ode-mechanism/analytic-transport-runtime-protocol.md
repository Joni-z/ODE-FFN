# Accelerate the same analytic observer before full trajectory transport

The complete learned-field test now passes on all eight4096-input cells.
The remaining causal link propagates that analytic forcing through the actual
learned sampler and target functional. Keep the256-step FP64observer, both
solver defects, actual tensors/buffers and original sampling banks.

## Measured cost and reason for this preparation

The completed GPUhistory records a median7.232sfor one256-step observer call
on4096inputs, including the existing stage bookkeeping. At Heun2048, two
streams of eight banks would require65,536such calls for the primary weight
set, roughly132GPUhours at that rate. This is an extrapolation, not a measured
trajectory runtime. Do not start that long cohort before measuring an exact
execution improvement.

## Fixed optimization object

Factor one coupled RK4step out of variation_solve and compile ONLY that tensor
step with the already installed torch.compile. Reuse the original forcing
function and unchanged attributed rk4_step_func. Keep the original complete
network forward, all six local solves,256steps, FP64, learned P, ISRUresponse
and BOTHdefect prefactors. The canonical eager observer remains untouched.
No lower reference, fewer banks, surrogate response or fitted coefficient.

First use the default compilation mode with fullgraph=True,dynamic=False.
Record a failure rather than silently falling back to a different numerical
method or installation. Compiler caches belong in a new task-specific external
cache directory. Reuse the warm41450454environment; inspect local compiler
availability before attempting a build, with no package changes.

## Identity and numerical gates before speed claims

Use every actual field input from all four frozen models/two streams. Compare
compiled complete primal/tangent at256with the saved eager256arrays.
Component tolerances: primal1e-10+1e-9*abs(eager), tangent1e-10+1e-8*abs(eager).
Require all points, unchanged native output after patch removal and unchanged
tensors/methods/RNG. Compiler warmup may be wrapped in a forked RNGcontext;
it must not consume experimental data draws or alter model state.

Retain the independent whole-network JVP/FDand zero/sign controls on the same
prebound first8inputs, using the existing original formulas. Validate both
native/crossed solver orders. A scalar/vector comparison must not be replaced
by similar final quality scores. Preserve all failed flags and source hashes.

## Timing and next decision

After fixed warmup, use three paired device-resident repetitions of eager and
compiled256-step observers on ProductRK4confirmation's same4096inputs.
Synchronize around every timing. Report all times, medians, compilation time,
peak memory and break-even call count separately. No timing-only run is a
quality result. Keep small-prefix and full-batch timings separate.

Only an independently documented numerical witness plus a measured speed gain
can select compiled execution for the next target experiment. If it fails
identity or is slower, preserve the result and profile the same equations;
do not quietly simplify the scientific test to make it cheaper or easier.
Only empty physical145GPUs4-7may be used;0-3and foreign tasks remain prohibited.
