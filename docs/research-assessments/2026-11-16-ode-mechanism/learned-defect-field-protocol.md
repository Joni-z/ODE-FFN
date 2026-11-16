# Can an ODE-derived correction predict the learned network's field change?

Registered before applying the analytic observer to learned checkpoints.
The ongoing population-resolution study retains its own decision and all
observations. This field experiment does not replace that sampling test.

## Question

Changing an internal solver changes the complete network's velocity output.
The measured change can predict distributional response, but that calculation
already knows the changed field. Can the independently derived ODE correction
predict this field change without observing or fitting it first?

For f(y)=Pq(y), use BOTH analytic defects in rk4-defect-derivation.md:

    G(y) = (1/n_cross)^p_cross g_cross(y)
           - (1/n_native)^p_native g_native(y),
    (n_midpoint,p_midpoint)=(4,2), (n_RK4,p_RK4)=(2,4).

The existing observer integrates y'=f and z'=Df(y)z+G(y), passing the incoming
variation through all six internal flows and the original intervening network
operations. Its output b(x,t) is the unfitted first variation about the common
continuous hidden reference. It is not a derivative about a coarse native map.
Compare b with delta_v=v_cross-v_native at the actual trained coarse budget.

## Fixed identity

Four original20kseed0weights, both existing target-transport field streams,
all4096original inputs/time/oracle values per stream. Export the actual
Mac-restored tensors AND nonpersistent time buffer exactly as for the completed
portable-resolution readiness. Copy existing field arrays by hash; no new draw,
training, response design, fitted scale, checkpoint or time-window selection.

All formal cases retain every input. The first16inputs are a prebound small
CLIwitness; the first8within it are the independent derivative controls.
The four models/two streams remain separate empirical units.

## Numerical checks fixed before learned observations

1. Hash every exported tensor and field array; unchanged tensors, solve methods
   and CPU/CUDARNGafter each case. Original native output replays exactly after
   observer patches are restored.
2. Compute complete primal and tangent at referenceRK4steps16/32/64. At every
   input,32->64component change must be<=1e-7*(1+abs(primal64)) and
   <=1e-8+1e-5*abs(tangent64). Retain both differences and all failed points.
   Aggregate refinement differences should decrease unless both are<=1e-12.
3. On the first16inputs, reuse complete_forward's original full-network
   two-tolerance DOP853reference (rtol/atol1e-10/1e-11and1e-12/1e-13).
   Loose/tight output difference<=1e-9*(1+abs(tight)); observer primal64must
   agree within1e-7*(1+abs(tight)). No native-state entrance substitution.
4. On the first8inputs, at reference16, compare with the separate forced_model
   whole-network epsilon-JVP: component error<=1e-11*(1+abs(reference)).
   Central differences at BOTH1e-4and1e-5use2e-7+2e-4*abs(FD). Include zero
   forcing and reversed solver order; output variation must be zero/negated
   within1e-12. Do not alter the readout of a learned checkpoint.
5. CPU/CUDAportable witness must agree on actual primal/tangent at16within
   1e-9+1e-8*abs(CPUreference). This is numerical identity, not prediction fit.

Numerical failure makes the affected prediction unresolved. Preserve the
original thresholds and all arrays; do not drop points or silently add a finer
reference. A separately registered refinement can follow a real failure.

## Prediction and falsification

Primary models are Product16RK4weights on BOTH field streams. Use all4096
points and all coordinates. Define

    score = 1 - mean(||b-delta_v||^2) / mean(||delta_v||^2).

The leading analytic account passes its field criterion only if all numerical
gates qualify and score>0on both primary streams. This compares with predicting
zero change, with no regression or fitted amplitude. A zero signal yields an
undefined score, never a pass. Report relative RMSerror, dot-product alignment,
pointwise norms, time breakdown and all eight case results regardless of outcome.

Secondary: compare actual oracle-risk change with the fixed prediction
2*mean(<v_native-v_oracle,b>)+mean(||b||^2). Keep its signed error separate
from vector accuracy. Report deterministic full-cohort quantities; no claim of
training-replicate inference or ordinary-FFN superiority.

If the field criterion passes AND the separate sampling benefit survives its
outer-mesh test, propagate this analytic forcing through the actual sampler
and target functional. If field prediction fails, distinguish an unresolved
reference from a failed coarse-step asymptotic approximation before another
architecture. This experiment alone cannot establish a sampling gain.

## Execution

Reuse hidden_defect_transport.py unchanged and the original full-model builder,
sampler-independent field inputs and attributed torchdiffeqRK4kernel. Prepare
the small learned CLIwitness on warm Mac CPU04f11f75. Publish/freeze and obtain
the required independent documented execution before formal diagnostics.
Use warm145CUDA41450454for full batches only after the current population
workers release their contexts; fresh empty physical4-7only, never0-3or another
user's task. No package changes. Final reports/native image files stay fixed.
