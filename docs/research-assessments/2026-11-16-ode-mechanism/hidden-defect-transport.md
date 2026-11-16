# Propagate the prescribed ODE defect through the complete existing model

This is deterministic preparation for the conditional next mechanism test.
No learned checkpoint, data bank, training or generation is used. The original
target-transport study and its gates remain unchanged. The measured-delta
calculation must be assessed before applying this observer to learned weights.

## Quantity predicted

For every local flow f(y)=P ISRU(y), both numerical methods approximate a
modified field. With their actual steps etaM=1/4 and etaR=1/2, the leading
crossed-minus-native drift is

    G(y) = eta_cross^p_cross * g_cross(y)
           - eta_native^p_native * g_native(y)
    p_midpoint=2, p_RK4=4.

Use BOTH defects from rk4-defect-derivation.md. The prefactors happen to equal
1/16; this does not make the two defect functions equal or allow dropping RK4.
Do not infer the sign from a model's performance.

Let the entire existing model with continuous local drifts f+epsilon*G be
F_epsilon. The prediction is dF_epsilon/depsilon at0, without a fitted scale.
It is the leading crossed-minus-native discretization effect about the common
continuous hidden reference, not a tangent about the native coarse solver.
This distinction must remain explicit when later comparing with actual fields.

## Local variation and whole-model propagation

For a layer entrance with primal h and incoming variation z0, integrate

    y' = f(y),             y(0)=h
    z' = P[q'(y)*z]+G(y), z(0)=z0.

Here q'(y)=(1+1.2*y^2)^(-3/2). The incoming variation is generally nonzero:
every previous flow, SiLU, normalization and residual has already contributed.
Returning only the local forced increment would miss that chain.

Use forward-mode dual tensors to pass (y,z) through the ORIGINAL complete model.
Patch only each local solve method while the observer executes, then restore
it. Ordinary LayerNorm, time conditioning, residuals, middle SiLU and final
readout are not rewritten. Internal solves use the already copied/attributed
classical RK4 kernel at16/32/64steps; these are diagnostic reference refinements,
not replacement training settings. No dense Jacobian/Hessian is formed.

An independent construction executes the same original model with ordinary
local drifts f+epsilon*G. Differentiate this whole computation with torch.func.jvp
and central finite differences. At a fixed reference resolution, RK4 applied to
the coupled variational equation equals the derivative of the RK4 algorithm
in real arithmetic. This checks all six local solves and every intervening
module without relying on a hand-copied network forward.

## Deterministic controls, registered before observations

Two untrained fixtures: input dimensions2/16, native width64/depth3, seed0,
eight fixed cosine inputs and times linearly spaced.1to.9. The standard builder
initializes the head to zero, which would make a derivative test vacuous.
For these fixtures only, prescribe output.weight=.02*sin(arange(numel));
retain the zero output bias. Do not modify any actual checkpoint.

1. Torch defect formulas vs the independent NumPy polynomial-jet construction:
   max absolute error<=1e-12 for midpoint and RK4 on fixed input states.
2. Coupled observer vs independent whole-model epsilon JVP at16steps:
   primal/tangent component error<=1e-11*(1+abs(reference)).
3. Central finite differences at epsilon1e-4/1e-5,16steps:
   tangent error<=2e-7+2e-4*abs(reference). Retain both resolutions.
4. Zero forcing yields exactly zero output variation. Reversing solver order
   negates the variation within1e-12 and preserves the primal. Require the
   original nonzero forcing's output norm>1e-10 to exclude a vacuous head.
5. Record16/32/64step primal/tangent differences. Require final32->64changes
   <=1e-7primal and1e-8tangent, and decrease from16->32unless both are<=1e-12.
6. Snapshot every parameter, actual buffer and solve method before/after. They
   must remain unchanged. Native model output must replay exactly afterward.

These checks establish an implementation, not that a leading expansion is
accurate at the trained coarse step, that it predicts SW2, or that an ODE model
outperforms an ordinary FFN. Such comparisons require their own full-cohort
learned tests and existing strong controls.

## Execution

Warm response-screen-cpu@04f11f75; CPU, one thread, no package changes or GPU.
Raw preparation outputs belong under the current target-aware-transport root
in hidden-defect-preparation/. Publish/freeze the implementation and obtain a
fresh exact documented execution before declaring it ready. The original
native queue and all cached postprocessors retain their output ownership.

The fresh witness works from the frozen archive at
/Users/mr.z/ode-workspace-prep/coefficient-20261116/target-aware-transport/hidden-defect-preparation/repo/.
Its source-freeze.json in the parent folder binds every regular source file
and the source commit. Check all hashes before/after; execute this command
verbatim, once, without repairs or package installation:

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python studies/muon_pd_20261102/target_transport_20261117/check_hidden_defect_transport.py --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/target-aware-transport/hidden-defect-preparation/controls-readiness.json
```

Record actual terminal exit, any invocation divergence and source hashes in
hidden-defect-preparation/readiness-receipt.json. This command has no data-root
or checkpoint argument and may not launch the learned observer or any queue.

## Author controls

Actual execution13835 exits0. Both2D/16Dfixtures pass all fixed controls:
polynomial-jet error<=2.61e-17, whole-model JVP error<=8.14e-20, central-FD
error<=8.95e-12. Final32->64reference changes are<=4.43e-11primal and
6.79e-13tangent. Nonzero output variations have norms1.60e-4/8.80e-4;
zero/sign tests are exact and the original model replays exactly afterward.
Complete values: hidden-defect-controls.json. These are implementation controls;
no learned-state prediction or generation finding is included.
