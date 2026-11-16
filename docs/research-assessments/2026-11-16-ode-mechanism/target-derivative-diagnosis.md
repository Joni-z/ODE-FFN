# Diagnose finite-difference failures without changing the transport gate

Two completed Product16 banks fail the registered target-functional derivative
check at epsilon=1e-4, while their original endpoint-tangent and primal replay
checks pass. Both pass the target check at epsilon=1e-5. This document records
the diagnostic before inspecting rank-change calculations or smaller-epsilon
results. The running a6f4d42 source, original thresholds, saved arrays and primary
decision remain unchanged. This is a post-observation numerical diagnosis, not
a replacement preregistration or a new model result.

## Question and exact identity

Does the fixed finite perturbation cross projection-order boundaries, even
though the infinitesimal empirical SW2-squared derivative is correct?

In each projection let a and b be baseline endpoints and tangent values ordered
by a, and let c be sorted target projections. With the average over every point
and projection, define

    L0 = mean((a-c)^2)
    D = 2*mean((a-c)*b)
    Q = mean(b^2)
    Lfixed(e) = L0 + e*D + e^2*Q
    C(e) = mean((sort(a+e*b)-c)^2) - Lfixed(e).

Without baseline ties, D is the ordinary directional derivative. Sorting changes
the target assignment. Rearrangement gives C(e)<=0, and the central difference
satisfies the exact real-arithmetic identity

    FD(e)-D = [C(e)-C(-e)]/(2*e).

Evaluate C by the difference-of-squares product, avoiding subtraction of nearly
equal losses. Separately record floating-point projection/reduction residuals.
The nearest adjacent crossing radius is min((a[i+1]-a[i])/abs(b[i+1]-b[i]))
over unequal adjacent tangent values, across all projections. It establishes
a sufficient symmetric interval where the original ordering is unchanged.
Exact baseline ties require a one-sided treatment and are not certified by
this no-tie diagnosis; reject them explicitly rather than silently picking a
permutation.

## Fixed diagnostic comparisons

1. Independently compute D with Torch autograd through the actual sorted target
   metric. Absolute discrepancy <=1e-11*(1+abs(D)). No model invocation.
2. Compare the fixed-order quadratic and the directly summed fixed-order loss:
   residual <=1e-12*(1+L0) at every listed epsilon.
3. Reconstruct the finite-difference error with the two assignment corrections
   and separately measured floating-point projection, fixed-quadratic central
   reduction and correction-subtraction effects: residual <=1e-9*(1+abs(D)).
   Retain the unadjusted residual and every rounding term, not just pass flags.
4. Record changed-rank counts at epsilon=1e-4,1e-5,1e-6,1e-7,1e-8. These are a
   complete secondary resolution scan, not a search for a replacement gate.
5. Reproduce every stored original loss/derivative/check and flag. Require exact
   asset/output hashes and retain each original qualification value.

Deterministic controls first: fixed ordering, an explicit pair crossing, tangent
sign reversal, permutation invariance and rejection of exact baseline ties.
The two initial failed banks are a debugging subset, not a full-cohort result.
After the original queue ends, apply the identical diagnostic to all64banks and
all8pooled cases. Do not select sources, projections or banks by their outcome.
Do not assert a derivative implementation bug, or its absence, before these
independent checks. Even a correct infinitesimal derivative does not establish
that a first-order prediction explains the full-scale generation change.

## Execution boundary

Warm response-screen-cpu@04f11f75, one CPU thread, no package changes, no GPU,
no training or new sampling. Read cached endpoints/tangents and bound targets
only. New diagnostic outputs go under the raw target-aware-transport root in
metric-rank-diagnosis/, separate from the existing collector/postprocessor's
owned outputs. Preserve finalized reports and native image code.

The standalone source is frozen under metric-rank-diagnosis/source/; its
source-freeze.json binds the three regular source/document/spec files. Work in
/Users/mr.z/ode-workspace-prep/coefficient-20261116/target-aware-transport/.
A fresh agent must execute these three commands verbatim, sequentially, and
record actual terminal exits and hashes before/after. Do not overwrite existing
outputs or edit source to obtain a pass.

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python metric-rank-diagnosis/source/diagnose_metric_rank.py --controls --output metric-rank-diagnosis/readiness/controls.json
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python metric-rank-diagnosis/source/diagnose_metric_rank.py --input formal/confirmation/product_rk4_trained --bank 0 --output metric-rank-diagnosis/readiness/confirmation-product-rk4-bank0.json
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python metric-rank-diagnosis/source/diagnose_metric_rank.py --input formal/selection/product_midpoint_trained --bank 1 --output metric-rank-diagnosis/readiness/selection-product-midpoint-bank1.json
```

After complete native execution, the full cached command is:

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python metric-rank-diagnosis/source/diagnose_metric_rank.py --formal-root formal --output metric-rank-diagnosis/full.json
```

This command requires all eight original completion bindings and checks all
64banks plus8pooled cases. Do not launch it before native completion or duplicate
its output owner. No changes to the existing three-command postprocessor.

The first deterministic execution84537 exited1 at the identity check on the
fixed-order fixture, before any learned-bank diagnosis. At epsilon1e-8, central
subtraction of two finite-precision losses also contributes rounding. The
implementation now records that contribution and the correction-subtraction
rounding separately, as well as the originally planned projection rounding.
The identity threshold and every original transport gate remain unchanged.

## Debugging-subset observations

Author control35007 and actual cached executions43228/87183 exit0. The
infinitesimal derivatives agree with independent Torch autograd within2.17e-19.

| Case/bank | Original epsilon1e-4 error | Predicted sorting effect | Changed rank slots (+/-) | Unadjusted identity residual |
|---|---:|---:|---:|---:|
| Product RK4 confirmation0 | 2.672e-6 | -2.672e-6 | 3984/4103 | 1.927e-14 |
| Product midpoint selection1 | 1.957e-6 | -1.957e-6 | 2675/2667 | 1.718e-14 |

Rank slots count sample/projection positions, not unique samples. The first
adjacent crossing radii are3.399e-9 and2.504e-8, respectively. Thus the original
fixed perturbations are not inside a single assignment region. The sorting
correction explains the original discrepancy; there is no derivative-code
failure in these two banks. This does not certify unobserved banks, change the
original numerical qualification or establish full-scale target prediction.
The full five-epsilon scan and original checks are retained in
metric-rank-debugging.json. Full-cohort diagnosis remains pending.
