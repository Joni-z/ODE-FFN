# Learned-field transport of the crossed-solver effect

Registered17November2026 before new learned-field tangent/intermediate-lambda
observations. Parent derivation: target-aware-bias-transport.md. This tests the
missing transport link, without training or proposing an image architecture.

## Fixed cohort and inputs

Keep all four20k seed0 checkpoints in cross-solver-cases.json, both tasks and
both splits. Product16 RK4-trained weights are the primary case because the
earlier crossed-solver benefit was resolved on both splits. The other three
cases remain controls. Copy the existing make_sharing_assets.py generator,
changing only import resolution and streams: selection2026112700,
confirmation2026112800. These stream IDs were absent from prior studies.
Bind hashes before learned outcomes. Eight4096-source/target banks and256
directions per task/split, plus4096 field inputs, are unchanged in size.
New streams are evaluation draws; all model training remains seed0.

## Native computation and intervention

Use the unchanged reference/synthetic/train.py::generated_samples, Heun128.
v0 is the native trained hidden solver;v1 substitutes only the other hidden
solver at the same weights/eight P applications. v_lambda=v0+lambda*(v1-v0).
Register lambda=.125,.250,.500,1.000. No fitted coefficient or selected layer,
source, checkpoint, projection or time window. Replay original Heun32 final
test examples before any substitution. Retain both native FP32 endpoint sets.

For derivatives, cast restored FP32 tensors AND actual runtime buffers to FP64.
Use forward-mode JVP of v0 in an augmented field (x,u), with forcing v1-v0;
pass this through the SAME native Heun algorithm. Use a direct primal forward
for x, and separately check its agreement with the JVP primal. Preserve FP64
native/crossed/intermediate endpoints, tangent and eight accepted-time snapshots.
No assertion that FP32 rounding is differentiable or the outer ODE exact.

## Numerical qualification (fixed before observations)

1. Independent linear Heun recurrence, affine forcing and rotation controls:
   FP64 component error <=1e-11*(1+abs(reference)).
2. SW2-squared directional derivative: independent Torch-autograd control,
   translation, zero-distance and tied-projection right-derivative controls.
   Lexicographically sort by baseline projection then tangent within exact ties.
3. Learned JVP: first64 fixed field inputs, deterministic direction, central
   finite difference epsilon1e-5; component gate2e-6+2e-5*abs(reference).
4. Augmented baseline replay against ordinary FP64 native Heun is exact on
   every source. JVP primal/direct field agreement <=1e-12*(1+abs(primal)).
5. Whole-sampler tangent: first64 sources in EVERY bank, central lambda
   finite difference epsilon1e-4, gate2e-5+1e-3*abs(reference). All4096 sources
   still enter all quality/tangent outcomes; distinguish these derivative probes.
6. Target functional derivative: all generated endpoints, perturb X by
   epsilon*u at epsilon1e-4/1e-5, right finite difference (central where no
   baseline ties); gate1e-6+1e-3*abs(analytic derivative). Retain actual residuals.
7. Quantify FP32/FP64 endpoint RMS/max, SW2 changes and change-of-change on every
   bank. No endpoint is dropped for precision. Transfer of a direction to native
   FP32 requires the FP64 target-effect sign to survive the measured precision
   change; otherwise label the native interpretation unresolved.

Numerical failure preserves the entire case; it cannot be removed from pooled
reporting or repaired by relaxing a gate after opening learned outcomes.

## Predictions and decisions

Primary functional L is the SAME empirical squared SW2 with fixed targets and
directions. Compare actual L(lambda)-L(0) to lambda*L'(0), and actual endpoint
change to lambda*u. Report all four lambdas, banks and pooled endpoints.
Small-perturbation calibration and lambda1 explanation are separate. Report
absolute residual and improvement over the zero-change prediction, defined
as1-(predicted_change-actual_change)^2/actual_change^2 (undefined at zero).
Use no fitted scale. A full-scale explanation must predict the correct sign
and beat zero-change on both splits of the primary case; retain other cases.
Paired-bank intervals concern evaluation variation, never training variability.
For the eight full-scale case/split contrasts, enumerate all6435 multinomial
bootstrap count vectors of eight paired banks (8^8 ordered resamples), report
95% percentile intervals and nominal eight-contrast-adjusted intervals with
tail probability.05/16. This is conditional empirical-bank uncertainty, not an
exact population coverage guarantee or a training-seed test. Pooled32768-point
scores are separate from bank-mean differences. Intermediate lambdas are
calibration curves, not additional searched hypothesis tests.
Do not transform a negative linearly predicted L into a valid SW2 by clipping.

Only after numerical/target-transport qualification, replace measured delta_v
with an independently derived hidden-solver defect, including BOTH midpoint
and RK4 defects. A successful measured-delta tangent is not itself ODE-specific.
Ordinary control results from older banks remain historical context; no new
architecture superiority is claimed from this frozen-weight mechanism test.

## Execution boundaries

Warm response-screen-cpu@04f11f75, no package changes, one CPU thread/worker.
Use at most four Mac mini workers after a real-entry timing witness. Raw files:
/Users/mr.z/ode-workspace-prep/coefficient-20261116/target-aware-transport/.
Before formal dispatch, publish/freeze source, document exact invocation and
obtain a fresh agent-follows-doc pass. Keep Reports1/2 and image source unchanged.
145 physical0-3 prohibited;4-7 only if no compute PID and<500MiB, no preemption.
