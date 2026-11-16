# Next missing link: transport hidden numerical bias to the target metric

The completed experiments isolate three facts. The hidden midpoint bias is
quantitatively predictable. Removing it can increase complete-field error.
Its field-risk direction does not reliably rank generated distributions.
Product16 RK4-trained weights improve under midpoint inference even though
that substitution slightly worsens field MSE. The unresolved question is
which part of the ODE-derived change causes that target-quality improvement.

This is a mathematical follow-up to the observed effect, not another response,
order or architecture search. No coefficients may be fitted to the known SW2.
The earlier oracle-path first-order endpoint account failed at full model error;
do not reuse it as a general explanation. Here the reference is the actual
learned baseline, the perturbation is the controlled SAME-WEIGHT solver change,
and the final functional is the target sliced distance rather than paired
source/oracle endpoint error.

## 1. Exact tangent of the existing discrete sampler

For each retained model let v0 be its native field, v1 the crossed-solver field,
and delta_v=v1-v0. Define v_lambda=v0+lambda*delta_v. Under the unchanged Heun
sampler with step h, store its actual baseline states x and predictor p.
Differentiate the discrete algorithm at lambda=0, keeping time fixed:

    a = v0(x,t)
    da = J_x v0(x,t) u + delta_v(x,t)
    p = x + h*a
    dp = u + h*da
    db = J_x v0(p,t+h) dp + delta_v(p,t+h)
    x_next = x + h/2 * (a + v0(p,t+h))
    u_next = u + h/2 * (da + db),  u_initial = 0

This is the tangent of the real-valued Heun computational graph, not an
assertion of exact continuous outer integration. Kernel and finite-difference
checks must measure floating-point resolution. JVPs avoid forming complete
Jacobians and keep the existing model/sampler algebra. Preserve native endpoint
replay before interpreting any tangent prediction.

## 2. Differentiate the actual distributional functional

For fixed common target samples Y and unit directions theta, let pi_r order the
baseline projected generated endpoints. Here N counts generated samples and D counts projection directions. For squared
sliced distance L:

    residual_ir = <X[pi_r(i)], theta_r> - sort(<Y,theta_r>)[i]
    L'(0) = 2/(N*D) sum_ir residual_ir * <u[pi_r(i)],theta_r>
    SW2'(0) = L'(0)/(2*sqrt(L(0))) when L(0)>0

Unlike paired endpoint error, this is the directional derivative of the SAME
empirical target metric used in evaluation. Record projection ties. If ties
occur, use the right directional derivative: order tangent projections within
tied endpoint groups for lambda approaching0 from above. Do not pick a favorable
permutation. Keep squared-distance and square-root-distance predictions distinct.

## 3. Separate calibration, explanation and confirmation

First use the recorded cases to calibrate exact JVPs and native replay; known
lambda=1 outcomes are retrospective evidence, not a new prospective result.
Prepare fresh common evaluation banks, hashes and native-target controls BEFORE
new tangent/intermediate-lambda observations. Training remains seed 0; new
held-out evaluation streams are not additional model training runs.

Primary case study: Product16 RK4-trained weights, where midpoint inference has
a resolved sampling benefit on both old splits. Retain the other three frozen
weight/task cases as controls. Register lambda=.125,.250,.500,1.000, sources,
Heun resolution and numerical gates before dispatch. Measure actual interpolated
samples alongside the unfitted lambda*u endpoint and lambda*L'(0) metric changes.
Small-lambda calibration and lambda=1 explanatory success are separate requirements;
do not extrapolate a fine-perturbation success when the native-scale prediction
fails. Keep every source and failed gate. Conditional paired evaluation intervals
remain separate from numerical resolution. No time-window or layer selection.

## 4. The ODE-specific part must also be tested

A successful transport calculation for measured delta_v would not yet explain
why the ODE block produced it. Only after that stage qualifies, replace measured
delta_v with the independently derived hidden-solver bias and test again:

    midpoint modified field: f + eta²[-H[f,f]/24 - J(Jf)/6] + higher terms
    RK4 modified field: f + eta^4*g_RK4 + higher terms

Derive g_RK4 from the fifth-degree local Taylor defect of the actual copied
classical method. For f=P ISRU, coordinatewise derivative jets and shared P
permit a direct algebraic construction; verify linear and scalar controls before
learned observations. Propagate both hidden defects through the complete readout,
then through the learned-field sampler tangent and target functional above.
Do not silently drop the RK4 defect or assert it is negligible at two steps.

The intended evidence chain is: prescribed hidden computation -> predicted
hidden bias -> predicted full-field change -> predicted target-metric effect.
Each arrow needs its own numerical check and unfitted held-out comparison.
If this chain fails, keep the measured discretization effect and reject that
mechanistic explanation rather than add another activation. No image candidate
is promoted while ordinary controls remain stronger.

## 5. Keep endpoint transport and target-functional curvature separate

Added during full execution, from the following analytic control; the original
primary linear-L test and its gates remain unchanged. This is a secondary error
accounting of the already fixed endpoints, not another fitted predictor.

Even an exact endpoint tangent need not give an accurate first-order loss change
at lambda1. Let X(lambda)=X0+lambda*u exactly. The squared distance is generally
nonlinear in lambda; for a pure translation toward an identical target cloud,
L(lambda)=(1-lambda)^2*L0. Its tangent predicts -2*lambda*L0, while the actual
change is (-2*lambda+lambda^2)*L0. At lambda1 the tangent has the correct sign
but only ties the zero-change predictor in squared error, although endpoint
transport is exact. Thus failure of the registered linear-L test alone does
not identify failure of the endpoint tangent.

For every fixed case/split/lambda, the following exact decomposition is available
without new model evaluation:

    Lhat(lambda)=SW2_squared(X0+lambda*u,Y)
    Lactual-L0-lambda*Lprime = (Lactual-Lhat)
                             +(Lhat-L0-lambda*Lprime)

The first term is target-functional error from endpoint linearization. The
second is the exact functional remainder along the linearized endpoints,
including any projection sorting changes. Do not fit either term or relabel the
original primary test as passed. Report all four cases and both splits if this
secondary decomposition is used.

An analytic FM counterexample also separates field fitting from generation.
For independent standard-Gaussian source/target, the interpolation variance is
c(t)=(1-t)^2+t^2 and the oracle velocity is a(t)*x, a=(2t-1)/c. Consider

    vA=a(t)*x+.1
    vB=a(t)*x+.2*(2t-1)

Their uniform-time field MSEs are .010 and .013333..., respectively. The final
variance is1 for both. Their means are mA=.1*sqrt(2)*asinh(1) and mB=0, since
the endpoint forcing weight1/sqrt(c(t)) is symmetric and B's forcing odd around
t=.5. Population W2 is |mA| for A and0 for B. Larger field MSE can therefore
give exact distribution matching through temporal cancellation. This is a
general flow property, not an ODE-FFN advantage. A native Heun128 deterministic
check must agree with these endpoints/tangents to1e-4 before using the example.

## 6. Post-completion diagnosis of individual and distributional responses

The full cohort has now completed. The pooled functional prediction has the
correct sign in all eight cases and beats zero-change prediction in each.
The original primary decision is false because of its numerical qualification;
the four failed bank checks remain preserved. Independently, Product16 has
large negative endpoint-prediction fractions, and applying the exact metric to
the linearized endpoints gives very large errors. Thus functional agreement
and faithful finite sample trajectories must not be conflated.

Do not add an endpoint-accuracy requirement to the original functional gate
after observing this result. A distributional first variation can be useful
without accurately predicting each finite-intervention particle displacement.
Conversely, pooled scalar agreement alone does not prove why it occurs. The
exact secondary decomposition shows large signed terms that can cancel.

Before opening source-level sensitivity tails, prescribe this cached diagnosis
on all64banks and8pooled cases: compare norms of u, actual crossed-minus-native
displacements and their residual; record quantiles50/90/99/99.9/100percent,
energy shares of the largest1/10/ceil(.01*N) sources ranked by tangent energy,
and the actual/residual energy at those same indices. Reconstruct every original
endpoint fraction. Record whether the largest tangent lay in the registered
first64-per-bank finite-difference probe, without changing that probe or removing
any source. This is descriptive, post-completion analysis, not a new hypothesis
test or a reason to change the original gate.

For each of the eight case/split combinations, plot the original nine recorded
time snapshots for the single largest endpoint-tangent source, chosen by this
explicit criterion. Compare tangent and actual displacement norms. Preserve
bank/source identities and all raw zeros; logarithmic figures omit only the
zero-valued initial display point. These are extreme-case illustrations, not
typical trajectories. No model call, new sampling, fitted coefficient, selected
projection or selected time window is allowed.
