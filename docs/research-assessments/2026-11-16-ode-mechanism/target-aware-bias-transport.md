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
