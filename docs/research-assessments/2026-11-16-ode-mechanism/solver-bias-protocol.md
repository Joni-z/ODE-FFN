# Parameter-free prediction of the trained hidden solver error

Registered before observing learned-checkpoint bias predictions. The derivation
and deterministic linear/quadratic/ISRU controls are in solver-bias-derivation.md
and solver-bias-math-check.json. They validate the formula, not the learned data.
This study concerns state-updated midpoint models, not the image Taylor K8 layer.

## Fixed data and prediction

Use the same six shared/PAL/ALT checkpoints, six local layers and all80 original
inputs per checkpoint. Read the exact captured entrance, four stage matrices,
qualified continuous references and native endpoints. Reuse4/8/16/32 from the
energy study and64/128/256 from the native-reference study, with hashes checked.
Those endpoints remain valid observations even where a prior downstream-output
qualification failed. Do not select by generation score or exclude a layer.

Integrate the continuous state and two variational contributions across each
exact stage boundary. For q=ISRU, Jv=P(q' elementwise v), and

```text
g_H = -P(q'' elementwise f elementwise f)/24
g_J = -P(q' elementwise P(q' elementwise f))/6
z_H' = J z_H + g_H,   z_J' = J z_J + g_J,   both initially zero.
predicted error at N steps = (z_H(1)+z_J(1))/N^2.
```

No learned scale, regression, selected subspace or extra parameter is allowed.
The two components may cancel and are retained separately. Compare with
native endpoint minus the existing continuous reference at every resolution.

## Numerical gates

Use the existing DOP853 tolerance pairs1e-10/1e-11 and1e-12/1e-13. At every
captured state require both state references and their agreement with the
existing tight endpoint within1e-9 absolute/relative, componentwise. Require
each variational contribution to agree between tolerances within1e-8 absolute/
relative. Keep every unresolved state; a numerical failure is not support.

Independently estimate the derivative of the final flow under f+epsilon*g
by central perturbations at epsilon1e-4 and1e-5, using state-only DOP853at
rtol1e-12/atol1e-13 and the same stage boundaries. At both perturbation scales,
require componentwise agreement with z_H+z_J within2e-6 absolute/2e-5 relative.
This checks the variational calculation by a different computational route.
Native endpoint/source-array bindings and all available resolution arrays
must be independently checked, with no new native-model training or sampling.

## Primary hypotheses and outcomes

The asymptotic prediction requires pooled relative residuals to decrease from
N64 to128 to256 in each of the six models. Every point/layer/level stays in
the pooled energies; also retain layer and bank breakdowns.

The trained-scale hypothesis is separate: at N4, the parameter-free prediction
must beat a zero prediction for both complete error and potential-normal
error in each fixed model. Define each prediction fraction as

```text
B = 1 - sum||actual_error-predicted_error||^2 / sum||actual_error||^2.
```

Project both actual and predicted differences onto q at the qualified tight
reference for the normal comparison. B>0 is better than zero; no arbitrary
performance margin is introduced. Any nonpositive point fraction rejects the
corresponding six-case trained-scale claim. Passing fine resolutions alone is
not evidence for the actual trained4-step computation.

Eight input banks are conditional uncertainty units. Jointly bootstrap the
actual and residual energies with exact8^8 enumeration. Report ordinary95%
and nominal12-contrast-adjusted intervals for N4 total/normal B over six
models. Intervals crossing zero do not resolve the trained-scale claim.
Zero denominators remain unresolved. Report normal enrichment predicted by z,
actual normal enrichment at every N, and the two signed source contributions;
do not select one source after seeing the outcome.

This is an explanation of hidden numerical error, not a generation metric.
It neither proves an ODE-specific benefit over SwiGLU nor promotes a new image
candidate. A direct output/generation intervention requires its own subsequent
test if the trained-scale prediction qualifies and is supported.

## Execution

Two one-thread CPU workers, one dataset each; existing NumPy/SciPy environment,
no packages or GPU. A fresh exact documented first-layer witness retains all
three models/all80 inputs and the independent perturbation checks. Full runs
retain all six layers. Save every numerical failure and actual terminal exit.
Preserve final Reports1/2, original source freezes and the image trainer.
