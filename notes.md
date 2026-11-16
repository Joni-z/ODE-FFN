# Current ODE mechanism evidence

## Matched solver result

Same f=PISRU(y), parameters, initialized tensors/RNG, data stream and eight
P applications. Midpoint4 vs classical RK4 two-step, seed 0/20k. Both complete
training/evaluation pipelines exit 0. Normalized hidden squared error drops by
45.580x (Ring8) and 22.245x (Product16), all 1,920 points qualified. Confirmation
pooled SW2 does not improve on either task; both paired adjusted intervals include
zero. Product16 ordinary SwiGLU remains stronger than both focal ODE models.
All selection/confirmation banks and raw precision are retained.

Canonical result: docs/research-assessments/2026-11-16-ode-mechanism/rk4-results/.
Audit: same parent/rk4-audit.json. Raw: /Users/mr.z/ode-workspace-prep/coefficient-20261116/rk4-control/.
Metrics reconstructed independently in NumPy (2.78e-17 max), separate exact
conditional-bank statistics (2.71e-20 max). Parameters/MACs match; 100-repeat
same-process forward/backward costs are descriptive. Final Reports1/2 unchanged.

## What the result changes

The established midpoint modified-equation prediction explains numerical bias,
but reducing that bias during training does not produce a useful generation
gain here. Continuous accuracy is not a sufficient architecture rationale.
The complete-dimension null also rejects a special tangential-output explanation.
Do not reopen activation search or promote a new image FFN from either result.

## Next registered diagnosis

rk4-followup-protocol.md fixes four learned weights, full 4,096 selection-field
inputs and every hidden solve. Cross solver/weight combinations to test the
strong diagonal compensation prediction; compute exact complete-field risk
change 2<e,delta>+||delta||² and its numerical qualification. Distinguish small
output exposure from learned compensation before another training architecture.
This follow-up is registered, not yet implemented or observed.
