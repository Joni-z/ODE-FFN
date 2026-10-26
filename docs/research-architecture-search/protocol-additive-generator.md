# Conditional generator versus positive multiplicative clock

Registered before training. The completed identity-location factorial does
not recover a useful ODE block: every mixed/both-increment arm fails the
baseline-sensitive screen. It also shows why the scalar increment improvement
under the older initialization cannot be treated as a stable structural
benefit. Stop that axis and retain both original map identities here.

The next hypothesis comes from the [local derivation](clock-generator-mathematics.md).
Replace only the original frozen generator DP by cP+diag(s-c), where c=.6
is the existing gate midpoint. Compare

```text
mul: F(h)=h+D(h)Ph
add: F(h)=h+cPh+(D(h)-cI)h.
```

This keeps K1, both independent solves, middle SiLU, gate range/tau, hidden
width and training objective. It changes the controller's receiving feature
from Ph to h. No temperature/range change, spectral regularizer, whitening,
sharing, gamma sweep or new loss is combined with it. K1 is an Euler generator
polynomial; the proposal does not claim higher integration accuracy.

## Cells and matching

Six arms: scalar_mul, scalar_add, vector_mul, vector_add, swiglu_budget
(hidden96), swiglu (hidden128). Width64/depth3, scalar/vector controller
hidden64/32 and reference_hidden64 retain existing budgets. Zero the last
controller affine in all four ODE arms; D=cI initially. Each mul/add pair has
identical state tensors, RNG draws, initial nonzero-head function, parameter
count and dense linear MACs. Nongate tensors also match across scalar/vector;
their small budget difference is reported, not hidden. All gates subsequently
train normally. This initialization is held fixed across the comparison.

Gaussian2D, Ring8 and rotated product16, existing versioned data and cached-bank
pair labels; seed0,5000updates,batch512,AdamW3e-4,wd0,clip1. Existing evaluation
streams, five-time complete velocity/Jacobian diagnostics, validation SW2 and
Heun32 remain unchanged. Eighteen total cells. Complete logs and checkpoints
at0/1000/2000/3000/4000/5000 are retained.

Run all arms on the Mac mini M1 CPU, torch2.10.0, in a separate frozen runtime:
two one-thread workers and 10s process/memory/progress telemetry. No GPU is
reserved. Both SwiGLU controls are rerun on the same platform. Previous
Windows controls motivate this study but are not substituted for new Mac
controls. Hardware microbenchmarks are outside formal quality evidence.

## Decision, fixed before outcomes

Continue an additive arm only if it improves primary validation oracle MSE
by at least10% over both SwiGLU sizes on a nonlinear task and improves its
same-family mul parent there; is at most5% worse than the better SwiGLU on
every other primary task; and has complete Jacobian error and validation SW2
within10% of the better SwiGLU on both nonlinear tasks. These are practical
screens, not significance tests. Report both additive candidates and all
controls. If both pass, order by product16 oracle MSE, then linear MACs,
then Ring8 MSE. Report actual wall time, including extra pointwise operations;
equal dense MAC counts do not guarantee equal speed.

If neither passes, stop these settings without order/temperature/gain rescue.
Passing nominates a longer matched-budget study, a new independent holdout,
and a controller-response gain/direction control. It does not establish a
winner, prove the scalar sign mechanism explains dense hidden networks,
or meet the original practical architecture objective on its own.

## Required checks

Before launch: production scalar/vector full Jacobian and P-gradient tests;
finite-controller direction-switch witness; identical initialized nonzero-head
functions/gradients where predicted, RNG and budgets; unsupported combinations
rejected; unchanged defaults and uniform-paired predecessors retain three-step
bit-identical optimizer trajectories. Record source/dataset hashes and actual
runtime/interpreter paths. A short seed0 trainer/resume workflow check is
separate from formal cells.

After completion: frozen-source/checkpoint/config hashes; expected initial
pairing and zero last-controller affines; identical snapshot training-generator
states across arms; independent recomputation of oracle and common-point
field/Jacobian arrays. Save all primary endpoints and screen decisions even
for negative outcomes. Do not modify old runtimes or submitted Report1.
