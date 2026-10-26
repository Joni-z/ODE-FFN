# Which solve needs its zero-order identity?

Registered after the audited Stage13 result, before any location outcomes.
Removing both identity terms improves product16 but does not beat SwiGLU,
and worsens Ring8. The matched-function vector control retains that tradeoff.
Changing both at once cannot identify the two terms' separate contributions.

## Factorial operator, one common initial hidden function

```text
F1(h) = gamma1*h + D1(h)*P1*h
u     = SiLU(F1(h))
F2(u) = gamma2*u + D2(u)*P2*u
gamma1,gamma2 in {1,0}.
```

Compare MM, MI, IM and II (M=map, I=increment), crossed with the existing
scalar/vector gate granularity. Every controller starts with its last
affine weight/bias zero, giving D=cI,c=.6. For each increment solve set
P_i=P_i,legacy+I/c; map solves keep the legacy draw. All eight ODE arms then
share the complete initial hidden function mathematically, across both gate
families, with actual float32 defects measured. All common P/backbone/RNG
draws agree. Each family has identical parameter/MAC counts across patterns;
the existing scalar/vector budget difference is reported explicitly.

This is an identification policy held fixed across the factorial, not a new
temperature or loss modification. Controllers train normally after init.
Return all other components to the original setting: independent clocks,
tau10,sigmoid[.2,1],K1,middle SiLU,rates after P. No stopped reparameterization,
range change, whitening, nonlinear drift, sharing or spectral penalty is added.

## Mechanistic predictions and limits

The chain derivative is A2*diag(SiLU'(F1(h)))*A1, where
A_i=gamma_i*I+D_i*P_i+diag(P_i*z_i)*J_si. At the paired initial point all
complete solve/input Jacobians match. Removing the first identity adds an
identity-feature controller response propagated through the activation and
second solve. Removing the second adds a direct activated-feature response.
Their locations are therefore not interchangeable. The mixed-pattern result
cannot be inferred from either MM or II alone.

At both raw P zero, generic block gradients in (P1,P2) are respectively
(active,active),(zero,active),(active,zero),(zero,zero) for MM,MI,IM,II.
Both mixed patterns avoid II's joint-zero first-order degeneracy. Actual
paired initialization is not at raw P zero, and the common zero output head
initially masks all backbone gradients. These boundary identities are not
claims about the first training step or which pattern must fit better.

The specific hypothesis is that preserving one identity may retain the
Ring8 behavior while recovering some product16 increment benefit. MI's
direct second-controller modulation is a motivated possibility, not an
assumed winner. Both mixed patterns and both gate families are declared
before outcomes. Same initial functions do not imply the same global
state-dependent function families or optimizer trajectories.

## Frozen experiment and continuation

Ten arms: scalar_mm/mi/im/ii,vector_mm/mi/im/ii,small SwiGLU(ffn_hidden96),
full SwiGLU(ffn_hidden128). Unchanged width64/depth3 and scalar/vector gate
hidden64/32. Gaussian2D,Ring8,product16 use the validated data, cached-bank
pair labels,seed0,5000 updates,batch512,AdamW3e-4,wd0,clip1,existing evaluation
streams and Heun32. Two one-thread CPU workers,10s telemetry,no GPU lease.

Primary comparisons screen MI/IM/II against the same-family MM and both
SwiGLU budgets. Continue only if an arm improves primary oracle MSE by at
least10% over both SwiGLU sizes on a nonlinear task and improves its MM
parent there, regresses at most5% against the better baseline on every
other primary control, and keeps complete Jacobian error and validation SW2
within10% of the better baseline on both nonlinear tasks. Multiple passing
arms are ordered by product16 oracle MSE, then linear MACs, then Ring8 MSE.
These are practical screens, not p-values.

Report all factorial cells and main/interaction contrasts descriptively;
do not select a subset after seeing outcomes. If no arm passes, stop these
settings without scale/order/seed/shrinkage rescues. Passing only nominates
a longer matched-budget, fresh-holdout and gating/capacity-control study;
all original architecture completion requirements remain in force.

## Checks and provenance

Test the full chain Jacobian, zero-P boundary table, all-eight common initial
functions/RNG and within-family exact budgets. Record full-width CPU/CUDA
paired-function defects with nonzero heads using predeclared float64/float32
absolute and relative tolerances2e-13/3e-6. Defaults and earlier uniform
paired controls must match the frozen Stage13 predecessor's three-update
nonzero-head trajectories bit-for-bit.

Use a separate immutable runtime and explicit delta hashes. Audit expected
per-layer P shifts, zero controller finals, common tensors and all snapshot
data-generator states. Native oracle/validation audits, selected raw collection
and independent Mac final-field/Jacobian replay retain distinct scopes. No
previous runtime or submitted Report1 is changed.
