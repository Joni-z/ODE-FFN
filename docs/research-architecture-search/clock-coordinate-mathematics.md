# Clock coordinates can change learning without changing the ODE map class

## Evidence selecting this question

The matched fresh-population experiment removes conditional label noise but leaves a substantial fixed-budget ODE gap:2.43 times the better mean-supervised SwiGLU error on Ring8, and14.42 times on product16. Its native audits independently regenerate every sampled minibatch and match all checkpoint fingerprints. Noise is consequential, but does not explain away the relative fitting gap. This still does not distinguish nonconvex optimization from a representational limitation.

The next comparison keeps the original operator, every feasible function, parameter count, MACs and initial function fixed. It tests whether the coordinates used for the learned durations impede fitting. No spectral restriction or richer nonlinear field is added.

## An exact coordinate change

For one original controller, write hidden features a(h)=SiLU(Bh+b), final affine logit g(h)=W*a(h)+c and duration s=.2+.8*sigmoid(g/tau). Set rho=1/8 and transform

```text
W' = rho*W, c' = rho*c, tau' = rho*tau,
g'/tau' = g/tau.
```

For every input, s'=s. P is unchanged, so the original K1 map h+D(h)P*h and the two-solve composition have identical initial functions. The equivalence also holds for the original higher-order frozen-generator polynomial, but this screen uses the previously selected K1 only.

For any learned W',c' there is an inverse W=W'/rho,c=c'/rho. Thus the global feasible function class is identical; this is a parameterization study, not a new expressivity or stability theorem. All random initialization draws are consumed before applying the deterministic transformation. The factor8 is a fixed power-of-two scale bringing tau10 to1.25; binary scaling permits an exact floating-point initial-function check. It is not chosen from task outcomes or swept.

At paired parameter values and the same loss/input,

```text
gradient_W' = gradient_W/rho,
gradient_c' = gradient_c/rho,
gradient_other = gradient_other_reference.
```

The derivative with respect to the original hidden feature parameters B,b remains unchanged, since W'/tau'=W/tau. P and the surrounding backbone derivatives also remain unchanged. The full-network nonzero-head test checks these response relations; comparing zero output heads would be uninformative.

For a fixed batch, flatten the network's outputs and split its parameter-response matrix into last-clock and other columns. The corresponding Euclidean tangent kernels obey exactly at paired parameters

```text
K_reference = K_other + K_last,
K_reparam   = K_other + 64*K_last = K_reference + 63*K_last.
```

The addition is positive semidefinite. This identifies a change in the weighting of existing response directions, not new directions or a larger function class. In squared-loss full-batch gradient flow, the instantaneous residual decay increases in the last-clock feature directions. A discrete optimizer can still need a smaller stable step, and a larger kernel condition number is possible; neither Adam nor stochastic generalization is covered by that gradient-flow observation.

The actual network's output head starts at zero, so every backbone parameter response, including K_last, is zero at step0. The first head update is identical under the paired transformation. The clock-coordinate effect can begin only after the head is nonzero; the kernel identity is then informative at paired parameter values. This is why the derivative test installs the same nonzero head rather than using a vacuous zero-output comparison.

Conditional gradient signal and noise covariance in this parameter group both scale by64, so its instantaneous signal/noise ratio, where defined, is unchanged. The coordinate change is not a denoiser. Under correspondingly scaled Adam moments and negligible epsilon, adaptive normalization cancels one power of the scale, leaving an approximately eight-times weighted last-clock response rather than the Euclidean factor64. Joint clipping and evolving moments require the explicit qualifications below.

## Why Adam does not make the coordinate change irrelevant

Adam normalizes gradient scale, but its parameter step is still expressed in parameter coordinates. In invariant clock coordinates C=W/tau, ignoring joint clipping for this derivation, the same Adam step on W is exactly an Adam-form update on C with

```text
learning_rate_C = learning_rate_W/tau,
epsilon_C = tau*epsilon_W.
```

Consequently the new coordinates, trained with the unchanged optimizer settings, provide an eight-times larger nominal step in C. This is not the claim that small gradients necessarily train slowly under Adam; it is a change in function-space response to a comparable parameter step. Joint clipping and time-dependent moments can modify the realized response, so neither an eight-times speedup nor better quality follows as a theorem.

An exact equivalence control states what would be needed to undo the coordinate effect. With zero weight decay, transform the last-affine Adam state and settings as

```text
m' = m/rho, v' = v/rho²,
learning_rate' = rho*learning_rate,
epsilon' = epsilon/rho.
```

To preserve global clipping, compute its norm in the reference coordinates:

```text
norm_reference² = sum_other ||gradient'||²
                  + rho²*(||gradient_W'||²+||gradient_c'||²).
```

Apply the resulting common clipping factor to all reparameterized gradients. Induction on the Adam recurrence then gives W'_new=rho*W_new, c'_new=rho*c_new at every update, with the other parameters identical. A double-precision20-step test checks this with active clipping, including the epsilon and moment recurrence. This is a mathematical control, not an additional optimizer modification used to improve the candidate.

The actual candidate keeps ordinary AdamW3e-4,epsilon1e-8,wd0 and ordinary global clipping1 for every parameter. Only the clock coordinate representation changes. The comparison therefore measures parameterization plus the optimizer's response to it, rather than new operator capacity. Any useful result must still appear under ordinary pair-target FM and beat both practical SwiGLU references.

## Distinguish coordinate geometry from initial gate variation

Compare the exact-function-paired transform against an additional tau1.25 arm that leaves initialized W,c unscaled. The latter changes the initial duration function as well as the coordinates. Both have the same global function class, counts and linear MACs, but only the paired transform controls the initial function. Record actual durations and the invariant norms ||W/tau|| and ||c/tau||, not just the raw controller weights.

If the paired transform helps, it demonstrates that a bad fixed-budget result was not an expressivity impossibility. If only the unscaled arm helps, altered initial functions remain a competing explanation. If neither helps, stop this coordinate scale; do not rescue it with a scale sweep, wider gates, new losses, caps or higher order. Passing a screen is only a lead; the full goal still requires longer training, frozen independent verification, derivative/generation quality and measured cost.
