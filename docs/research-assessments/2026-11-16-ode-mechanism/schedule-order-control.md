# Next mechanism control: cancel stage drift without removing parameters

Status: proposed, not implemented or trained. Finish publication of the
completed density analysis before execution. No response search, spectrum
penalty, optimizer change or image run is part of this proposal.

## Question from the completed evidence

The released models use stage directions and ordering, but release also
changes capacity and optimization. Mean-preserving coordinates did not give
a resolved density advantage over shared P; on Product16 they worsened KL
against anchored coordinates. Thus the next comparison should hold parameter
coordinates fixed while changing one composition constraint.

Hypothesis: suppressing the leading order-dependent stage term improves the
learned generative field at a fixed parameter and matrix-work budget. The
alternative is that this term is useful, irrelevant, or outweighed by higher
orders. The experiment can reject the hypothesis; it does not presuppose an
ODE advantage.

## Matched local maps

Keep q(y)=y/sqrt(1+1.2*y^2), f(y)=Pq(y), g(y)=Bq(y). Use two trainable
matrices P and B in both models, W+=P+B and W-=P-B.

| Arm | Chronological four-stage order | Average matrix |
|---|---|---|
| Palindrome | +, -, -, + | P |
| Alternating | +, -, +, - | P |

Each stage is one explicit-midpoint update of duration tau/4. Both arms use
two occurrences of each matrix, eight matrix applications per local solve,
the same tensors and optimizer coordinates. Initialize B=0, so their initial
functions equal the shared-P reference exactly. Preserve all other weights,
the two local solves, intermediate SiLU and data-RNG stream. A new B allocation
must not consume the original initialization RNG. This is a matched pair of
piecewise time-dependent hidden ODEs, not an ODE-versus-non-ODE comparison.

## Mathematical prediction

The already derived four-stage expansion gives

```text
S_tau - C_tau = tau^2/32 * sum_(i<j) [f_i,f_j] + O(tau^3),
[a,b] = J_b a - J_a b.
```

Write f_i=f+s_i g. Bilinearity gives
`[f_i,f_j]=(s_j-s_i)[f,g]`. The sum of sign differences is 0 for the palindrome
and -4 for the alternating schedule. Consequently,

```text
S_pal - C = O(tau^3)
S_alt - C = -tau^2/8 * [f,g] + O(tau^3)
S_alt - S_pal = -tau^2/8 * [f,g] + O(tau^3)
```

Here C uses P at every stage, and

```text
[f,g](h) = B diag(q'(h)) Pq(h) - P diag(q'(h)) Bq(h),
q'(h) = (1+1.2*h^2)^(-3/2).
```

This cancels one leading term; it does not remove all stage effects or make
explicit midpoint self-adjoint. The previous unit-horizon approximation
errors 0.408/0.441 show why small-time cancellation cannot guarantee sample
quality. Symmetric neural compositions and splitting principles are established
[prior work](literature-notes.md); the potential contribution is a demonstrated
task-relevant constraint, not the cancellation algebra itself.

## Execution requirements before four trajectories

1. Independently check the coefficient/sign with exact linear matrix
   exponentials and nonlinear native midpoint maps. Verify cubic remainder
   for the predicted difference and the B=0 null. For B proportional to P,
   the leading bracket vanishes; finite numerical differences need not vanish.
2. Check both full native models at B=0 against shared P: outputs, input
   gradients, all common-parameter gradients and unchanged initialization RNG.
   New B gradients may differ by design; measure them rather than rescale them.
3. Reuse the pinned trainer, native continuation and actual evaluator. Require
   a fresh documented native pass, source freeze and publication before runs.
4. Run only the two arms on Ring8/Product16: four seed 0 trajectories, each 20k
   updates, batch 512, AdamW 0.0003, unchanged weight decay and clipping. Reuse
   shared P, Free ISRU and SwiGLU as fixed ordinary/reference controls on the
   same existing banks. Do not tune an extra B learning rate or initialize B
   from a favorable trained direction.

## Measurements and rejection rules

Report oracle field MSE, training-fit curves, all eight common4096-source
bank SW2 scores, pooled SW2, parameters and actual runtime. Check learned B
activity and feature-weighted brackets; B remaining inert makes a bracket
interpretation inconclusive. Repeat inner 4/8/16 and outer 128/256/512 evaluation
at frozen weights to distinguish stage-order effects from a coarse scheme.

Keep every task and bank. A useful primary within-pair generation difference
must survive refinement; a tie, reversal, or no gain rejects the corresponding
benefit claim at this budget. A lower MSE alone is insufficient. If sampling
supports further measurement, register the same paired all-source density
test before running it; a general transport claim must also address the
observed SW2/KL mismatch. No checkpoint or numerical setting is selected
after examining these outcomes.

Even a positive result would concern this composition constraint. Ordinary
controls decide whether it is useful beyond the ODE family; a faster or
more accurate ODE solve alone is not the contribution.
