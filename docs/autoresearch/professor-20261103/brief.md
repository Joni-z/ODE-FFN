# Professor-guided ODEFFN exploration

The working question is whether an input-dependent matrix polynomial can make a
useful FFN for learning flow-matching fields. The present experiments use a small
time-conditioned synthetic velocity network. They establish neither JiT image
quality nor an advantage from ODE terminology alone.

The professor proposed examining Muon and replacing a scalar s(h)P by P D(h).
The completed comparisons now distinguish three things that were initially mixed:
update-spectrum shape, matrix scale, and the function actually computed after
input-dependent gating.

Muon spreads update energy across more directions. In the original Product16
scalar model, the selected higher-rate Muon run has update stable rank32.7 versus
AdamW's3.1. P energy is also less concentrated, but its spectral norm grows19.2
versus3.4. A read-only row/column decomposition shows that most of this observed
growth accompanies increasing row magnitudes, with much less growth of normalized
row coherence. Conditioning does not improve uniformly. Matching ordinary momentum
to the current hypothetical Muon update norm does not reproduce all field gains;
the same direction benefit can also occur in SwiGLU. Tested weight decay reduces
P and local Jacobians but worsens fitting. These results support an optimization
interaction, not an ODE-specific spectral mechanism.

The diagonal question is broader than where to place D. We tested PD,DP and a
balanced sandwich; then held PD fixed while changing nonlinear versus affine
controllers, bounded versus affine outputs, factorized versus full linear logits,
and direct input diagonals. Most changes fail to improve strong controls. Direct
D(h)=diag(h) is the clearest remaining local lead: on Ring8, field MSE0.01210 and
SW2 0.17270 improve on the tested scalar ODE, but remain behind stronger ordinary
SwiGLU results. On Product16 the same construction improves some field comparisons
while worsening sampling. This is not a general architecture win.

The algebra explains why P alone is an incomplete object of study. In the affine-D
family, P'=P C^(-1),D'=C D gives exactly the same PD and polynomial. A trained
checkpoint demonstrates a P singular norm changing1.44 to7.56 while the complete
prediction changes only2.2e-15 in float64. This exact scale freedom generally does
not hold inside the original fixed sigmoid range. It also changes optimization
coordinates: even ideal Muon at the same nominal LR need not produce the same
effective PD step. The relevant diagnostics therefore include PD, controller input
derivatives and the complete predictor, not only raw weights.

A separate one-dimensional example makes the limitation concrete: P=-1 has
benign spectrum and condition number1, yet an allowed sharp bounded gate makes
the K2 input-dependent map's derivative negative. Entrance-frozen exact exponentials
can also fold. This does not establish that folding harms velocity prediction;
it shows why an eigenvalue constraint is not a complete argument for a learned
conditional block.

The raw-input lead computes explicit quadratic/cubic features. Its very large
hidden Jacobians do not translate directly into huge complete-field Jacobians,
and frozen float32/float64 checks do not explain its field gain as a precision
artifact. The registered order comparison removes the tied second-order term,
for both the original scalar and direct-input candidates. A smaller native
SwiGLU64 completes the parameter-budget comparison; SwiGLU128 already nearly
matches the direct K2 dense-MAC budget. All22 trials completed and replayed.
Scalar K1 improves field error in all eight paired settings. Ring8 raw K1 reaches
0.01091, slightly below the strongest ordinary0.01123, but its sampling remains
worse than matched SwiGLU64 (0.17792 versus0.15038); Product16 still loses to
ordinary controls. The quadratic feature interpretation is now more credible
than a benefit from the tied second-order Taylor term in this setting.

The frozen Heun32/64/128 check also completed. Ordering does not change, and the
Product16 raw-K2 sampling gap to SwiGLU persists. Coarse integration is not
supported as the explanation for that frozen discrepancy. Time-binned field
errors describe the remaining mismatch without proving its transport cause.

These completed tests weaken the second-order explanation. The remaining lead
concerns feature construction and optimization, with an unresolved gap between
field fitting and sampling. A matched ordinary FFN remains the necessary control.
No current result justifies image transfer or an FM-specific inductive-bias claim.

Cycle4 completed a smaller K1 feature contrast: constant1, bounded direct sigmoid,
and h+1 diagonals, holding P/common tensors and all training settings fixed.
The constant arm is an ordinary bias-free SiLU FFN under W=I+P; the shifted arm
adds quadratic curvature at the same local value/Jacobian at zero. Afterward,
a frozen parity diagnostic verified a known property of both population targets:
u(-x,t)=-u(x,t). The shift1 arm does not improve the best result; the constant
ordinary FFN remains a useful control. On Product16, excess even energy accounts
for74% of rawK1's paired-error gap to Swi128. This is a diagnostic decomposition,
not evidence that removing it improves sampled quality. A raw-versus-abs diagonal
has exactly identical conditional singular values at matched P,h but even versus
odd increments. It is the next selected contrast, with a matched ordinary bilinear
FFN and all earlier SwiGLU controls. No symmetry projection or extra loss is added.

Primary artifacts: [logic and derivations](logic-chain.md), [spectral evidence](spectrum/findings.md),
[cycle1](cycle1-results/findings.md), [cycle2](cycle2-results/findings.md),
[cycle3](cycle3-results/findings.md), [solver check](solver-results/findings.md),
[cycle4 results](cycle4-results/findings.md), [parity readout](symmetry-results/findings.md),
[signed-diagonal branch](signed-diagonal.md),
[two ODE interpretations](nonlinear-flow.md), [literature branches](literature-directions.md).
