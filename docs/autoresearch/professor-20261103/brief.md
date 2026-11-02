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
artifact. The next registered comparison removes the tied second-order term,
for both the original scalar and direct-input candidates. A smaller native
SwiGLU64 completes the parameter-budget comparison; SwiGLU128 already nearly
matches the direct K2 dense-MAC budget. These22 trials are currently running.

The chain remains falsifiable: if K1 retains the benefit, weaken the second-order
story; if a matched ordinary FFN explains it, retain that comparison; if field and
sampling disagree, inspect the frozen external-solver/time-error readout while
keeping primary Heun32 results. A useful method must survive those tests before
image transfer or any claim of an FM-specific inductive bias.

Primary artifacts: [logic and derivations](logic-chain.md), [spectral evidence](spectrum/findings.md),
[cycle1](cycle1-results/findings.md), [cycle2](cycle2-results/findings.md),
[current execution](cycle3-execution.md), [literature branches](literature-directions.md).
