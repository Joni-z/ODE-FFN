# Literature-guided branches,3November2026

These are primary-source screening notes, not a completed systematic review.
Keep the completed controlled results as the decision basis. No paper's optimizer
or training recipe is inserted into the running cycles.

## Spectrum shape, magnitude and inexact orthogonalization

[Muown,2026](https://arxiv.org/html/2605.10797v1) separates row magnitude and
coherence in its LLM experiments. Our archived P readout exhibits similar scale
versus coherence separation, but the harmful decay outcomes prevent inferring
that norm control alone will improve the ODE. This motivates measuring the actual
PD operator and update rather than treating a small raw P as the goal.

[Beyond the Ideal, AISTATS2026](https://proceedings.mlr.press/v300/shulgin26a.html)
studies approximate rather than idealized exact orthogonalized updates and reports
coupling with learning rate/momentum. Our ideal-polar rescaling derivation is
therefore explicitly labeled ideal; any later exact-SVD versus five-NS test needs
controlled magnitudes/rates, not a claim that more accurate orthogonalization is
necessarily better. Current cycles retain official five-step Muon unchanged.

## Optimization coordinates and factorization

[Riemannion, ICLR2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/8505a7e1da40320bf5692d7edec5aa45-Abstract-Conference.html)
uses fixed-rank matrix-manifold optimization for LoRA to address factorization
ambiguity. Its existence rules out presenting parametrization-aware optimization
in itself as a new idea. Our P D(h) has input-dependent diagonal factors and a
polynomial map, so a fixed-matrix LoRA method cannot simply be assumed applicable.
Full method/code review would precede any implementation.

Follow-up scoped PDF read: Sections3–4, Eqs6–14, of the
[conference paper](https://proceedings.iclr.cc/paper_files/paper/2026/file/8505a7e1da40320bf5692d7edec5aa45-Paper-Conference.pdf)
specify tangent projection, momentum transport, orthogonalization and fixed-rank
retraction. Orthogonalizing factors separately is not the intrinsic update.
Their tangent-projected orthogonal step is approximate; they report that more
accurately solving the tangent-space local subproblem brought little convergence
benefit in their experiments. This narrows the relevant prior beyond the abstract.
Our conditional PD(h) is not one fixed-rank adapter, so copying that optimizer
would require a new function-space metric and treatment of singular D. No such
implementation or novelty claim is made. The separate fixed-D calculation in
muon-metric.md is a local identity, not a derived Riemannion extension.

[Reassessing Muon for Matrix Factorization, TAG-DS2026](https://proceedings.mlr.press/v334/parviz26a.html)
reports that Muon's superiority over tuned AdamW is not consistent in its controlled
matrix-factorization setting. This supports our use of ordinary baselines and
all tested rates. It is not evidence that Muon must fail on the present FM tasks.

The direct algebra already gives a narrow task-specific question: equivalent
P,D coordinates can have identical functions but different raw spectra and
optimizer steps. Cycle1 supplies a frozen trained case; cycle2 supplies a
same-initial-function factorized-versus-full-affine comparison. Any observed gap
must distinguish rank, factorization/gradient geometry, and function class before
proposing a new optimizer. A quotient/natural metric is a prospective mathematical
route, not an accepted method or novelty claim.

## Architecture versus an ODE label

[GLU Variants,2020](https://arxiv.org/abs/2002.05202) attributes multiplicative
gating and different gate nonlinearities to established FFN designs. With
q(h)=d(h)*h our PD polynomial is

```
F1=h+P q(h)
F2=h+P q(h)+.5 P[d(h)*P q(h)].
```

A successful D may provide useful gated features even if the second tied term
is unnecessary. The completed K1/K2 comparison improves the original scalar in
all8 field comparisons and retains a small raw-D Ring8 lead. ODE representability
alone therefore cannot establish an FM-specific bias. Cycle4 now distinguishes
constant, bounded and shifted K1 features.

The same primary paper's Section2 explicitly includes a bilinear FFN with the
gate activation omitted: output=W3[(W1h+b1)*(W2h+b2)]. This is an established
quadratic feature reference, not a new proposal or a frontier-paper recommendation.
If a quadratic-feature arm remains the lead after cycle4, a native smaller
SwiGLU with just SiLU replaced by identity is a useful ordinary control. It would
keep the exact same learned tensors/initialization/linear-MAC budget as that
SwiGLU, while testing whether an ordinary learned bilinear feature explains the
gain without a tied ODE polynomial. Paper bias settings differ from our inherited
native implementation; record that difference rather than calling it an exact
paper replication. No such family is launched in cycle4.
