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
is unnecessary. The already planned K1/K2 comparison is the discriminator;
ODE representability alone cannot establish an FM-specific bias. Complete the
registered D construction cycle first, then choose this test from its evidence.
