# What useful ODE computation remains after the constant-core control?

## Research question
Can a precisely specified internal ODE restriction supply task-relevant
computation or a reproducible inductive advantage beyond ordinary SiLU with
matched initialization and the constant-P training coordinates?

## Current evidence
The registered20k CIFAR screen gives nativeK1FID173.436914, cold fixed0.6P
174.708609, matched-init ordinarySiLU178.056840, defaultSiLU64190.755168 and
same-c posthoc native216.070195. Fixed P passes the unchanged allocation rule
with half the FFN matrix work. These are finite ordinary-class controls.
The current conditional-mean target is not an invariant endpoint map.

## Working hypothesis to examine
The next method must add a useful computation or justify a useful restriction;
an ODE interpretation, higher approximation order or a larger P norm is not
enough. The remaining native advantage is not identified as a gate mechanism.
A larger controlled image setting may be needed to establish transfer, but a
scale study alone cannot establish an ODELayer successor.

## Missing evidence
A distinct operation, the assumptions linking it to the actual denoising
target, strong ordinary realizations at measured cost, and matched empirical
quality beyond the exposed small-image screen. Literature performance in a
different target/representation is not this evidence.

## Support and falsification
Support requires improvement against both the stronger constant/matched-init
controls and ordinary SwiGLU under a fixed protocol, while retaining fitting,
sampling, cost and matrix histories. Reject a proposed mechanism if it folds
into an unacknowledged static map, relies on a false closure/consistency claim,
changes the regression target silently, or repeats an already failed setting.
No thresholds for a new experiment are selected from future outcomes.

## Minimal next action
Read a few primary sources against the actual stop rules. Select a concrete
bounded test only if that review supplies a new reason to run it. Otherwise
record the exact missing evidence and take the next available research step;
do not mark the broader research objective complete.


## Current decision
The train-only reference is complete and failed its fixed gates. See
[ER-20261028-gaussian-reference-01](literature-review.md#er-20261028-gaussian-reference-01--completed-ordinary-reference).
The [linear-support review](manifold-transfer-check.md) also supplies no new
P constraint. The broader successor question remains unresolved.

## Next eligibility question, before proposing a method
Question: can a finite posterior-state evolution avoid the specific
covariance-only closure error, and does its proposed layer reduce to an
ordinary attention/RBF computation?
Type: exploratory theoretical eligibility, not an architecture experiment.
Hypothesis: a finite parametric family may carry the third-moment contribution
through component state, but this may only re-express an ordinary nonlinear
map and offer no new computation.
Why it matters: [cycle22](../cycle22-denoiser-dynamics/analysis-report.md)
refutes the prior-independent covariance-only evolution, while the cold
controls show why an ODE name or change of coordinates is insufficient.
This question is not an inference that a Gaussian reference failure proves
such a layer useful.

Current evidence: the explicit third-moment counterexample, exact existing
mixture oracles, fixed-P/ordinary equivalence and the completed Gaussian
reference. No measured gain supports a finite-state layer.
Missing evidence: a correctly defined state/predictor, assumptions linking it
to the clean conditional mean, complete derivative and an operation that is
not merely ordinary attention at a different temperature.

What would support eligibility: exact finite-family algebra that retains the
third-moment term, distinguishes fixed observation from a moving FM state,
and identifies a nonredundant operation with explicit ordinary controls and
costs. This would only justify a later experiment, not a benefit claim.
What would reject it: a false closure, a silent target change, privileged
mixture labels/centers/basis, folding into an ordinary map without a distinct
restriction, or reopening an existing stopped setting.

Minimal next action: derive the mean/covariance/component-weight evolution
for a fixed finite Gaussian-mixture prior in precision coordinates, then
compare its fixed-time predictor to an ordinary softmax/RBF realization.
Check the complete state/time derivatives and the actual object before
nominating any implementation. No image prior refit, new data, Gaussian-mixture
reference, optimizer sweep, target change or training is selected. Keep all
old stop rules and the stronger ordinary controls.
Decision: explore this mathematical eligibility question only.
