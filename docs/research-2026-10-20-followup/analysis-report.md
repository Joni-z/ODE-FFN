# Frozen-order and operator-geometry analysis

This is a descriptive, seed-0 analysis. It does not establish an architectural
improvement on images or a statistically significant population-level advantage.

## Questions and evidence

1. Does the trained ODELayer benefit from a more accurate exponential action,
   with learned parameters held fixed?
2. Do similar separately trained K4/K8 scores imply interchangeable operators?
3. Do eigenvalues and the largest singular value determine hidden-flow gain?

The frozen-order stage contains 60 ODE interventions plus six unchanged controls.
All 66 generated-array hashes and 18 native checkpoint reproductions passed.
The maximum native-versus-archive SW2 drift was 4.47e-8. The precision control
also passed for each checkpoint. `collection-audit.json` records the checks;
`manifest-145.json` binds inputs and scientific source to hashes.

A post hoc Ring8 check evaluates twelve fixed model/order combinations at four
solver resolutions: 48 cells, all sample hashes verified. Every Heun32 score
and generated array reproduces its preceding-stage counterpart exactly on 145.
No weights, training seed, or optimization settings were changed.

## 1. Low-order checkpoints are adapted to their finite polynomial

![Complete order matrix](figures/09-frozen-order.pdf)

The full matrix is important: no evaluation order was selected as a best test
configuration. Replacing K1 or K2 by the direct matrix exponential worsens SW2
on every target, by 16.1% to 238.9%. These are six fixed-checkpoint comparisons,
not six independent seeds. The exact numerical table follows below.

Ring8 K2 changes from SW2 0.184820 to 0.381333 and oracle velocity MSE 0.028707
to 0.169435. The latter is evaluated on common interpolated validation inputs,
before numerical generation. Thus endpoint integration alone cannot explain
the changed field fit.

![Representative fixed-index endpoints](figures/11-ring8-examples.pdf)

The displayed points are the first 1,000 actual test/source indices, not selected
for visual effect. K2-to-exponential produces broader bridges and altered
allocation around the eight modes. These examples support inspection of the
distribution, not a quantitative claim about individual mode coverage.

Interpretation: learned weights and surrounding layers are adapted to the
finite polynomial used during training. Removing its truncation error is a
functional change, not a guaranteed accuracy improvement for flow matching.
This does **not** show that a model trained with exact exponentiation would fail.

## 2. K8 is near-exponential, but K4/K8 are not interchangeable

K8-to-exponential changes SW2 by at most 0.0030% in this stage. Fixed-input
local action errors are small, and endpoint RMS changes are at most 2.57e-5.
This supports numerical saturation at K8 for the measured checkpoints.

However, on Ring8, frozen K4-to-K8 worsens SW2 by 3.225%, while frozen K8-to-K4
improves it by 2.672%. Separately trained K4/K8 scores differ by only 0.198%.
Training adaptation can therefore hide a residual functional difference.
The initial blanket interpretation of K4/K8 similarity as sufficient
evaluation accuracy must be revised. These are performance changes, not
equivalence or significance tests; the prespecified 1% flag is descriptive.

![Fixed-input error](figures/10-fixed-input-error.pdf)

For this diagnostic only, each layer's input and gate value are captured from
the native network and reused. Error decreases with evaluation order, even when
the full-network SW2 worsens. Full-network interventions freeze parameters but
allow later hidden states and gates to change. Conflating these two interventions
would overstate what is isolated.

## 3. Solver resolution does not remove the Ring8 findings

![Solver resolution](figures/13-solver-check.pdf)

At 256 Heun steps, K1-to-exponential changes SW2 from 0.201174 to 0.685820;
K2-to-exponential changes 0.186130 to 0.383708. Frozen K4-to-K8 still worsens
SW2 by about 3.1%, and frozen K8-to-K4 still improves it by about 2.6%.
The same-update SiLU/SwiGLU control ranking is preserved.

The differences survive four tested resolutions. This weakens an explanation
based mainly on the original 32-step solver. It is finite-resolution evidence,
not a convergence theorem. The check was selected after observing Ring8's
intervention effects; Gaussian and Moons interventions were not all re-evaluated
at finer resolutions.

One useful counterexample is Ring8 trained K4: exponential substitution slightly
reduces oracle MSE (0.034378 to 0.033714), but increases SW2 (0.175176 to
0.180832). Lower aggregate velocity error need not improve the induced endpoint
distribution. Error timing, input-distribution shifts and nonlinear sensitivity
remain possible explanations; none is isolated by these experiments.

## 4. P-only constraints are incomplete descriptions

![Controlled geometry and learned P](figures/12-geometry-and-training.pdf)

The 3D family `P(theta) = -0.25 I + a(theta) N(theta)` has
`N[0,1] = cos(theta)`, `N[1,2] = sin(theta)`, all other entries zero.
Choose positive `a(theta)` to make `||P||2 = 1`. Since `N^3 = 0`, all eigenvalues
are exactly -0.25 and
`exp(sP) = exp(-s/4) [I + saN + (sa)^2 N^2 / 2]`.
At s=1, exact gain increases from 1.225180 to 1.397337 over the five shapes,
while eigenvalues and largest singular value remain fixed.

This is an illustrative counterexample to sufficiency of those two summaries,
not a new theorem or an isolated causal effect of a single non-normality measure.
Other singular values and geometry change. It does not establish that any
specific constraint improves generation.

Earlier training evidence complements it: the spectral cap reduces the measured
full-layer Jacobian norm in all six comparisons but improves SW2 in only three.
The full layer derivative includes the gate's rank-one correction, so P alone
is not even the complete hidden-layer Jacobian. The learned singular-value
concentration is descriptive; it is not evidence that flattening the spectrum
will help.

## Exact fixed-checkpoint summary

Values are from identical fixed sample/direction sets at Heun32. SW2 is lower
better. Endpoint RMS is componentwise RMS relative to the same checkpoint's
native computation. Native reproduction and same-order precision controls
justify comparison across the two implementations.

| Target | Trained K | Native SW2 | Exact SW2 | Change (%) | Endpoint RMS |
|---|---:|---:|---:|---:|---:|
| Gaussian | 1 | 0.06106055 | 0.18407348 | +201.4606 | 0.19196232 |
| Gaussian | 2 | 0.06743678 | 0.10627661 | +57.5944 | 0.07643213 |
| Gaussian | 4 | 0.07030911 | 0.06964251 | -0.9481 | 0.00245165 |
| Gaussian | 8 | 0.07022126 | 0.07022125 | -0.0000 | 0.00000025 |
| Ring8 | 1 | 0.20082606 | 0.68055439 | +238.8775 | 0.85954982 |
| Ring8 | 2 | 0.18482010 | 0.38133332 | +106.3268 | 0.43260607 |
| Ring8 | 4 | 0.17517591 | 0.18083210 | +3.2289 | 0.02130676 |
| Ring8 | 8 | 0.17482907 | 0.17483427 | +0.0030 | 0.00002568 |
| Moons | 1 | 0.03150706 | 0.08852708 | +180.9754 | 0.15465112 |
| Moons | 2 | 0.02708329 | 0.03144395 | +16.1009 | 0.03052979 |
| Moons | 4 | 0.02768154 | 0.02781393 | +0.4782 | 0.00498435 |
| Moons | 8 | 0.02766445 | 0.02766444 | -0.0000 | 0.00000049 |

Machine-readable complete results: `frozen-order-results.csv` and
`frozen-order-summary.json`; local probes: `probes-145.json`; finer resolutions:
`solver-check-summary.json`. No rows are silently dropped.

## Claim candidates

- Claim: exact evaluation can harm a checkpoint trained at low Taylor order.
  - Source evidence: all six K1/K2 rows in `exact-effects.json`; Ring8 solver check.
  - Allowed wording: observed consistently across these fixed seed-0 checkpoints.
  - Forbidden stronger wording: exact-flow models are intrinsically worse.
  - Uncertainty: training adaptation versus specific weight compensation is not isolated.
  - Next check: only if selected, train a separately specified exact/high-order control.
  - Decision: keep.
- Claim: K8 saturates numerical action accuracy on measured checkpoints.
  - Source evidence: K8-to-exact SW2, endpoint RMS and local probes.
  - Allowed wording: small changes on the fixed evaluation set.
  - Forbidden stronger wording: globally equivalent functions or all K >= 4 equivalent.
  - Uncertainty: unseen hidden states, other seeds and image models are untested.
  - Next check: use validation-chosen hidden probes in a future training stage.
  - Decision: keep for K8; revise the former K4/K8 generalization.
- Claim: eigenvalues and maximum singular value do not determine finite-time gain.
  - Source evidence: five matched 3D matrices in `controlled-operators.json`.
  - Allowed wording: constructive examples with these two controls.
  - Forbidden stronger wording: non-normality is proven to cause poor FM quality.
  - Uncertainty: the examples are not learned generative networks.
  - Next check: relate gain and gate correction to training trajectories before choosing a constraint.
  - Decision: keep with scope.
- Claim: ODE-FFN is not uniformly superior to the baselines.
  - Source evidence: earlier same-update and forward-MAC-budget tables.
  - Allowed wording: advantages depend on target and budget in the tested runs.
  - Forbidden stronger wording: a statistically reliable ranking or image-generation gain.
  - Uncertainty: one seed, endpoint sensitivity and approximate compute matching.
  - Next check: precommit a validation endpoint rule for any future comparison.
  - Decision: keep.

## Decision for Report 1

The project now supports a mechanism-focused progress report: explain the original
layer mathematically, show its useful and negative controls, establish that
numerical exponential accuracy is not the learning objective, and distinguish
learned adaptation from saturation. Stop this diagnostic stage. Future training
should answer one new question at a time, with the research direction chosen by
the user. No new architecture branch or spectral regularizer is selected here.
