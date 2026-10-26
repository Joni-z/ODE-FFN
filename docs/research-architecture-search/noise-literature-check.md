# Posterior supervision: exact conditioning versus finite candidate estimates

[Posterior Augmented Flow Matching](https://arxiv.org/html/2605.00825v1), May2026 preprint, is directly relevant to the noisy-supervision question. Sections3–4 distinguish the FM path distribution, posterior target supervision and a practical self-normalized importance-sampling approximation. Its image experiments concern a different implementation and do not establish an ODELayer advantage here.

Our diagnostic uses the analytic conditional mean under a matched fresh population. The squared-risk and expected-gradient identities then follow directly from conditional expectation, up to the recorded floating-point implementation. We do not need a finite-candidate approximation to justify them.

The paper's displayed variance/ESS expression should not be treated as a general exact identity for random self-normalized importance weights. The weights depend on the sampled endpoints and hence on their target values. A minimal counterexample makes the missing qualification visible.

Let a prior/proposal put probability1/2 on Y=0 and Y=1. At a fixed intermediate state let the relative likelihoods be1 and9, giving exact posterior mean E[Y|x]=.9. Draw two iid proposal candidates and normalize their likelihood weights. The estimate and its probability are

| Candidate pair | Probability | Normalized weighted estimate |
|---|---:|---:|
| (0,0) | 1/4 | 0 |
| (1,1) | 1/4 | 1 |
| (0,1) or(1,0) | 1/2 | .9 |

The finite estimator has mean.7 rather than.9 and variance.165. The posterior variance is.09; the expected inverse ESS is.66, so the proposed product would be.0594, not.165. The likelihood ratio is attainable with an independent Gaussian linear path, so this is not an abstract sign-changing-weight example. Affine conversion from endpoint to velocity preserves the bias issue.

This counterexample concerns iid-proposal SNIS. Including the actual generating endpoint in a specially symmetrized candidate construction can change the joint law and requires its own analysis. It does not refute exact posterior conditioning, nor assess every augmentation or retrieval implementation in the paper. It is why the present study uses an exact teacher and records the input law instead of assuming that a large candidate set automatically produces an unbiased control or an ESS-determined variance.

No posterior-aggregation loss is added to a proposed final architecture in this stage. The objective remains to find an ODELayer design that is useful under ordinary pair-target training.
