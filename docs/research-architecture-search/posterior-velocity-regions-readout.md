# The Product16 velocity gap mostly occurs in confident regions

The saved Stage17 predictions weaken an explanation based mainly on ambiguous component assignment. Coordinates with normalized posterior entropy below .1 contribute **78.26%** of the ODE parent's excess velocity error over SwiGLU128, and 78.21% over SwiGLU96. The most ambiguous coordinates, with entropy at least .9, contribute only 1.51% and 1.66%, respectively. This locates the error; the networks do not output component probabilities, so it does not directly measure classification mistakes.

The [protocol](protocol-posterior-regions.md) fixed four entropy bins before the new outcomes were computed. We used only 15 existing Product16 arrays: three final Stage17 models, five times, and 512 anchors per time. Error is measured in the distribution's known orthogonal coordinates, where each coordinate has a two-component posterior. The [CSV](posterior-velocity-regions.csv) retains every bin, time and model, including empty bins. Its five-time field MSEs reproduce the native values: ODE .256488, SwiGLU96 .106572, SwiGLU128 .088921. These are the saved five-panel errors, distinct from the primary uniform-time evaluation.

| Normalized entropy | Coordinate fraction | ODE conditional MSE | SwiGLU128 conditional MSE | Share of ODE–SwiGLU128 gap |
|---|---:|---:|---:|---:|
| [0,.1) | 47.51% | .390069 | .114031 | 78.26% |
| [.1,.5) | 9.39% | .447369 | .213469 | 13.11% |
| [.5,.9) | 12.61% | .173303 | .078755 | 7.11% |
| [.9,1] | 30.49% | .023910 | .015618 | 1.51% |

Time matters because early inputs naturally have ambiguous component identity. At t=.1, 95.72% of coordinates occupy the highest-entropy bin, but total ODE MSE is only .020841. Pooling alone would mix this relatively easy early problem with later ones. The same-time comparisons support the conclusion: at t=.7, the lowest-entropy bin contains 91.30% of coordinates and contributes **93.71%** of the gap over SwiGLU128 (96.19% over SwiGLU96). Within that bin, conditional MSE is .649492 for ODE, .220744 for SwiGLU96 and .169648 for SwiGLU128. More ambiguous intermediate coordinates can have higher individual error, but their small prevalence limits their contribution to the observed gap.

At t=.5, error is less concentrated: the four bins contribute 46.54%, 31.05%, 20.66% and 1.75% of the gap over SwiGLU128. Thus intermediate posterior uncertainty still matters; the result does not say that only confident coordinates need fitting.

For each latent coordinate the exact oracle is `v*(u,t)=A(t)u+B(t)tanh(kappa(t)u)`. In confident regions the second term is close to a signed offset, while the first still requires the correct within-component slope. The observed gap could reflect learning that slope, the signed offset, their time dependence, or unwanted cross-coordinate effects. Regional velocity errors cannot separate those causes. A generic sigmoid feature could help represent the signed offset, but these data do not establish that benefit or support an ambiguous-boundary-only motivation. They provide no basis to select a new ODE architecture by themselves.

The earlier [posterior-region analysis](posterior-regions-readout.md) instead studied Jacobian row error from older 20,000-step models; it did not supply this paired Stage17 velocity attribution. All old artifacts remain unchanged. [Provenance](posterior-velocity-regions.provenance.json) records the 15 input hashes, manifest, target bank, protocol and [analysis source](../../scripts/diagnose_saved_velocity_regions.py). Common inputs/oracles match exactly; oracle reconstruction differs by at most 2.67e-15, and rotation preserves MSE within 1.12e-16. These are exposed evaluations, not independent confirmation or an architecture-wide capacity bound.
