# A linear-support theorem is not a constraint on hidden P

Zhizhe Zhang · October 28, 2026

## Primary evidence: ER-20261028-manifold-support-01

[Pi et al., Learning Manifold Data with Flow Matching, ICML 2026](https://proceedings.mlr.press/v306/pi26a.html).
Read the final publisher-linked PDF, Assumption 2.1, Theorem 3.1, Lemma F.1,
Limitations and Appendix H.1. Its geometry is exact linear support `X=U h`.
The theorem separates latent transport and analytic normal contraction.
Lemma F.1 recovers that support under noiseless assumptions. Appendix H
feeds projected coordinates using U and reconstructs the normal field.
The stated extension to general nonlinear manifolds remains open. These
are the paper's boundaries, not an empirical ODE-FFN result.

Primary PDF SHA256:
`50dda7cac7aa35120af9c15f8a076817eac364d6ded20d0f541cab68426f9701`.
Downloaded directly from the publisher-linked mlresearch/v306 repository;
OpenReview presented a browser challenge. Local text extraction was readable.

## Specializing the assumption to our path

This is our algebraic specialization, not a new theorem. For zero-centered
data with exactly zero support outside U and the uncapped path
`Z_t=tX+(1-t)e`, orthogonal projection gives

```text
Z_perp = (1-t)e_perp
E[X_perp | Z_t] = 0
v_perp(z,t) = -z_perp/(1-t),  0<t<1.
```

A nonzero fixed mean adds its corresponding affine term; it does not create
a learnable normal field. Our existing cap/discrete sampler also changes the
endpoint law and cannot be omitted when comparing generated samples.

## What our completed experiments already resolve

- [Cycles 07–08](../cycle08-readout-invariant/analysis.md) already separate
  known thin/thick normal geometry and prove an output-span sampling law.
  More hidden P computation cannot remove that fixed readout obstruction.
- [Cycle 11](../cycle11-conditional-frame/analysis.md) learned input-dependent
  hidden directions. Its thin improvement did not establish a higher-order
  mechanism or pass thick-data sampling. Rotating P is not a new consequence
  of the theorem and that setting remains stopped.
- The [Gaussian reference](../image-gaussian-reference/analysis-report.md)
  leaves 14.22% of empirical training variance outside its particular PCA59
  span. Thus that span does not discharge exact zero normal support. This
  observation does not refute a curved low-dimensional manifold, nor infer
  its local tangent from global PCA.
- The [cold controls](../image-clock-training/analysis-report.md) show that
  a fixed K1 core is an ordinary square linear map in shifted coordinates.
  A restriction on an arbitrary hidden-coordinate P still needs a connection
  to the learned full denoiser. Data-space U does not supply that connection.

## Decision

The paper supports separating a known analytic component from a learned
component under a stated data law. It does not validate a P-only spectral
constraint, equal singular values, another Taylor term, an oracle projection
in the image model, or another thin-only benchmark rescue. No new experiment
is selected from this reading. Higher-order outer dynamics and the professor's
internal polynomial order are also different quantities.

The unresolved design task is to give an internal state a defensible role
under the actual clean conditional-mean objective. The next eligibility
question below comes from the existing third-moment counterexample, not from
treating this paper or the failed Gaussian reference as evidence of a gain.
See the [question card](research-question-card.md). No new fitting, target,
oracle input, solver or architecture is introduced here.
