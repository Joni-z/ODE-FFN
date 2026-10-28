# What the new evidence permits

The [ordinary image controls](../cycle20-image-silu/analysis-report.md) remove
the native K2 FID lead at equal matrix work, with a parameter-count caveat.
The [clock collapse](../cycle19-image-clocks/analysis-report.md) also shows
that its local input variation is unnecessary for the recorded image ordering.
Neither observation identifies a new ODE computation worth training.

Three attractive shortcuts remain unsupported. More internal iterations
already failed the [recurrence screen](../../research-recurrent-depth-2026-10-26/results/analysis-report.md).
Free effective matrices did not repair the
[local teacher-fitting gap](../../research-effective-map-2026-10-26/analysis-report.md).
Hidden covariance dynamics passed an internal energy diagnostic but failed
the [ordinary-baseline screen](../../research-architecture-search/analysis-covariance/analysis-report.md).
An internal certificate cannot substitute for task quality.

## Relevant literature, checked28October2026

[PDE-SSM, Gal et al.,2026v1](https://arxiv.org/html/2603.13663v1), Sections2–3,
replaces token attention with a spatial convection–diffusion–reaction operator.
In Fourier coordinates its local linear dynamics have an analytic exponential
multiplier. Its rationale includes spatial structure and mixing cost, rather
than only resemblance to an outer ODE. This is direct prior art against
calling a generic heat/Laplacian token mixer a new idea. It changes a different
component from our channel FFN. The present review does not reproduce its
results or infer that copying the block improves JiT.

[Graph Flow Matching, Siddiqui et al.,2025v3](https://arxiv.org/html/2505.24434v3),
Section3.1, adds a graph correction to the predicted velocity. A node is an
entire VAE latent image; neighbors are other samples, not adjacent pixels or
patches. This distinction matters: its graph uses information across samples,
whereas our predictor acts independently on each image. A reaction–diffusion
analogy alone neither supplies our missing FFN mechanism nor licenses a
cross-sample change to evaluation. It is useful precedent for specifying the
evolving variable and neighborhood precisely, not a method selected here.

The older [Neural Flows, NeurIPS2021](https://proceedings.neurips.cc/paper_files/paper/2021/file/b21f9f98829dea9a48fd8aaddc1f159d-Paper.pdf)
is a mathematical precedent, not a frontier recommendation: its central
distinction is parameterizing a solution map instead of integrating a learned
field numerically. Our fixed-clock polynomial is a learned map with a flow
interpretation; the interpretation needs its own evidence. These sources do
not establish a theorem that an exact hidden flow improves FM.

## One remaining finite question

The old [synthetic frozen-order study](../../research-2026-10-20-followup/analysis-report.md)
already found worse generation after low-order-to-exponential substitution.
Do not turn that stopped avenue into a new trained exponential family. A
bounded image counterpart can instead determine whether the new image P
matrices are close to their claimed linear flows, and whether replacing the
polynomial preserves the frozen image behavior. Include matched matrix-
application arithmetic and retain every ordinary reference.

The [registered protocol](protocol.md) selects only that frozen diagnostic.
The [derivation](theory.md) separates an exact local-flow identity, norm bounds,
finite polynomial semigroup defects and unproved task-quality interpretations.
No spatial mixer, recurrence, new spectral constraint or training rescue is
selected by this review. A future candidate still needs a missing task-relevant
computation and evidence beyond these generic constructions.
