# Which Muon claims transfer to this project?

These are focused primary-source checks on3November2026, not new optimizer
adoptions. No paper is treated as a guarantee for our learned PD blocks.

## Associative memory: a data/architecture-dependent mechanism

[Wang et al., Muon Outperforms Adam in Tail-End Associative Memory Learning,
ICLR2026](https://proceedings.iclr.cc/paper_files/paper/2026/file/3eec5006051d9544e717067de3220198-Paper-Conference.pdf).
Read: abstract, introduction, preliminaries and Section3.1; the later theorem
proofs have not been reviewed here. The authors separate Transformer components,
including gated FFNs, and connect benefits to associative-memory learning on
imbalanced classes. Their stated theory concerns a one-layer memory model.

Inference for us: feature/data structure matters; an isotropic matrix spectrum
alone is not a sufficient explanation for a generative architecture gain. Our
co-adapting hidden features and continuous FM regression are not that theorem's
model. No rare-class benefit or universal superiority is inferred for our tasks,
and no class-imbalance dataset is introduced to obtain a desired result.

## A spectral-flow theorem is not a theorem for literal training

[Beneventano et al., The Spectral Dynamics and Noise Geometry of Muon,
June2026 v1](https://arxiv.org/html/2606.08388v1).
Read: abstract, Sections5.1/5.2,7 and limitations. The spectral result studies a
projected self-polar flow on an interpolation manifold, with constant rank and
simple singular values. Its normalized flattening criterion depends on measurement
geometry. A separate local bridge concerns gradient/momentum polar flow; the
paper explicitly distinguishes these from discrete Muon and limits the proposed
Transformer reduction.

Inference for us: this cannot certify monotonically flatter learned P or justify
a weight constraint in our nonlinear network. Frozen-feature interpolation,
shared-frame/polar alignment and the required rank/gap conditions have not been
established. Our native optimizer includes finite approximate orthogonalization,
momentum, stochastic gradients and other co-adapting parameters. The completed
constraint and feature-response experiments remain the evidence for this project.

## A controlled counterpoint

[Parviz et al., Reassessing Muon for Matrix Factorization,
TAG-DS2026/PMLR334](https://proceedings.mlr.press/v334/parviz26a.html).
Scope: published abstract only, not the full experimental appendix. Their tuned
low-rank-factorization comparison does not find consistent Muon dominance over
AdamW. It supports retaining tuned adaptive controls; it does not predict our
factorial outcome. This is a controlled counterpoint, not a new mainline method.

The current missing-corner experiment has its own algebra and fixed controls.
Its result must stand independently of these interpretations. No optimizer,
data, sampler, loss, threshold or training budget changes follow this review.
