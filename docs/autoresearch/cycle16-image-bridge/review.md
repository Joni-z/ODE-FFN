# Returning to the image Transformer question

No new operator is selected by this review. The next necessary work is an
image baseline bridge: literal professor K1/K2 and ordinary SwiGLU controls
inside the existing JiT token/attention backbone. It does not rehabilitate
a failed synthetic candidate or replace the synthetic evaluation.

## Why another covariance field is not the next experiment

The tempting construction is f(h)=U^T softmax(Uh)-h, whose Jacobian is a
covariance-shaped matrix minus identity. Its gradient-flow interpretation is
established Hopfield algebra. Our [Stage9 protocol](../../research-architecture-search/protocol-covariance.md)
already tested tied, untied and linearized versions and explicitly stopped
higher-order, atom-count, temperature and initialization rescues. Its
[analysis](../../research-architecture-search/analysis-covariance/analysis-report.md)
finds ordinary SwiGLU better on both nonlinear tasks. A K2 version would
violate that stop; it is rejected before implementation.

The complete-gradient readout, cubic split, shared recurrence, time adapter
and Fourier-feature reviews also remain stopped. Cycle15 further rules out
direct hidden poisoning by one large forced-normal loss component. None of
these observations supplies a new trainable matrix or optimizer trick.

## What the literature does and does not justify

- [Hyper-SET](https://arxiv.org/html/2502.11646), inspected v3, derives symmetric
  attention/FFN modules and recurrent depth from a hyperspherical energy.
  It is relevant architecture prior, not evidence for our polynomial layer.
- [Basin-preserving Hopfield discretizations](https://arxiv.org/html/2608.21304v1),
  August2026 preprint, distinguishes dissipation, basin behavior and numerical
  order. Its assumptions concern retrieval energies; they do not transfer to
  an arbitrary complete FM network. The log-partition field is not new here.
- [JiT](https://arxiv.org/html/2511.13720v1), Sections3–4 and AppendixA, compares
  image prediction in raw-pixel Transformers. Its main results concern large
  patches on ImageNet, not a tiny CIFAR model or an ODE FFN. It establishes the
  relevance of the prediction-space/architecture question, not an ODE benefit.

## The unfilled experimental scope

Recent controlled experiments use a tokenless vector network. A64D hidden
feature followed by a128D global readout has a fixed output-null subspace.
The image model instead predicts48pixel coordinates per4x4RGBpatch from64
hidden channels at each token, with attention coupling64tokens. A full-row-
rank patch head is possible. The same dimension-forced null-space theorem
does not follow; neither does an ODE advantage. Real images also lack our
analytic tangent plane and oracle vector field.

This motivates checking the original question in an actual image Transformer,
while retaining the failed synthetic results as failed results. It is not
permission to move covariance/frame/C variants onto CIFAR in search of a win.

## Concrete next step and boundaries

Prepare official CIFAR-10, unconditional labels discarded, and fixed training/
validation indices drawn only from its50000training images. Keep the official
10000test images unopened for selection. Use the existing JiT code without
modifying src, datasets/evaluators from sealed studies, or submitted Report1.

First verify literal input-clock semantics, common surrounding initialization,
P gradients, model/checkpoint restoration and CPU/MPS throughput. No quality
training is launched before a fixed budget and evaluation protocol are committed.
The preparation/throughput result is not an architecture screen and cannot
complete the useful-successor objective. Generated sample quality and a
proper feature-space image metric will still be needed for later image claims.
