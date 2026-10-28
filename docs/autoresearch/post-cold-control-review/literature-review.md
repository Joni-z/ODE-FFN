# What the cold controls change in the literature reading

Zhizhe Zhang · October 28, 2026. Relevant primary sections were checked live;
this is a bounded narrative review, not a comprehensive survey or novelty proof.

## ER-20261028-cold-controls-01 — direct project evidence
Source: [sealed cold training](../image-clock-training/analysis-report.md),
seal aad29b214ba537a2ee2034c60987c98f220eef00ca97b0105fb2e3bbadffff9c.
Type: experiment artifact. Cold fixed P passes at FID 174.71; matched-init
ordinary SiLU improves 190.76→178.06; same-c frozen native is 216.07.
Supports: stronger ordinary controls and a distinction between cold training
and frozen intervention. Does not establish an ODE-specific mechanism.
Claim strength: supported for this fixed image screen.

## ER-20261028-jit-preconditioner-01 — keep the prediction object
[JiT v1](https://arxiv.org/html/2511.13720v1), Appendix B.2.
Type: full primary preprint, relevant sections inspected. Table 10 holds
v-loss fixed and reports FID50K 8.62 for direct clean prediction versus 35.49
and 46.25 for the two input-skip preconditioners. The network's direct target
changes when a noisy-input skip is added. This blocks treating a boundary
identity or velocity-like residual as a harmless replacement for our present
predictor. Its ImageNet setting does not predict a CIFAR FFN gain.
Claim strength: supported boundary on transfer; no new architecture nominated.

## ER-20261028-skipless-initialization-01 — architecture and optimizer differ
[Cutting the Skip v1](https://arxiv.org/html/2510.00345v1), Table 1 and Discussion. Type: full primary preprint, relevant sections inspected.
For ViT-Base/ImageNet, skipless AdamW accuracy 61.4% rises to 78.1% with the
proposed initialization, versus 80.3% for the skip baseline. With SOAP,
initialized skipless reaches 80.8%, versus 80.1% skip. Thus the full result
cannot be summarized as initialization universally outperforming residuals
under unchanged optimization. Its conditioning analysis also states a block
incoherence assumption. This supports taking initialization seriously, not
copying a vision-classification recipe into ODE-FFN.
Claim strength: observed literature contrasts; transfer to FM remains speculative.
The OpenReview PDF browser challenge was not claimed as an accessible review;
the arXiv v1 full text supplied these checked numbers.

## ER-20261028-lift-scope-01 — target-aligned inner progress is a separate choice
[LiFT v1](https://arxiv.org/html/2610.05538v1), Appendix C.1–C.2.
Type: October 2026 primary preprint. An intermediate prediction targets a
fraction of the correction from a detached anchor to the outer FM target.
Appendix C.2 explicitly distinguishes this supervision from a guaranteed
inner numerical ODE solver. Its learned recurrence need not satisfy a fixed
point or integrate a specified vector field. This is not evidence that the
professor's polynomial block should improve by adding loops. Our stopped
recurrence/feature-flow settings remain stopped.
Claim strength: supported description of the construction; no transferred gain.

## ER-20261028-fist-scope-01 — sample-level optimization changes the task
[FiST v1](https://arxiv.org/html/2609.32376v1), Abstract and Sections 3.3/4.1.
Type: September 2026 primary preprint. The staged shared Transformer starts
from a distilled teacher, then replaces regression with adversarial and
auxiliary classification endpoint losses through a pretrained latent feature
backbone. These ingredients and the class-conditional latent setting differ
from our unconditional clean-pixel regression. The paper motivates separating
regression and generated-distribution quality, but does not justify adding a
new FFN or importing its reported FID into our setting.
Claim strength: supported transfer boundary, not an eligible architecture here.

## Decision
The current evidence does not nominate another nonlinear Taylor, recurrent,
covariance-head, spectral or preconditioned-output architecture. Do not revive
those stopped settings or change evaluation to obtain a win.

Select one bounded ordinary reference before another neural design: a
train-only low-rank Gaussian density and its exact conditional-mean predictor,
with an isotropic Gaussian as attribution context. It asks how competitive
this explicitly linear statistical model is under our unchanged image banks.
It cannot causally explain a neural model, prove nonlinearity necessary if it
fails, or complete the useful-ODE-successor goal. The question and exact
[protocol](../image-gaussian-reference/protocol.md) are fixed before fitted
statistics or quality outcomes. No neural architecture is nominated.

## Classical methodological source, not a frontier recommendation
[Probabilistic PCA, Tipping/Bishop 1999](https://www.cs.columbia.edu/~blei/seminar/2020-representation/readings/TippingBishop1999.pdf),
Section 3.2, Equations 7–8. The density covariance retains leading sample
eigenvalues and uses the average discarded eigenvalue for the isotropic tail.
This is a classical baseline, not claimed as a new ODE method. The Gaussian
conditioning and sampler formulas in the new theory note are separately
derived for our time convention. The Microsoft-hosted PDF text extraction
was garbled; the published Columbia-hosted copy provided readable equations.
