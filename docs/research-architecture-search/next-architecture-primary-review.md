# What recent architecture papers actually support

Read-only review, 2026-10-25. No architecture or training run is selected. This bounded search found **no verified, cost-controlled evidence that a hidden conditional linear Taylor block should outperform SwiGLU in our setting**. Three primary sources are useful because their ablations expose the limits of apparently relevant architectural gains.

## 1. DyDiT++: allocate computation across physical diffusion time

First submitted **2025-04-09**; reviewed version **v4, 2026-01-14**. The authors report TPAMI acceptance on 2026-01-09. [Version record](https://arxiv.org/abs/2504.06803)

The change is time-only masks over attention heads and FFN channel groups, plus token-dependent FFN bypass. It allocates existing computation; it does not integrate a hidden linear field. The [official DiT implementation](https://github.com/alibaba-damo-academy/DyDiT/blob/0ddc5097d2bc38c9b68b6c32ac6b53adadc1f865/DyDiT/models.py#L296) confirms separate time and token routers. This repository does not independently establish the newer SiT results.

DySiT Table 7 reports 58.38 versus 118 inference GFLOPs with identical ODE-sampled FID 2.11; parameters are 678M versus 675M. Removing token bypass at the half-compute target worsens FID to 2.73 (Table 30). From scratch, 400k updates give **19.43 versus 17.20**; 600k give 17.21 (Table 31). [Full text, Sections 9 and B.12–B.13](https://arxiv.org/html/2504.06803v4)

**Cost-scope clarification:** training multiplies dense features by masks, uses straight-through/Gumbel estimators and adds a FLOPs loss; sparse execution concerns inference. Reported active FLOPs therefore cannot be multiplied by updates to establish executed training MACs or wall time. The fine-tuned result also inherits pretraining cost. Its inference tradeoff remains valid; our comparison does not establish total training efficiency. [Section 3.4](https://arxiv.org/html/2504.06803v4#S3.SS4)

## 2. Diff-MoE: distinct feature processors with time conditioning

Published at **ICML, 2025-07-13–19**, PMLR 267. [Proceedings record](https://proceedings.mlr.press/v267/cheng25d.html)

Each token selects experts, with additional shared experts. Equation 5 writes a weighted sum of expert FFNs with expert-specific time conditioning; the expert FFNs use SiLU-gated value projections. This changes which feature processors are available, rather than repeatedly applying one hidden operator. [Primary PDF, Sections 4.1–4.4](https://raw.githubusercontent.com/mlresearch/v267/main/assets/cheng25d/cheng25d.pdf)

Table 6 isolates adding expert-specific conditioning: FID improves **32.74→28.71**, but activated parameters increase **38.37M→49.02M**. Subsequent global recalibration and low-rank components change the comparison again. Table 1 also distinguishes 36M activated from 107M total parameters for the small model. These are not equal-total-parameter or demonstrated equal-MAC controls against our SwiGLU.

The [released implementation](https://github.com/kunncheng/Diff-MoE/blob/468d3793e8b649157e77eecf5baabc33d387284b/models.py#L205) is more specific than Equation 5: each expert normalizes/modulates its input, applies SwiGLU with pooled-context multiplication before down-projection, and returns an internally gated residual. Consequently, the printed post-FFN AdaLN expression should not be copied as an exact implementation specification. The diffusion result motivates examining conditional feature allocation, not adopting this multi-component design or assuming an FM benefit.

## 3. RAE: a useful counterexample to transferring architectural gains

First submitted **2025-10-13**; subsequently **ICLR 2026**. Numerical details below refer explicitly to the arXiv v1 full text. [Version record](https://arxiv.org/abs/2510.11690), [official proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/3c4141c12660ad3625eb4ae845e0a6f9-Abstract-Conference.html)

The shallow, wide DDT head separately receives noisy input and conditioning from a narrower semantic trunk; this is a prediction-architecture change, not a replacement activation. [Official forward path](https://github.com/bytetriper/RAE/blob/a4d18c4db766419cbe7cb8c02cd9f7ceb0ec9041/src/stage2/models/DDT.py#L339)

Theorem 1 concerns an input/output bottleneck `B f(Ax,t)` with hidden width below ambient dimension; its bound is not a theorem about an internal square `P`. Our input dimension 2/16 and hidden width 64 do not satisfy that bottleneck assumption. Most decisively, Table 10 reports that the added head **worsens low-dimensional VAE FID from 7.13 to 11.70**, while improving DINOv2 latent FID from 4.28 to 2.16. Appendix G.3 compares head widths/depths, but the overall recipe also changes latent representation, noise scheduling and training settings. This supports checking the actual representation bottleneck before transferring a successful head. It does not license another width/head rescue here. [Full text, Sections 4–6 and Appendices D, G.3](https://arxiv.org/html/2510.11690v1)

## One structural question, with a deliberately limited conclusion

The useful hypothesis is **whether the original block spends too much of its budget modulating a shared set of feature directions, relative to learning feature directions useful at different physical times**. This is a question about allocation and useful conditional variation, not an assertion that the network lacks time conditioning.

For the original K1 solve, `h + D(h) P h`, changing `D` rescales rows of the same `P`. Two solves, the middle SiLU, the controller's input dependence and external time FiLM make the complete network substantially richer than this single frozen-matrix statement. We cannot infer a full-network rank/capacity bound, approximate the learned clocks by constants, or identify physical time with hidden integration duration.

The local evidence makes a new training proposal premature. [Stage23](analysis-affine-forcing/analysis-report.md) leaves Product16 errors at .266155 for the parent and .269071 for conditional bias, versus .092326 for SwiGLU128. The [regional readout](posterior-velocity-regions-readout.md) places 78.26% of the parent's excess error in already-confident coordinates, ruling out an ambiguity-only account. Concentrated middle-time error does not establish conflicting gradients across times, missing expert directions, or insufficient conditional amplitude.

Before elevating the hypothesis, existing checkpoints must show what their learned conditional variation actually contributes to field accuracy. A finite, whole-network conditional-versus-collapsed comparison can establish dependence and error alignment; it cannot establish the benefit of a replacement trained architecture. If a future architecture is independently justified, a time-conditioned **generic SwiGLU** counterpart at the same total parameters, actual MACs and training budget is necessary. Gains over an unmodified SwiGLU alone would not isolate an ODE mechanism. Prediction target, output head, loss weighting, time sampling and solver must remain matched; changing them creates a separate question.

**Decision:** no new ODE architecture is supported for implementation by this review. Time-dependent feature allocation remains a structural hypothesis, with substantial counterevidence against easy transfer. None of the three papers validates reopening the stopped activation, clock, order, bias, coupling or width families.

Search scope: current primary architecture papers in diffusion/flow matching; screened dynamic FFNs, expert routing and trunk/head allocation. Read full text and the linked official code paths. Objective-only methods and unrelated sequence-model analogies were excluded. This is not an exhaustive proof that no relevant paper exists.
