# Evaluation plan for an ODE-inspired FFN paper

Zhizhe Zhang · 11 November 2026 · Research recommendation, not a launch plan

## Recommended endpoint

Use **ImageNet 256** as the main image-generation benchmark. Retain the current
**unconditional CelebA 64** comparison for screening and ablations, and the
analytic synthetic tasks for mechanism evidence. Add **ImageNet 512** or a
second native generative-transformer backbone after the main comparison earns
that expense. The primary metrics should be **FID50k, precision/recall and
quality versus measured compute**, with IS and sFID for table compatibility.

The paper must show which benefit is established: improved distributional
quality at a fixed budget, reduced cost to reach a fixed quality, or a verified
mechanism explaining when the structure helps. Dataset size alone does not
establish any of these claims.

## What comparable papers actually evaluate

| Primary source | Relevant benchmark | Evaluation practice |
|---|---|---|
| [JiT, v2](https://arxiv.org/html/2511.13720v2) | Pixel-space ImageNet 256/512 | FID50k and IS; additional precision/recall; matched architectural ablations |
| [SiT, v2](https://arxiv.org/html/2401.08740v2) | Conditional ImageNet 256/512 | FID50k, progress curves, equal architecture/parameters/compute comparisons |
| [REPA, v3](https://arxiv.org/html/2410.06940v3) | ImageNet generation | FID, sFID, IS, precision/recall on 50K samples; ADM reference/evaluator |
| [RAE, v1](https://arxiv.org/html/2510.11690v1) | ImageNet 256/512; separate unconditional ImageNet 256 | Guidance/task-specific tables; 50K conditional samples with 50 per class |
| [MS-pDiT, 5 Nov 2026](https://arxiv.org/html/2611.08802v1) | Pixel-space conditional ImageNet 256/512 | FID/IS/sFID/precision/recall, parameters and NFE; current architecture benchmark practice |

These papers motivate the recommendation. They are not equivalent FFN controls:
changing latent representation, loss or backbone changes the experiment.

## Dataset priorities

| Priority | Dataset/protocol | Question it answers |
|---|---|---|
| Current | Original aligned CelebA, 64, unconditional | Does the selected FFN transfer to real images within the author's actual transformer body? Supports affordable order/ordinary controls. |
| Main paper anchor | Full ImageNet-1k, 256 | Does it help on a diverse, standard, high-dimensional image-generation task? |
| Next evidence, conditional on the main result | ImageNet 512 and/or a second backbone such as native SiT | Does the improvement survive resolution or architecture/representation changes? |
| Mechanism | Existing mixtures, rotated product distribution and embedded manifolds | Where can field error, mode probabilities and feature responses be measured against analytic targets? |

CelebA is the original aligned dataset, **not CelebA-HQ**. A future FFHQ or
CelebA-HQ experiment would be another dataset/protocol, not a new name for
the current data. CIFAR-10 can be a quick implementation screen if needed;
it is not necessary to add it simply to increase the benchmark count.

For the ImageNet experiment there are two defensible options:

- **Match JiT's published main task:** class-conditioned training with separate
  no-guidance and fixed-guidance readouts. Use exactly 50 samples per class
  for final conditional evaluation, following JiT. This is the recommended
  final-paper option for direct main-table comparison.
- **Retain the user's unconditional objective:** all labels are null and
  sampling has no CFG. Report it as unconditional ImageNet; RAE Appendix L
  provides a relevant precedent, including a separately identified AutoGuidance
  setting. Compare our arms under the same unconditional/no-guidance recipe.

The current experiment remains unconditional. This note recommends options;
it does not change that decision or apply a conditional published FID as its
qualification target.

## Metrics and their roles

| Metric/readout | Role | Required interpretation |
|---|---|---|
| FID50k, lower | Primary generation-quality comparison | 50,000 generated images; pin reference stats, Inception implementation and preprocessing. |
| Precision and recall, higher | Fidelity and coverage estimates | Report both; recall is particularly relevant to our observed mode-allocation issue. These are feature-manifold estimates, not exact mode probabilities. |
| IS, higher | Compatibility with ImageNet generation tables | Supplementary; it does not compare generated images directly to the real-data distribution. |
| sFID, lower | Spatial feature-distribution cross-check | Useful alongside REPA/MS-pDiT-style tables; keep the implementation fixed. |
| Optional FD-DINOv2, lower | Cross-check conclusions in another feature space | Identify encoder/checkpoint, preprocessing and reference. It is not numerically interchangeable with Inception FID. |
| FID versus accumulated training FLOPs and GPU-hours | Convergence/efficiency evidence | Plot checkpoints. Record hardware, precision, batch and actual full-step work. State what is included in FLOPs. |
| FID versus outer sampler NFE and sampling seconds | Sampling tradeoff | Count guidance work; report actual time, not only nominal integration steps. |
| Parameters, peak training memory and images/second | Engineering cost | Report whole-model and FFN quantities separately, on the same device/configuration. |

The [ADM evaluator](https://github.com/openai/guided-diffusion/blob/main/evaluations/README.md)
implements the common quality/coverage suite. Its reference batches include
whole-dataset statistics plus a smaller precision/recall image subset; record
both. The [JiT author implementation](https://github.com/LTH14/JiT/blob/cbc743a2ada5e9762697da2c83f8c4f8379e8c17/README.md)
uses its own torch-fidelity fork and reference statistics for FID/IS. Preserve
the current author path and validate any additional evaluator on identical
saved samples before combining results. Do not silently switch FID libraries.

[Stein et al.](https://arxiv.org/abs/2306.04675) supports checking representation
dependence rather than treating FID as a complete measure of human realism.
An extra feature metric is useful, but collecting many correlated metrics
cannot replace a clear hypothesis and strong ordinary controls.

## Fair FFN comparison

Keep the author backbone, attention, objective, time/noise law, data order,
optimizer recipe, effective batch, training budget, EMA and sampler controlled.
Preserve the native FFN of each backbone: JiT's SwiGLU is the practical main
reference; a different backbone need not have native SwiGLU. Tune any proposed
method-specific settings with a comparable, recorded budget for ordinary arms.

For the current K8 candidate, the staged scientific controls are:

1. Native JiT SwiGLU versus selected ISRU-D K8: practical replacement result.
2. ISRU-D K1: shared generator/diagonal structure with less internal work.
3. Parameter-matched ordinary SwiGLU: capacity control.
4. Free ISRU: whether the nonlinear feature response can explain the result.
5. A forward-MAC-matched ordinary control or a matched total-training-compute
   comparison when an efficiency claim is made. Real full-step cost must still
   be measured, because matching dense forward MACs does not match every cost.

This is a paper-level evidence ladder, **not authorization to run all arms**.
The existing CelebA two-arm allocation remains its own bounded experiment.

At width 768, the selected FFN contains two square matrices, or 1,179,648
matrix parameters. K8 applies sixteen square matrix products per token across
its two internal layers, totaling 9,437,184 forward dense MACs. Native JiT
SwiGLU with intermediate width 2048 has 4,718,592 matrix parameters and dense
MACs. Thus **quarter FFN matrix parameters does not imply lower compute:
K8 has twice this native FFN's dense forward work**. Pointwise operations,
attention, conditioning, backward, optimizer and hardware overhead are outside
those arithmetic counts.

The eight internal terms do not count as eight outer sampling steps. A sample
generated using the same NFE can still cost more because each forward pass
does more internal work. Conversely, fewer weights may reduce memory without
improving latency. Use the measured quality/cost curves to decide.

## Mechanism evidence worth retaining

Use held-out samples from analytic synthetic distributions to measure population field MSE, SW2,
and signed mode-mass errors separately. Preserve common source-noise banks and
solver-resolution checks. On images, a paired flow-regression loss is not an
analytic population-velocity error; report it as held-out objective loss,
optionally binned by global time.

At preselected checkpoints, relate P's spectrum to the actual Pq(h)/polynomial
response and the full feature-map Jacobian, including gate derivatives. For
K8, measure truncation error against exponential action on a small, explicitly
specified token fixture. These diagnostics can test a claimed mechanism;
spectral plots alone do not establish an improvement in generation.

In conditional ImageNet evaluation, requesting 50 images per class imposes
balanced conditioning. It cannot demonstrate spontaneous between-class mode
balance. Use coverage/within-class variation for that task, and retain explicit
mode-mass measurements where assignments are meaningful in synthetic or
separately defined unconditional experiments.

## Execution and publication gates

- Preserve training seed 0. Reusing source-noise banks across fixed models
  improves comparison; repeated sample banks measure sampling variation,
  not independent-training variation.
- Use smaller fixed sampling sets for development, then lock the evaluation
  protocol and checkpoint rule before final FID50k. FID at another sample
  count should be named accordingly and not substituted into the main table.
- Share initial noises and, for conditional tasks, labels across methods.
  Report raw and EMA checkpoint identity, solver, time grid, actual NFE,
  guidance scale/interval and image/reference preprocessing. If guidance/EMA
  is tuned, use separate development samples and equal search budgets.
- Retain the user's FID50k < 50 **initial interpretation gate**. Passing it is
  not a publication target: the native baseline must also reach credible
  same-task, same-size, same-budget performance. An immature baseline cannot
  establish a useful architectural improvement.
- A positive ImageNet result should survive the relevant ordinary controls
  and cost accounting before adding 512 resolution or a second backbone.
  Record negative results rather than opening an unbounded rescue sweep.

No new run is submitted by this research. Exact ImageNet budgets require
dataset availability and measured throughput under the chosen recipe.
