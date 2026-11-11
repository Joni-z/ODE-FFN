# Primary-source research notes

Date: 11 November 2026. Sources and recommendations are recorded separately.

## Project context

- The current root plan records an ISRU-D K8 image candidate, native SwiGLU
  control, seed 0 and a CelebA deployment on torch. No live job status was
  queried for this research task.
- Finalized reports are protected. This note is independent of Report2.

## Sources checked

1. [JiT, v2](https://arxiv.org/html/2511.13720v2), Sections 4–5,
   Appendix A Evaluation and B.5. Main image benchmarks: ImageNet 256 and
   512. FID uses 50,000 generated images against training-set statistics;
   conditional sampling uses exactly 50 images per class. IS uses the same
   samples; precision/recall appear in the appendix. Guidance and EMA are
   selected using a smaller development sample budget before the final run.
2. [JiT author PyTorch README, pinned](https://github.com/LTH14/JiT/blob/cbc743a2ada5e9762697da2c83f8c4f8379e8c17/README.md).
   Provides 256/512 recipes, a customized torch-fidelity evaluator and author
   reference statistics. The example 600-epoch recipes were tested on eight
   H200s; this is not a measured requirement or runtime estimate for our study.
3. [SiT, v2](https://arxiv.org/html/2401.08740v2), abstract, Table 1 and
   experimental setup. Conditional ImageNet 256/512 with DiT-style backbones;
   matched architecture/parameters/compute and FID versus training progress.
4. [REPA, v3](https://arxiv.org/html/2410.06940v3), Section 4.1 and
   Appendix E. Reports FID, sFID, IS, precision and recall on 50,000 samples;
   reuses ADM reference batches/evaluation. Representation probing is an
   additional diagnostic for REPA's own representation claim, not a universal
   requirement for every FFN paper.
5. [RAE, v1](https://arxiv.org/html/2510.11690v1), Appendices E, K and L.
   Conditional evaluation uses 50 images per class and ADM whole-ImageNet
   reference statistics. Appendix L separately evaluates unconditional
   ImageNet 256; one reported model uses AutoGuidance. This confirms that
   unconditional ImageNet is viable, but not comparable to conditional CFG
   results without retaining the task/guidance distinction.
6. [MS-pDiT, 5 November 2026 preprint](https://arxiv.org/html/2611.08802v1),
   Tables 1/4 and Appendix B.2. Current pixel-transformer architecture work
   still uses conditional ImageNet 256/512, 50K samples, FID/IS/sFID/precision/
   recall, with parameter and NFE information. Its main sampling recipe differs
   from JiT's; do not import its solver or guidance into one comparison arm.
7. [ADM evaluation README](https://github.com/openai/guided-diffusion/blob/main/evaluations/README.md).
   Author evaluator supports FID, sFID, IS, precision and recall. Provided
   reference batches contain whole-dataset statistics plus a 10K-image subset
   for precision/recall. Therefore metric reference sample counts should be
   recorded separately rather than assumed identical.
8. [Stein et al., 2023](https://arxiv.org/abs/2306.04675).
   Studies mismatches between image-generation metrics and human realism;
   investigates self-supervised feature extractors including DINOv2. Supports
   optional feature-space cross-checks, not substituting DINO distance for
   standard Inception FID while retaining the same metric name.

GitHub files were read through the connected GitHub tool. Papers were read
through official arXiv records/HTML. Search aggregators and blogs were not
used as evidence. This is an evaluation-protocol review, not a novelty audit.

## Project-specific checks

- [Current image candidate](../2026-11-11-image-candidate.md): K8 freezes
  its input-dependent diagonal within each Taylor expansion. It is distinct
  from the old K1 ISRU-D positive result.
- [Current CelebA protocol](../../celeba-k8-20261111/protocol.md): original
  aligned CelebA, 64 pixels, official training split, unconditional, seed 0,
  author-derived JiT-B body, native SwiGLU comparison. This is a declared
  project port, not an author-published CelebA baseline.
- K8: two square matrices, 2d^2 matrix parameters, 16d^2 forward dense MACs
  per token. Native ratio-4 JiT SwiGLU: about 8d^2 for each. Thus quarter
  FFN matrix parameters and twice FFN forward dense MACs; whole-model
  training cost and latency require measurements.
- Current active image code uses the author's torch-fidelity path. ImageNet
  uses author-provided resolution-specific stats; CelebA uses deterministic
  training-split crops. A change of benchmark does not authorize changing
  the active reference or submitting another job.

## Synthesis

ImageNet 256 is the recommended final anchor. CelebA 64 is useful for the
current unconditional screen and ablation. ImageNet 512 adds resolution
evidence; another native backbone adds portability evidence. Neither is an
automatic requirement to run every possible dataset. CIFAR-10 is optional
debugging, not the primary evidence for a modern high-dimensional pixel-FFN
claim. Synthetic cases retain analytic mechanism tests.

For this project, combine final distribution quality and coverage with
training/sampling cost. Extra internal operator applications do not appear
as extra outer sampler NFE, so NFE alone is insufficient for an efficiency
claim. Conditional labels prescribed to be balanced do not establish that
the model discovered balanced modes.
