# Search and extraction notes

Search date: 2026-10-25, Asia/Shanghai.

## Sources and search method

The parallel-cli executable is not available. The built-in web tool is the fallback for discovery and primary-paper reading. Target records include arXiv, OpenReview, official conference proceedings, and ACL Anthology. Secondary discovery pages are not used to support technical claims.

## Initial queries

- "Tied-Flow"
- "Tied Flow" diffusion transformer
- "continuous depth" diffusion transformer ODE 2025 2026
- "Neural ODE" "diffusion transformer" architecture

Initial screening found many application papers and broad dynamical analogies without a relevant architecture/compute comparison. These are excluded from the shortlist. Existing project notes also did not identify an unambiguous primary paper named Tied Flow; no claim will rely on that name without verification.

## Interpretation boundary

The original ODELayer approximates a frozen-clock linear flow inside a hidden feature computation. It is not automatically a solver step of the image-generation ODE. A useful literature synthesis must track which state and which time a paper evolves.

## Follow-up searches

- "ODE Transformer" "Sequence Generation"
- "looped" "diffusion" transformer weight sharing
- "recurrent" "diffusion transformer" architecture weight sharing
- "weight-tied" "flow" generative transformer
- "Neural Flows: Efficient Alternative" NeurIPS2021
- "Universal Transformers" OpenReview2019
- site:openreview.net "diffusion" "recurrent" "2025" transformer
- "Gudovskiy" "length consistency"
- "ODE" "network length" "2026" diffusion
- "ODE_t" "ODE_l"
- "DeepFlow" "Shin" "2025" diffusion
- "LoopDiT" "MeanFlow"
- "DiffusionBlocks" ICLR2026
- "Thinking with Looped Flows"
- Publisher-domain checks for DeepFlow, Neural ODE and Universal Transformers.

## Primary-paper extraction

- LiFT, arXiv2610.05538v1,2026-10-04: read methods3.1–3.2, experimental budget/quality sections4.1–4.4, limitations6 and relevant AppendixB.4. The paper explicitly denies interpreting its discrete recurrence as solver refinement. Section4.4 provides the within-scale25-step/4-loop comparison; B/2 failure and unmeasured hardware latency are retained.
- Looped-DiT, arXiv2609.40305v1,2026-09-30: read method overview and3.2–3.3, Tables2 and4. The matched-compute58.1/59.1 comparison is a four-benchmark mean. The complete method combines recurrence, supervision and attention modulation; pure sharing is not credited with all gains.
- DeepFlow, arXiv2503.14494v2 and official ICCV2025 record: read method outline,4.3.1–4.3.4 andTables7–8, plus limitationsA.8. The34.4/33.0/28.1 progression is the uniform-time component ablation; it must not be mixed with the lognormal-time table.
- ODEt(ODEl), arXiv2506.21714v3 and WACV2026 publisher record: method4 and selected quantitative/ablation material. It is an existing direct precedent for nested time/depth, not the fresh shortlist's main recommendation.
- Neural Flows, official NeurIPS2021 paper: Sections2–2.2 and application definitions. Check identity/smooth-invertible map conditions and distinguish general non-autonomous flows from an autonomous semigroup.
- Neural ODE and Universal Transformers: primary metadata/abstract for foundational categories. ODE Transformer: official ACL record and abstract/introduction for RK-inspired computation; no image-generation transfer claim is taken from it.
- DiffusionBlocks: official ICLR2026 proceedings abstract/overview and paper introduction. Used only to distinguish block-wise denoising training from a hidden numerical ODE.

## Screened but not shortlisted

- LoopREG and Looped Diffusion Language Models: relevant recurrence neighbors, but not needed for the focused image-architecture shortlist.
- Thinking with Looped Flows: interesting stateful reasoning setting, distinct from fixed-input image denoiser depth; no reasoning scores are imported.
- Single-author continuous-depth language-model preprints, assorted application papers and code-only continuous-attention prototypes: insufficient task-relevant controlled evidence for recommendation here.
- Mamba/sequence-state analogies: different evolution axis; not used as direct evidence for ODE-FFN image-generation gains.

## Access and diagram

Some CVF PDF/page fetches returned403. Indexed publisher records verified venue/metadata; accessible arXiv full texts supplied methods. An OpenReview forum request hit a browser challenge; official proceedings supplied the paper. None is represented as a failed paper or as an unread full source that was fully verified.

The schematic skill's OpenRouter credential is unavailable. The available image-generation tool produced the conceptual figure instead, using only generic public architecture concepts. The first image used a decreasing-time label, which was corrected through image editing; the final figure uses x_(t+Delta_t). Original generated files are preserved outside the repository. The figure is an illustration, not paper evidence.

## Bibliographic and artifact checks

The skill's packaged verify_citations.py was attempted but exited before any verification because requests is not installed. This is not recorded as a citation PASS. Titles and arXiv identifiers were checked in the actual arXiv records; conference status was checked in the publisher/proceedings records. All factual performance examples point to the inspected primary section/table. No secondary summary supports a numerical claim.
