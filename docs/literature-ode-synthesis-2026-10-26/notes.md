# Search, extraction and verification notes

Date: 26 October 2026. This is a focused narrative synthesis, not an exhaustive systematic review or a performance leaderboard.

## Search scope

Prior project reviews supplied discovery leads and were preserved unchanged. Fresh searches covered arXiv, ICLR/OpenReview, PMLR, ACL Anthology, CVF and AMS primary records. The `parallel-cli` executable is absent; the available web tool was used without installing or changing an environment.

Queries:
- `"Neural ODE Transformers" DiffEqFormer`
- `"LiFT" "Loop Flow Transformers"`
- `"Looped Diffusion Transformer"`
- `continuous depth diffusion transformer neural ODE architecture 2025 2026`
- `"TiedFlow" transformer`
- `"diffusion" "continuous-depth" transformer architecture 2026`
- `"neural ODE" "feed-forward" transformer 2025 2026 architecture`
- `"A mathematical perspective on Transformers" "427" "2025"`

Select work with a concrete hidden-state field, a distinct numerical construction, directly relevant generative recurrence, or a mathematical review. Exclude unrelated physical forecasting, outer-sampler acceleration alone, and architecture-generation papers. CoDA is a useful efficiency lead but does not provide direct evidence for a JiT FFN; it was not added to the reading list. The TiedFlow query did not produce a verified primary record for inclusion. ODEt(ODEl) is known direct prior art already discussed by the user; it is not presented as a new recommendation.

## Evidence locations

| Work | Primary locations checked | Interpretation limit |
|---|---|---|
| DiffEqFormer | arXiv v1 Sections 4.1–4.2, Table 1; official ICLR 2025 record | Language experiments, not a JiT result |
| LiFT | arXiv v1 Sections 3.1, 4.4, Appendix C.2 | October 2026 preprint; supervision theory is not an inner-solver guarantee |
| Looped-DiT | arXiv v1 method, Tables 4–5 | September 2026 preprint; benchmark mean is not FID; changes include supervision and attention |
| IIET | official EMNLP record and full 15-page PDF, Equations 7–9, Table 2 on PDF page 6 | Finite corrections and historical aggregation; actual work exceeds one block call |
| BAMS perspective | arXiv v5 modeling and full-FFN discussion; AMS published metadata | Main mathematical analysis excludes the full FFN |
| Tai et al. | arXiv abstract revision history and v2 Sections 2–4 | Particular ReLU/relaxation formulation; no publication venue asserted |

Source URLs are in review.md and seven raw tool-response batches are archived in sources/web-results.json. The IIET table was checked against the extracted PDF and a screenshot was requested; no independent local visual-table verification is claimed. The BAMS DOI and AMS full PDF fetch returned HTTP 403 through the web tool. Publication metadata was checked through the indexed AMS primary record; technical statements were read in the accessible arXiv full text. Do not describe the restricted publisher PDF as fully accessed.

## Project and figure checks

Read the saved recurrence and realizable-teacher analysis reports before linking project conclusions. They support a negative result for their registered settings, not an impossibility theorem for ODE architectures. No new result is claimed from unfinished or uninspected studies.

The existing AI-generated two-clock schematic was viewed locally at full available resolution. The labels distinguish fixed sampling inputs from internal evolving features and contain no experimental data. Its existing path is referenced without modifying the historical file.

## Workflow provenance

The literature-review and planning-with-files skills guided the synthesis; the scientific-schematics guidance was used when inspecting the reused figure. The current primary metadata was checked for Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel and Aubrey M. Brueckner (2026), [Scientific Agent Skills](https://doi.org/10.48550/arXiv.2609.00065). This acknowledgment concerns the research workflow, not evidence for a model design.

The repository had existing modified tracked files before this task. They were preserved. Only this review folder was created; no training, model edit, submitted-report edit, commit or push occurred.
