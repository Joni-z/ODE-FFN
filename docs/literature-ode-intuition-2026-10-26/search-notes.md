# Search notes: intuitive ODE backbone constructions

Date: 26 October 2026. Status: completed focused narrative synthesis, not a systematic review or experimental evaluation.

The scope was fixed to computation inside a generative predictor. Initial search and citation chaining covered arXiv and official ICLR, NeurIPS, ECCV and TMLR/OpenReview records. Sampling-only methods, unrelated physical-system forecasting and weakly relevant title matches were excluded from the main reading list. Parallel CLI and its credential were unavailable; the available web-search tool was used. No installation or new credential was required.

## Search strings

- `"Neural ODE Transformers" DiffEqFormer ICLR 2025`
- `"LiFT" "diffusion" "depth" 2026`
- `"Looped-DiT" 2026`
- `"Neural-RDM" residual diffusion models`
- `site.arxiv.org transformer "continuous depth" "2026" diffusion`
- `site.arxiv.org transformer "exponential integrator" architecture`
- `site.openreview.net "diffusion" "neural ODE" architecture 2026`
- `site.arxiv.org "flow matching" "weight sharing" transformer architecture`
- `"The Missing U for Efficient Diffusion Models" ICLR`
- `"diffusion" "architecture" "Runge-Kutta" transformer 2025 2026`
- `"transformer" "exponential" "integrator" neural architecture 2025`
- `"continuous-depth" diffusion models neural ODE architecture 2025 2026 review`
- `"ELT: Elastic Looped Transformers for Visual Generation" ECCV 2026`
- `"ELT" "Elastic Looped" "ECCV" 2026`

## Primary full texts and selected evidence

- https://arxiv.org/html/2503.01329v1 — DiffEqFormer Sections 4 and 6; official ICLR 2025 proceedings confirm venue.
- https://arxiv.org/html/2609.40305v1 — Looped-DiT Sections 2 and 3.3, Table 4; dated 30 September 2026, preprint.
- https://arxiv.org/html/2610.05538v1 — LiFT Sections 3–4, Appendix C.2; dated 4 October 2026, preprint. Its solver disclaimer and small-scale negative result are retained.
- https://arxiv.org/html/2604.09168v3 — ELT Section 4 and Section 5.4. Current https://arxiv.org/abs/2604.09168 comments state acceptance to ECCV 2026; the ECCV 2026 Videos search result also lists the title. Direct full conference-list extraction returned an internal error; no presentation type or award is inferred. The older review's preprint-only status is not repeated here.
- https://arxiv.org/html/2312.10794v5 — mathematical perspective, Sections 2–3 and the explicit limitation on complete-FFN analysis. The abstract page supplies the BAMS 2025 journal reference.
- https://arxiv.org/html/2310.20092v4 — The Missing U, Section 4, CPU timing footnote. The official indexed OpenReview PDF states TMLR April 2024. Direct OpenReview access encountered a browser challenge. Earlier v2 has a different title and is not used for final method attribution.
- https://arxiv.org/html/2406.13215v2 — Neural-RDM Sections 2.1–2.3 and limitations, with official NeurIPS 2024 search/proceedings evidence.
- https://arxiv.org/html/2511.13720v2 — JiT Sections 3.1–3.2, latest revision 7 January 2026; no JiT code read.
- https://arxiv.org/html/2609.29768v2 — flow-matching TTS recurrence study, revised 21 October 2026. Useful supporting evidence that reuse layout depends on sampling budget; excluded from the short main list because speech transfer is indirect.
- https://arxiv.org/abs/2609.00065 — current procedural-library metadata verified; workflow acknowledgment only.

The search also retrieved control-conditioned language models, weather/PDE models and sampling integrators. These did not displace the more directly relevant original papers. Secondary pages served only as discovery leads; scientific descriptions were checked against primary papers.

## Scope and safeguards against overinterpretation

Methods and selected ablations were read, not every proof or supplementary table. No citation-count ranking, exhaustive retrieval count or independent paper replication is claimed. The original-layer interpretation is an explicit derivation from the user-supplied recurrence. The local-adjoint conclusion is attributed to its saved report, not to the literature.

The existing two-clock schematic was visually checked and referenced without regenerating or changing it. This follow-up adds two Markdown files only. No training, submitted-report edits, model-code changes, commit or push is performed for this review.
