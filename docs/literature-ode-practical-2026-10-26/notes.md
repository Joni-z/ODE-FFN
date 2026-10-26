# Search and extraction notes

Search date: 26 October 2026. Scope: internal feature evolution within one generative-model call, rather than outer sampling solvers.

## Starting materials
The completed project review at `../literature-ode-backbone-2026-10-26/review.md` supplies discovery leads and previously checked ablations. It is preserved unchanged. Existing experimental claims will only be used with explicit links to saved reports.

## Refreshed queries
- `"Neural ODE Transformers" "DiffEqFormer"` — arXiv, OpenReview, ICLR proceedings.
- `"Looped Diffusion Transformer" "LiFT" "Loop Flow"` — arXiv.
- `continuous depth diffusion transformer neural ODE architecture 2025 2026` — arXiv, OpenReview, CVF.
- `transformer neural ODE review mathematical perspective 2025 2026` — arXiv, AMS.

## Screening
Directly relevant: explicit hidden ODE Transformers, recurrent generative backbones, numerical Transformer blocks, mathematical treatments of attention/FFN/normalization. Exclude physical ODE forecasting, symbolic equation solvers, and image depth prediction. A 2026 control-conditioned language preprint is less directly relevant than established numerical Transformer work and visual-generation papers.

## Interpretation rules
Learned hidden depth and sampling time are different coordinates. An outer flow-matching objective does not imply that a hidden representation must follow the same flow. Repetition is a computational construction; numerical consistency requires a specified field, horizon, and step-size rule. Exact flow properties need regularity assumptions and do not automatically hold for a truncated Taylor map.

## Refreshed primary evidence
- DiffEqFormer: https://arxiv.org/html/2503.01329v1 Sections 4.1–4.2, Table 1; ICLR 2025 venue checked at https://proceedings.iclr.cc/paper_files/paper/2025/hash/82e3509393670d9d478c359b4ce6f950-Abstract-Conference.html . Explicit attention-plus-FFN field; continuous-depth weight parameterization. Small-model WikiText103 perplexities: GPT 22.87, shared 36.27, DiffEqFormer 22.84.
- LiFT: https://arxiv.org/html/2610.05538v1 Sections 3.1, 4.4, Appendix C.2. Submitted 4 October 2026; treat as preprint. At 25 outer steps/four loops, L/2 R10 has FID 13.16 at 7.397 TFLOPs, versus dense L/2 16.94 at 8.069 TFLOPs/50 outer steps. This is joint compute allocation, not an isolated solver-order ablation. B/2 remains weaker in tested comparisons. Appendix C.2 disclaims an inner-solver guarantee.
- Looped-DiT: https://arxiv.org/html/2609.40305v1 Sections 2, 3.3, Tables 4–5. Submitted 30 September 2026; treat as preprint. Same clean-image target at intermediate exits. Mean benchmark scores: baseline 55.2, naive loops 56.2, loops plus supervision 57.5, full 59.1. Compute-matched deep supervised baseline with the attention modification: 58.8. These are not FID.
- IIET: https://aclanthology.org/2025.emnlp-main.453/ and full PDF, Sections 3.1–3.2, Equations 7–9, Algorithm 1, Table 2. EMNLP 2025. Explicit predictor plus finite fixed-point corrections; implementation adds historical-state context. Comparable-FLOP average accuracies: IIET340M 48.9, shared Transformer 46.3, Transformer1.3B 49.0. Exact backward Euler stability is not automatically inherited by finite corrections.
- Geshkovski et al.: https://arxiv.org/html/2312.10794v5 Sections 2–3; published metadata checked against https://www.ams.org/journals/bull/2025-62-03/S0273-0979-2025-01863-7/S0273-0979-2025-01863-7.pdf . BAMS 62 (2025), 427–479; DOI 10.1090/bull/1863. Section 2.3 explicitly distinguishes mechanisms beyond the main analysis, including the full FFN.
- Tai et al.: https://arxiv.org/abs/2510.03989 and https://arxiv.org/html/2510.03989v2 . Originally submitted 5 October 2025; revised 12 April 2026. Current title is A Mathematical Explanation of Transformers. Sections 2–4 describe an integro-differential/variational formulation, operator splitting, normalization as a constrained projection, and ReLU feature steps. Sections 3.4 and 4 specify the discretized architecture and relaxation. This is an interpretation of a specified Transformer formulation, not an empirical improvement or a theorem for our SiLU/SwiGLU JiT block. No publication venue is claimed.
- Direct nested depth-ODE prior: WACV 2026 ODEt(ODEl), official CVF record checked during the refreshed search. Retained only as prior-art context; the user has already read it.

## Project evidence checked
Read `../research-recurrent-depth-2026-10-26/results/analysis-report.md` and `../research-realizable-teacher-2026-10-26/analysis-report.md`. No new experiment is run here. Existing recurrence/intermediate-supervision results do not select an original-ODELayer candidate. Teacher fitting does not distinguish insufficient expressivity from optimization difficulty on the SwiGLU teacher.

## Figure and procedural provenance
Reused `../literature-ode-model-2026-10-26/figures/two-clocks.png`, an existing AI-generated schematic. Visually inspected this turn: readable labels, fixed noisy input/time inside one predictor call, separate evolving hidden depth, no measured data. No new image or remote compute job was requested.

The current Scientific Agent Skills record at https://arxiv.org/abs/2609.00065 was checked: Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel, Aubrey M. Brueckner, 2026; latest listed revision 2 September 2026. Workflow acknowledgment only.

## Artifact checks
Six raw web-response batches are archived in `sources/web-results.json`. `verification.json` records five valid local links, balanced code fences, and successful DOI Handle checks for the BAMS and workflow references. The bundled verifier was attempted but `requests` is absent from the research venv; the fallback used Python's standard library, leaving the environment unchanged. Bibliographic metadata and scientific claims were checked against the primary pages separately from DOI accessibility.
