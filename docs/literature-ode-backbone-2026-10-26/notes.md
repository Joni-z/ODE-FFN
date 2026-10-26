# Search and evidence notes

Date: 26 October 2026. Research documents are in English; conversational explanation will be in Chinese.

## Question and inclusion criteria
What architectural changes make one prediction call a dynamical computation, and which have credible empirical support relevant to a JiT-style generator? Include explicit continuous-depth models, numerical Transformer designs, and recurrent visual generators with relevant ablations. Exclude sampling-only solvers, training-free caching, and physical time-series forecasting from the main recommendations.

## Initial queries
- `"ODE" "Transformer" continuous depth diffusion 2026`
- `"Looped Diffusion Transformer" "LiFT"`
- `"Neural ODE Transformers: Analyzing Internal Dynamics and Adaptive Fine-tuning"`

## Prior project materials
Existing reviews in literature-ode-intuition-2026-10-26, literature-ode-model-2026-10-26, and literature-ode-design-2026-10-26 were read as discovery leads. Claims will be checked against original papers before reuse. The pre-existing parity experiment is outside this request.

## Workflow
The literature-review and planning-with-files skills guide source verification and persistence. Parallel CLI is unavailable; native web access is the fallback. An existing AI-generated schematic will be reused, with provenance retained. No experimental data are represented in that diagram.

## Additional queries
- `"PCformer" predictor corrector transformer numerical NeurIPS 2024`
- `"neural ODE" diffusion architecture "2026" FFN`
- `"continuous depth" transformer "2025" "2026" review`
- `"ELT: Elastic Looped Transformers for Visual Generation" ECCV`
- `"flow matching" "architecture" "continuous-depth"` (primary academic domains)
- `"diffusion transformer" "neural ODE" architecture 2026` (primary academic domains)
- `"ODEt(ODEl)" WACV 2026` (CVF/arXiv)
- `"Deeply Supervised Flow-Based Generative Models"` (CVF/arXiv)

## Verified sources and extraction locations
- DiffEqFormer: https://arxiv.org/html/2503.01329v1 Sections 4.1–4.2, 6.1, Table 1. Official ICLR 2025 proceedings confirm the title and venue: https://proceedings.iclr.cc/paper_files/paper/2025/hash/82e3509393670d9d478c359b4ce6f950-Abstract-Conference.html . Explicit attention-plus-FFN hidden ODE; weights depend on continuous depth. WikiText103 small-model perplexities 22.87 / 36.27 / 22.84 for GPT/shared/model.
- LiFT: https://arxiv.org/html/2610.05538v1 Sections 3.1–3.2, 4.2–4.4, Appendix C.2. October 4, 2026 preprint. Distinct intermediate targets, detached initial anchor. Appendix C.2 explicitly disclaims an inner-solver requirement or integration-error guarantee. L/2 R10 FID 20.30 at two loops to 10.95 at eight, at fixed 50 outer steps; this comparison spends more inference compute. B/2 remains worse than dense at tested costs.
- Looped-DiT: https://arxiv.org/html/2609.40305v1 Sections 2, 3.2–3.3, Tables 2/4/5. September 30, 2026 preprint. Same clean-image target at intermediate exits, plus self-modulating attention. Four-benchmark means 55.2 baseline, 56.2 naive loops, 57.5 loops plus supervision, 59.1 full method. Matched deep supervised non-loop model 58.1; XSA counterpart 58.8. Do not call this an FID table.
- IIET: https://aclanthology.org/2025.emnlp-main.453/ and authoritative https://aclanthology.org/2025.emnlp-main.453.pdf . EMNLP 2025. Equations 7–9 explain fixed-point correction; Algorithm 1 and Section 3.2 add historical context. Table 2: 340M IIET 48.9, shared Transformer 46.3, 1.3B Transformer 49.0 at comparable FLOPs. Abstract-only speed claims are not substituted for the comparable-cost table. Finite correction is not automatically an exactly solved A-stable implicit step.
- ELT: https://arxiv.org/html/2604.09168v3 Sections 4, 5.4 and Figure 8. Current https://arxiv.org/abs/2604.09168 comments say accepted to ECCV 2026; official ECCV video listing also contains the title. Intra-loop self-distillation trains useful intermediate exits. Covers masked and diffusion generators; headline FID 2.0 must not be attributed to a DiT result without the corresponding table.
- DeepFlow: https://arxiv.org/html/2503.14494 (v2), Section 4.3.1 / Table 7a. Official ICCV 2025 record: https://openaccess.thecvf.com/content/ICCV2025/html/Shin_Deeply_Supervised_Flow-Based_Generative_Models_ICCV_2025_paper.html . Uniform-time baseline/deep-supervision FID 34.4/33.0; full 28.1 includes other components.
- Mathematical perspective: https://arxiv.org/html/2312.10794v5 Sections 2–3. Current abstract page records BAMS 62 (2025), 427–479. Section 2.3 explicitly excludes complete practical mechanisms from the main analysis; Section 2.3.2 identifies the full FFN dynamics as beyond its methods.
- Direct flow-map precedent: https://proceedings.neurips.cc/paper/2021/hash/b21f9f98829dea9a48fd8aaddc1f159d-Abstract.html . NeurIPS 2021, Neural Flows. Definition/background only; sequence/density experiments are not image-generation evidence.
- Splitting precedent: https://arxiv.org/html/1906.02762 , Section 3.3. Macaron is historical context, not a recent recommendation.
- Direct diffusion depth-ODE precedent: https://openaccess.thecvf.com/content/WACV2026/html/Gudovskiy_ODEtODEl_Shortcutting_the_Time_and_the_Length_in_Diffusion_and_WACV_2026_paper.html . WACV 2026 record confirms that nested sampling/depth ODEs are already prior art. No claim that this is the newest work.
- JiT: https://arxiv.org/html/2511.13720v2 Sections 3.1–3.2 and 4.2. Only the paper was read. Direct clean-image prediction is compatible with a hidden ODE but does not imply one.

## Exclusions and limits
HyCa/feature caching concerns evolution across sampling calls; it is excluded from backbone-design recommendations. Broad queries also found physical forecasting, control-conditioned language models, and unrelated PCFormer title matches; these did not displace the direct architectural sources. No exhaustive hit count, citation-based ranking, independent paper replication, or comprehensive proof audit is claimed.

## Workflow citation
Metadata checked at https://arxiv.org/abs/2609.00065 on this date: Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel, Aubrey M. Brueckner; 2026; current v2 dated September 2. This procedural-library reference belongs to workflow provenance, not evidence for ODE architectures.
