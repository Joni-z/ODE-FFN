# Search notes

Date: 2026-10-26, Asia/Shanghai.

The prior curated review (2026-10-25) supplies leads; claims are checked again in primary records and full text. Built-in web search is used as the available fallback; prior notes document that parallel-cli is unavailable. This is a scoped narrative review, with no PRISMA completeness claim.

Initial queries:
- "diffusion transformer" "neural ODE" architecture continuous depth 2025 2026
- "LiFT" "Looped" diffusion transformer 2026
- "continuous depth" transformer flow matching architecture 2025 2026
- "DeepFlow" deep diffusion models 2025

Primary full texts reopened: LiFT (2610.05538), Looped-DiT (2609.40305), DeepFlow (2503.14494), ODEt(ODEl) (2506.21714).

Follow-up queries target continuous-depth architectures, flow-matching architecture, weight sharing, and exponential FFNs in primary arXiv/OpenReview records. Secondary summaries are discovery-only.

## Primary evidence and extraction

- LiFT, arXiv:2610.05538v1 (4 October 2026): sections 3.1–3.2, 4.4, 6, B.4, C.2. Its depth-conditioned recurrence does not receive interval length explicitly and is not required to implement an inner ODE solver. Intermediate targets form an anchor-to-target path; this differs from requiring the final target at every depth. The 13.16/7.397 versus 16.94/8.069 comparison is the within-scale L/2 operating point, not a general win at every scale. B/2 failure retained.
- Looped-DiT, arXiv:2609.40305v1 (30 September 2026): sections 2, 3.2–3.3, tables 2/4/5. The important counterfactual is 57.5 for loops plus deep supervision versus 58.1 for a cost-matched dense deeply supervised baseline. Full looped XSA reaches 59.1, while the dense deeply supervised XSA variant reaches 58.8. These are the four reported benchmark mean scores. The full-method 59.1 versus 58.1 comparison alone overstates attribution to looping.
- DeepFlow, arXiv:2503.14494v2; ICCV 2025 record verified through indexed CVF publisher page. Sections 3.2 and 4.3.1/table 7: uniform-time component progression. No mixing with lognormal-time or larger-model results.
- IIET, official EMNLP 2025 record and 15-page PDF, DOI 10.18653/v1/2025.emnlp-main.453. Sections 3.1–3.5, tables 1–2 inspected. Table 2 is on PDF page 6 (zero-based 5): near-equal FLOPs, the same 16B training tokens; 340M IIET 48.9, plain weight-sharing 46.3, 1.3B ordinary Transformer 49.0. This is a parameter-efficiency result on language tasks. The much larger same-parameter gain in table 1 also incurs substantially more computation; not presented as a free accuracy gain.
- ODEt(ODEl), arXiv:2506.21714v3; WACV 2026 publisher metadata verified. Section 4 and length-consistency objective inspected. Known direct prior for two nested coordinates, not presented as a new frontier recommendation.
- Neural ODE: official NeurIPS 2018 abstract consulted for the continuous-depth definition.
- Neural Flows: official NeurIPS 2021 record/paper for direct solution-map design and the need for flow-map conditions.
- ODE Transformer: official ACL 2022 record, DOI 10.18653/v1/2022.acl-long.571. Used as numerical-method architecture precedent, not image-generation evidence.

## Screening decisions

Searches covered primary records on arXiv, OpenReview, NeurIPS proceedings, ACL Anthology and CVF. This is a curated narrative review rather than a complete systematic search. Application-only flow-matching papers and outer sampler-only methods are excluded because they do not answer the architecture question. SLT (arXiv:2601.11630) was discovered but not shortlisted: teacher-feature distillation and teacher noise selection address a different setting. Secondary summaries were used only for discovery. No ambiguous paper name such as “Tied Flow” supports a claim.

Additional recorded queries:
- site:arxiv.org "diffusion" "continuous-depth" transformer 2026 architecture
- site:arxiv.org "flow matching" "ODE" "architecture" "2026" transformer
- site:openreview.net "diffusion" "weight sharing" "2026"
- site:arxiv.org "diffusion" "exponential" "feed-forward" architecture
- site:aclanthology.org "ODE Transformer" Runge Kutta 2022
- site:proceedings.neurips.cc "Neural Flows" "Efficient Alternative"
- site:arxiv.org diffusion transformer "ODE" "architecture" "October" "2026" -FlowLOB -Channel -Knowledge
- site:openaccess.thecvf.com "Deeply Supervised Flow-Based Generative Models" "ICCV 2025"
- site:openaccess.thecvf.com "ODEt(ODEl)" "WACV 2026"

## Access and verification limits

Direct CVF fetches returned 403. Indexed official CVF records verified publication metadata, and arXiv full texts provided the methods and tables. The ACL PDF was accessible as parsed text; the screenshot tool failed with a content-type error (and page 7 was initially requested rather than the relevant table on page 6). No successful screenshot is claimed. Relevant table entries were checked in the parsed PDF. Citations were verified manually against primary records; the unavailable packaged verifier is not reported as a PASS. No fresh experiment was launched, modified, or assessed for this review.

The existing AI-generated schematic was reopened and visually checked. It is reused by relative link, preserving the earlier review's image and provenance. No new image-generation API call was needed.

## Architecture explanation and project-evidence recheck

Rechecked on 2026-10-26 for the question “直观上怎么把一个模型 ODE 化”. This pass distinguishes learning a hidden derivative, training a repeated correction, organizing numerical substeps, and directly parameterizing a solution map. It adds no training or sampler run. The complete normalized-geometry analyzer from the preceding work was observed to have terminated; its files were not modified by this review.

Queries in this pass, grouped by discovery round:

1. `"diffusion" "continuous depth" transformer 2025 2026 ODE architecture`; `"flow matching" "loop" transformer 2026 LiFT`; `"Tied Flow" transformer flow matching`; `"ODE" "Transformer" "implicit" 2025 2026`.
2. `"ODEt(ODEl)" "2026"`; `"IIET" "Efficient Numerical Transformer"`; `"Looped Diffusion Transformer"`; `"flow matching" "architecture" "semigroup" neural 2025 2026`.
3. `site:arxiv.org "diffusion transformer" "ODE" "architecture" "2026" -Fractal -FlowLOB -Channel -soft`; `site:arxiv.org "neural" "flow maps" "semigroup" "2025" "2026"`; `site:openreview.net "transformer" "exponential" "2026" ODE`; `site:aclanthology.org "Transformer" "Predictor-Corrector" "2024"`.
4. `site:openaccess.thecvf.com "Deeply Supervised Flow-Based Generative Models"`; `site:openaccess.thecvf.com "ODEt(ODEl)"`; `site:arxiv.org transformer differential equation architecture exponential flow 2026 "feed"`; `site:arxiv.org "flow matching" "architecture" "inductive bias" 2025 2026`.

Many architecture-keyword hits were application-only, theoretical analogies, or unrelated proceedings indexes; they were not promoted into recommendations. The shortlist remains selective, not an exhaustive review or a claim that no other relevant work exists. No verifiable “Tied Flow” source was established in these searches.

Full-text checks:

- LiFT: methods 3.1–3.2, the within-scale result in 4.4, B/2 failure in 4.3, and Appendix C.1–C.3. The straight reference path has a fixed-anchor conditional-regression interpretation and a fixed-endpoint Euclidean minimum-action interpretation. These do not prove the shared recurrent core is an ODE solver, that every finite predictor realizes the population optimum, or that test-time recurrence improves monotonically. Keep the boundary explicit. No extra theory claim is added to the review's reading recommendation.
- Looped-DiT: tables 2, 4 and 5 were compared together. At effective depth 32 with deep supervision, ordinary layers score 58.1 without XSA and 58.8 with XSA; loops score 57.5 without XSA and 59.1 with it. The metric is the four-benchmark B/32 mean, not FID. This prevents attributing the complete-method gain to shared depth alone.
- IIET: the official ACL paper's equations 8–9, sections 3.2–3.5, and table 2. The exact implicit-Euler equation differs from its finite fixed-point approximation. The table has 48.9 / 46.3 / 49.0 for IIET / simple weight sharing / a larger ordinary model at comparable FLOPs. The browser screenshot returned no inspectable image content; verification used the parsed official PDF, and no new successful screenshot inspection is claimed.
- Neural Flows: section 2 conditions i–ii and the smooth invertible map construction. Its general time-dependent flows do not require an autonomous one-parameter semigroup. The professor's input-dependent duration cannot be assumed to preserve the fixed-time linear-flow properties.
- [Lu et al., 2019, Macaron primary paper](https://arxiv.org/html/1906.02762): sections 3.2–3.3 for the attention/FFN splitting example. Only used to explain a design mechanism; no frontier status or unverified venue is assigned.
- Neural ODE's NeurIPS record, and the official CVF records for DeepFlow/ODEt(ODEl), were reopened. LiFT and Looped-DiT remain labeled preprints. The skill-library attribution below was also rechecked against the current arXiv metadata.

Local evidence read: `research-recurrent-depth-2026-10-26/results/analysis-report.md` and `research-time-generator-2026-10-26/results/analysis-report.md`. The review now explicitly acknowledges the already-completed recurrence/deep-supervision screen and the negative time-dependent-generator result. It does not propose rerunning these settings on the strength of the literature. The reported best-family error comparison is descriptive across the tested arms; it is not represented as a single-factor paired comparison or transferred to JiT image generation.

The two-clocks image was reopened in this pass. Labels, increasing time direction, distinction between the three mechanisms, and the fixed outer input were visually checked. Its illustrative icons remain explicitly non-empirical. The previously saved diagram satisfies this review's conceptual figure requirement without another generation call.

## Workflow attribution

This review used the local literature-review and planning-with-files skills, and reused a diagram prepared under the scientific-schematics workflow. This attribution concerns research procedure, not evidence for an architecture. The skill-library citation was checked against the current arXiv record on 2026-10-26: Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel, and Aubrey M. Brueckner (2026), [Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://doi.org/10.48550/arXiv.2609.00065).

## Current-request source recheck

The current pass reused this existing review, rather than create another copy. Primary full texts for LiFT, Looped-DiT, DeepFlow, IIET, Macaron and JiT were reopened. The NeurIPS Neural ODE and Neural Flows records were also reopened. LiFT's 13.16/7.397 versus 16.94/8.069 operating point and B/2 failure were checked; Looped-DiT Tables 4–5 and DeepFlow Table 7 were checked together with their component descriptions. IIET equations 8–9 and the comparable-FLOP Table 2 were read from the official ACL PDF. CVF direct fetch again returned 403; the indexed official ICCV record verified DeepFlow's venue, with arXiv used for full text. The diagram was reopened with view_image and inspected.

Representative additional discovery strings: `"Tied" "Flow" transformer diffusion tied flow paper`; `"LiFT: Loop Flow Transformers"`; `"Looped Diffusion Transformer"`; `"Back to Basics" "JiT" diffusion paper`; `site:arxiv.org "diffusion" "continuous-depth" architecture "2026"`; `site:openreview.net "flow matching" "architecture" "ODE" 2026`; `site:arxiv.org "diffusion transformer" "exponential" architecture`.

No new shortlist entry was supported by the broad follow-up queries; application-only, caching and outer-solver hits were excluded. No exhaustive-search claim is made. The local recurrent-depth and time-generator reports were reread to avoid proposing already-failed settings. No experiment was launched or altered. parallel-cli is still absent; built-in browsing was the available fallback. The skill-library attribution was rechecked against the arXiv metadata. The synthesis now explicitly distinguishes global continuous depth from the local FFN parameterization and identifies the unproven link to JiT's objective.
