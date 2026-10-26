# Search notes

Date: 26 October 2026. This is a focused narrative review, not an exhaustive systematic review.

Initial queries via web search:
- "Tied Flow" transformer flow matching
- continuous depth diffusion transformer neural ODE architecture 2025 2026
- "JiT" "Back to Basics" denoising

Screening principle: retain sources that change internal model computation or clarify the distinction between hidden depth and sampling time. Sampling-only methods are contextual controls rather than architectural evidence.

Provider availability and all retained claims will be checked against primary pages/full text.

Provider check: parallel-cli and its API key are absent. Use the available web search/browsing tool instead; no installation or credential request is necessary. The schematic skill's external key is absent, but the available image-generation tool can produce the public conceptual diagram.

Additional queries:
- "Looped Diffusion Transformer" arxiv
- "Neural Residual Diffusion Models"
- "diffusion" "tied" "flow" transformer architecture weight sharing (arXiv/OpenReview)
- "ODE Transformer" "2025" "2026" architecture (arXiv/OpenReview/MLR)
- "Neural ODE Transformers: Analyzing Internal Dynamics" ICLR 2025 2026
- "Efficient Looped Transformers" diffusion "ELT"
- "Mean Flows for One-step Generative Modeling" official 2025
- "neural ordinary differential equations" transformer review survey 2025 2026
- "A mathematical perspective on Transformers" Geshkovski
- "Understanding and Improving Transformer From a Multi-Particle Dynamic System"
- "Neural Ordinary Differential Equations" 2018 NeurIPS
- "Flow Matching Guide and Code" 2025

Primary sources opened:
- LiFT (4 October 2026 preprint): https://arxiv.org/html/2610.05538v1
- Looped-DiT (30 September 2026 preprint): https://arxiv.org/html/2609.40305v1
- ELT (April 2026, latest v3 July 2026): https://arxiv.org/html/2604.09168v3
- DiffEqFormer (ICLR 2025): https://arxiv.org/html/2503.01329v1 and official ICLR proceedings.
- JiT (November 2025): https://arxiv.org/html/2511.13720v1
- Neural-RDM (NeurIPS 2024): official proceedings and PDF.
- MeanFlow (NeurIPS 2025): official proceedings and PDF.
- Neural ODE (NeurIPS 2018): official proceedings.

Do not identify an ambiguous historical name such as "Tied Flow" with LiFT, ELT, or another paper without a supplied title/link. Prior local reviews already record unsuccessful exact-name searches.

Access limits and corrections:
- The Neural-RDM NeurIPS PDF fetch failed. arXiv v2 HTML was read instead; the official proceedings confirm acceptance.
- A guessed Neural-RDM v3 URL returned 404. Its actual record has v1/v2; v2 was read.
- The OpenReview forum returned a browser challenge. The indexed camera-ready PDF and official ICLR 2025 proceedings confirm publication; no review-score claims are made.
- A guessed local report.md was absent; actual task_plan.md/notes.md establish the recorded recurrence result. No source report was edited.

Critical findings:
- LiFT Appendix C.2 explicitly disclaims any requirement that its core be an inner numerical ODE solver. The core sees current depth, not the previous depth or explicit interval length.
- Its B/2 dense baseline stays stronger; the favorable headline is not universal across scales.
- Looped-DiT Table 4 separates naive looping, supervision, and attention modulation. Report those factors rather than assigning the best result to looping alone.
- DiffEqFormer is genuinely non-autonomous: weights may vary with continuous depth. It is a language-model result, not a visual-generation validation.
- The professor's K-term polynomial has a frozen input-selected duration; it is not the full Taylor expansion of dh/ds=a(h)Ph.
- Same-dimensional exact ODE flows and clean-image regression maps have different invertibility requirements. An encoder/decoder matters.

The image-generation tool completed one figure, copied to figures/two-clocks.png and visually inspected. Its labels correctly distinguish outer sampling time and inner hidden depth. No data were plotted or invented.

Final checks: 11 catalog entries (10 topical/background sources and one procedural-library acknowledgment); 13 external links; all relative document/image links resolve. Dates/venues use actual records, not search-engine relative timestamps. The local Taylor expression was checked against the professor's recurrence with a scalar frozen gate and bias-free P. The review makes no novelty, solver-convergence, or image-generation claim for our current implementation.

Workflow acknowledgment for Scientific Agent Skills is included separately at the end of the review, rather than presented as an architectural reading recommendation.
