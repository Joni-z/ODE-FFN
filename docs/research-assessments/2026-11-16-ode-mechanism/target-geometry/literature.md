# What prior work actually supports

Scope: four identified primary papers, selected for the missing connection
between feature-depth dynamics and the FM prediction target. The two read-only
extraction shards covered all four; [coverage receipt](literature-coverage.json).
This is a targeted review of the listed sections, not an exhaustive literature
search or a replication of their proofs and experiments. No local PDF library
or optional Zotero/Obsidian connector was available.

| Paper | Venue/version | Method and observed result | Consequence for this project | Source |
|---|---|---|---|---|
| Wan, Wang, Mishne and Wang, *Elucidating Flow Matching ODE Dynamics via Data Geometry and Denoisers* | ICML2025 | Expresses ideal denoiser Jacobians as posterior covariance times alpha/beta²; analyzes the resulting FM dynamics. | Use the full clean-data predictor as the invariant object. The identity gives no condition on an internal FFN matrix. | [Proceedings](https://proceedings.mlr.press/v267/wan25e.html), AppendixC PropositionC.2; [earlier HTML](https://arxiv.org/html/2412.18730v2#S3.SS1), Proposition3.2 |
| Li and He, *Back to Basics: Let Denoising Generative Models Denoise* | Primary arXiv v2, January2026 | Separates prediction and loss spaces. Fixed velocity loss gives image MSE weighted by1/(1-t)². Table2 favors x-prediction strongly at ImageNet256/B16, while ImageNet64/B4 is close. | Equal conditional Bayes optima do not imply equal finite-network optimization or capacity requirements. Preserve the native x-prediction conversion. | [Section3.2 and Table2](https://arxiv.org/html/2511.13720v2) |
| Graikos, Jojic and Samaras, *On the spectral properties of generative denoiser Jacobians* | arXiv v1, September2026 | Studies complete denoiser Jacobians and perturbation-based regularization. Residual expansion can help in tested settings; excessive expansion hurts and CFG weakens gains. | Output response is more direct than P's spectrum. Measure skew and symmetric-part eigenvalues separately; learned Jacobians need not be symmetric/PSD. | [Section3.2,5.3; AppendixB.3/E](https://arxiv.org/html/2609.36210v1) |
| Derakhshani, Curvo, Burghouts, van de Meent and Snoek, *LiFT: Loop Flow Transformers* | arXiv v1, October2026 | Uses a shared depth-conditioned DiT core with intermediate velocity targets. The reviewed L/2 point improves FID at lower dense FLOPs; B/2 loses at the tested costs. | Internal computation can be supervised toward a task-defined target, but sharing/recurrence itself is prior work. More loops are not an inner-ODE resolution intervention. | [Section3.1 and depth/time discussion](https://arxiv.org/html/2610.05538v1); [prior local review](../../../literature-ode-architectures-2026-10-25/review.md) |

The useful distinction is between the **target's geometry** and the **network's
internal computation**. Wan et al. give an exact property of the Bayes denoiser.
JiT shows why its finite-capacity parameterization matters. Neither establishes
that inserting a matrix exponential into an FFN provides the required property.
Our derivative diagnostic tests that missing empirical connection at the complete
output rather than assuming that hidden P is a substitute.

The recent spectral paper makes this diagnostic timely but also limits a simple
spectral story. Its learned Jacobians are not exactly symmetric or PSD, so
singular values and eigenvalues cannot be interchanged. AppendixB.3 also reports
an improvement without a notable measured spectral change. Thus we preregister
full target-Jacobian error as primary and keep skew/PSD summaries secondary.

LiFT supplies a different bridge: explicitly supervising computation across depth
while the outer generative state/time are fixed. Its discrete recurrence is not
our prescribed hidden ODE. It makes a future task-defined internal target worth
considering, but does not justify claiming that a more accurate hidden solve must
improve samples. First complete the existing-checkpoint test; no architecture
or extra training is justified by these citations alone.
