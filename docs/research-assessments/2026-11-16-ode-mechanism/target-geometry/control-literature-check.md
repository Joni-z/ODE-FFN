# Literature check for the actual-law control

Checked 16 November 2026. This is a targeted novelty/scope check, not a survey.
The quantile/gauge construction is an explicit control for our native law;
velocity non-uniqueness and continuity-equation error analysis are prior ideas.

| Primary source | What was directly checked | Consequence for this project |
|---|---|---|
| Huang et al., *Improving Flow Matching by Aligning Flow Divergence*, ICML 2025 | The [official proceedings abstract](https://proceedings.mlr.press/v267/huang25ag.html) states a probability-path error PDE and a TV bound involving both CFM and divergence losses. | Distribution error cannot be inferred from pointwise velocity regression alone. A generic continuity-error identity or an added divergence loss would not be our novelty. The abstract does not justify any particular hidden-P constraint. |
| Praneeth, Shanker, Dubey and Seelamantula, *Velocity Adaptation for Flow-Matching Models*, ICML 2026 FoGen Workshop poster | [Official author publication metadata](https://openreview.net/profile?id=~Gouri_Shanker1) gives the workshop status. The indexed [primary PDF, Section 3](https://openreview.net/pdf?id=tdIwEN0fCi) distinguishes the continuity condition div(pu)=0 and describes a potential-network velocity adaptor. | Reweighting potential/residual velocity components is already explored. Unweighted div(u)=0 alone does not preserve an arbitrary density; retain the density weight in our derivation. No numerical performance or full-paper replication claim is made here. |
| *Flow Matching Generalizes Through Discretization Bias*, OpenReview submission | The indexed [primary submission abstract/introduction](https://openreview.net/pdf?id=JEn5B8JC5n) examines empirical velocity fields and beneficial outer discretization. Current acceptance status and complete experiments were not verified. | More accurate integration is not automatically a generative-quality objective. That setting is an empirical outer flow, whereas our hidden-refinement tests change a network's feature map; the paper does not explain our results by itself. |

The two OpenReview PDF opens returned browser-verification pages; the review
above is limited to the indexed primary excerpts and stated metadata. Those
papers are not counted as complete full-text reviews, and no unobserved
ablation, score or theorem condition is attributed to them.

## Current decision

The exact actual-law counterexample is useful because it falsifies a diagnostic
implication in this project. It does not establish an architectural contribution.
Keep the complete twelve-model calibration as an approximation test. Any
positive architectural claim must still come from matched trained controls and
a distribution-relevant effect; do not promote generic flow identities, a
potential-network adaptor or numerical accuracy itself as new ODE-FFN design.
