# Time-conditioned weights: a bounded prior-work check

The generic idea of diffusion-timestep-dependent weights is established.
The current experiment tests a particular restriction and training setting;
its ODE notation does not establish computational novelty. This check was
made while the registered sampler ran. It changes neither the experiment
nor its acceptance rule, and does not select another model.

| Primary work | Mechanism and setting | Relationship to this experiment |
|---|---|---|
| Soboleva, Alanov, Kuznetsov and Sobolev, **T-LoRA**, AAAI 2026 ([method, arXiv v2](https://arxiv.org/html/2507.05964v2), [published paper](https://ojs.aaai.org/index.php/AAAI/article/download/37861/41823)) | Sections 3.1/3.3 use a timestep-dependent binary rank mask inside low-rank weight adapters, together with orthogonal initialization, for diffusion personalization. | Direct precedent for timestep-conditioned weights. Its mask selects adapter directions; our scalar changes one unconstrained dense matrix direction per local layer. Personalization results do not establish cold unconditional generation gains. |
| Cho, Ohana, Jacobsen, Jothi, Chen, Mao and Can, **TC-LoRA**, NeurIPS 2025 SpaVLE workshop ([method, arXiv v1](https://arxiv.org/html/2510.09561v1), [workshop paper](https://openreview.net/pdf/44a0fcff46fa08c14ed149da3175d8a4d6f406a7.pdf)) | Section 3, Eq. 1 generates two low-rank adapter factors from layer index, diffusion timestep and spatial condition. The pretrained backbone is frozen; Section 4 evaluates depth control. | Direct precedent for dynamically generated diffusion weights. It differs from our cold unconditional training and one-scalar local matrix family. This is a workshop paper, not a NeurIPS main-conference result. |

Our exact local family is `W_j(t) = I + s_j(t) P_j`. Its curve lies on an
affine line in matrix space, even when P is dense and has high matrix rank.
The identity shift and scalar curve leave an ordinary conditional-matrix
implementation. The stronger interpretation would have to come from an
empirically useful restriction, rather than the first use of time conditioning
or a new implementation of that same function.

The difference from the cited adapters is descriptive, not a novelty
certificate or a general expressivity theorem. Existing JiT AdaLN already
uses time. No paper's personalization or depth-control scores are transferred
to our CIFAR-10 FID comparison. The fixed quality gate and old stop rules
remain in force, regardless of this literature check.

Coverage is deliberately small: the two canonical records in
[prior-work-coverage.json](prior-work-coverage.json) have primary method
evidence and were both extracted. Retrieval and venue checks were performed
by the parent; one read-only agent extracted the frozen evidence without
additional retrieval. That extraction is same-family provisional. No broader
literature coverage or exhaustive novelty search is claimed. The local arXiv
API helper was unavailable, so primary arXiv HTML and official publication
pages were read through web tools. No PDFs or packages were installed.
