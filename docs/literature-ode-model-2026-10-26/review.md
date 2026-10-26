# Making the predictor a dynamical system

Focused literature review, 26 October 2026. Scope: the computation **inside** one flow/diffusion predictor, with emphasis on recent visual-generation architectures. Primary methods and selected experiments were read; proofs were not independently reproduced. This is a narrative review, not an exhaustive systematic survey.

The useful architectural question is: **what state evolves, under what update rule, and what makes successive updates useful?** The fact that generation uses an ODE does not determine the architecture of the network that supplies its predictions.

## Two different clocks

![Sampling time and internal depth](figures/two-clocks.png)

The outer clock `t` indexes the noisy image and the sampling trajectory. Within one predictor call, that image and `t` remain fixed. A second coordinate `s` can index computation over hidden features. A JiT-compatible construction is:

```text
h(0) = Encoder(noisy_image, t)
dh/ds = F_theta(h, s; noisy_image, t),  s in [0, 1]
predicted_clean_image = Decoder(h(1), t)
```

The sampler converts the prediction into the quantity needed to advance `t`. JiT distinguishes direct clean-image prediction from the velocity-space training loss; it does not require the backbone to solve a hidden ODE. See Li and He, [Back to Basics: Let Denoising Generative Models Denoise](https://arxiv.org/abs/2511.13720), Sections 3–4. The hidden states need not be images or intermediate denoising states.

## Architectural interpretations and evidence

| Interpretation | Concrete commitment | What must be checked |
|---|---|---|
| Local dynamical FFN | A block implements an approximate finite-time transformation, as in our polynomial ODELayer | Meaning of its clock; polynomial approximation; complete block Jacobian |
| Continuous hidden depth | A vector field and integration horizon define the forward function | Solver accuracy, runtime, sensitivity, and predictive quality |
| Recurrent refinement | Shared blocks update hidden states repeatedly | Whether training makes intermediate states useful; behavior at unseen loop counts |
| Numerical composition | Attention and FFN act as separately integrated components | Consistency assumptions and benefit at matched compute |

### DiffEqFormer: an explicit hidden ODE

Tong et al., [Neural ODE Transformers: Analyzing Internal Dynamics and Adaptive Fine-tuning](https://proceedings.iclr.cc/paper_files/paper/2025/hash/82e3509393670d9d478c359b4ce6f950-Abstract-Conference.html), **ICLR 2025**. Read Sections 4 and 6 in the [full text](https://arxiv.org/html/2503.01329v1).

Attention and FFN contribute to the hidden-state derivative. Their weights are functions of continuous depth, implemented with a sinusoidal embedding, MLP, and projection. Thus an ODE architecture need not force the same numerical weight matrix at every depth. Experiments concern language modeling and solver-dependent fine-tuning, not image generation. WikiText103 small-model perplexities are 22.87 for GPT, 36.27 for a shared-parameter baseline, and 22.84 for their model. This supports investigating depth dependence; it does not establish a benefit for our FFN or justify its extra parameter cost.

### Looped-DiT: useful intermediate predictions

Chng et al., [Looped Diffusion Transformer](https://arxiv.org/abs/2609.40305), **30 September 2026 preprint**. Read Sections 2 and 3.3, especially Table 4.

The model repeats shared middle blocks within one pixel-space text-to-image prediction. Each intermediate state can be decoded and supervised against the same clean image. Self-modulating attention regulates repeated updates. In the reported ablation, the four-benchmark average is 55.2 without looping, 56.2 with naive looping, 57.5 with deep supervision, and 59.1 with deep supervision plus XSA. Consequently, its best result cannot be attributed to recurrence alone. It provides a close image-prediction comparison, but does not require the repeated core to approximate a continuous ODE.

### LiFT: assign different tasks to different depths

Derakhshani et al., [LiFT: Loop Flow Transformers](https://arxiv.org/abs/2610.05538), **4 October 2026 preprint**. Read Sections 3–4 and Appendices B.4/C.2.

Intermediate velocity predictions regress toward points between a detached initial estimate and the flow-matching target. The shared core receives a continuous depth coordinate, with input reinjection and an auxiliary initial-prediction loss. On latent ImageNet, L/2 R10 improves from FID 20.30 at two training loops to 10.95 at eight inference loops. Yet B/2 remains weaker than its dense comparison. The authors explicitly distinguish their recurrence from an inner numerical ODE solver; previous depth and step length are not explicit core inputs. Its main lesson is depth-specific supervision, not a guarantee that more accurate integration improves generation.

### ELT: learn useful exits through self-distillation

Goyal et al., [ELT: Elastic Looped Transformers for Visual Generation](https://arxiv.org/abs/2604.09168), **April 2026 preprint, revised July 2026**. Read Section 4 and the ILSD analysis in Section 5.4.

The deepest training rollout supplies a detached teacher for a sampled intermediate exit, alongside ground-truth supervision. This trains a range of useful compute budgets. The reported image/video comparisons include FID 2.0 on ImageNet under the stated matched inference-compute setting. This is evidence for a training strategy for recurrent architectures, not for the professor's Taylor layer. Distillation and its loss schedule would be additional changes in our project and must be separated from a local-layer test.

### Neural-RDM: a direct architectural neighbor

Ma et al., [Neural Residual Diffusion Models for Deep Scalable Vision Generation](https://proceedings.nips.cc/paper_files/paper/2024/hash/d51ceadaf09a4699f18986702df24987-Abstract-Conference.html), **NeurIPS 2024**. Read Sections 2.1–2.3 and the gate/depth comparisons; accessible [arXiv full text](https://arxiv.org/html/2406.13215v2).

It uses learned scale/shift gates on residual updates and a continuous-depth interpretation to motivate deep visual generators. This is relevant prior art for claims about making denoisers dynamically structured. Formal resemblance between residual dynamics and a probability-flow ODE does not identify hidden depth with sampling time, nor establish that the learned hidden field equals the data-transport field. A new project needs a more specific claim than dynamical compatibility.

### Numerical analysis and mathematical background

Lu et al., [Understanding and Improving Transformer From a Multi-Particle Dynamic System Point of View](https://arxiv.org/abs/1906.02762), **2019**, is useful historical context rather than a frontier recommendation. Operator splitting motivates the FFN–attention–FFN arrangement of Macaron Net. It illustrates a route from a numerical argument to a concrete architecture; numerical-order guarantees require consistent component flows and do not automatically apply to freely learned sublayers.

Geshkovski et al., [A mathematical perspective on Transformers](https://arxiv.org/abs/2312.10794), **Bulletin of the American Mathematical Society 62 (2025), 427–479**, develops interacting-particle and transport viewpoints. Read Part I and the assumptions preceding clustering results. Many main results concern simplified attention/normalization dynamics, not a full trained DiT. This is the strongest background reading here for formulating a precise mathematical question, rather than a ready-made generative method.

Chen et al., [Neural Ordinary Differential Equations](https://proceedings.neurips.cc/paper_files/paper/2018/hash/69386f6bb1dfed68692a24c8686939b9-Abstract.html), **NeurIPS 2018**, supplies the basic definition: parameterize the hidden derivative and integrate it. A SwiGLU can itself parameterize that derivative; replacing its activation with a polynomial is a separate choice.

MeanFlow, [Mean Flows for One-step Generative Modeling](https://papers.neurips.cc/paper_files/paper/2025/hash/6d13e085b79d454da5910e4ca82a3d9d-Abstract-Conference.html), **NeurIPS 2025**, instead changes the predicted flow quantity to average velocity over an interval. It is contextual evidence about targets and sampling, not evidence that a Transformer backbone has been made continuous-depth.

## Where the original ODELayer sits

For column-vector notation, the professor's layer computes:

```text
a = sigmoid(t_func(x) / tau) * scale + shift
out = x + a*P*x + (a^2/2)*P^2*x + ... + (a^K/K!)*P^K*x
```

This is a truncated matrix exponential for the linear dynamics `dh/dr = P*h`, initialized at `x`, evaluated at the input-selected duration `a(x)`. The clock is computed once and frozen during that solve. This gives the layer a precise local interpretation. The SiLU between two such layers and the surrounding independently parameterized blocks do not imply a single shared generator for the whole model.

Two useful distinctions follow from this calculation:

1. **Frozen duration versus evolving state-dependent clock.** If the intended field were `F(h)=a(h)*P*h`, its second Taylor derivative contains an additional term:

   ```text
   F'(h)F(h) = a(h)^2*P^2*h
                + a(h)*(grad(a(h)) dot P*h)*P*h
   ```

   The original layer omits the second term. It should not be described as a second-order Taylor solver for that nonlinear field.

2. **More loops versus finer integration.** To test a fixed hidden ODE over `[0,1]`, change `N` while changing the step to `1/N`. Applying the same unscaled residual update more times generally changes the modeled transformation or horizon. At fixed learned weights, solver refinement should approach that learned solution; it need not improve task quality.

An exact same-dimensional ODE flow is locally invertible under the usual regularity conditions. A clean-image regression need not be. Using an encoder, hidden evolution and decoder avoids imposing that invertibility requirement directly on the image-to-prediction map. This is another reason not to equate the hidden trajectory with the outer image trajectory.

## Implications for our project

These are deductions and design questions, not published or newly established performance claims:

- A credible claim must specify whether the contribution is the local operator, continuous depth, or training of iterative refinement. Each requires different evidence.
- For an ODELayer-focused project, first identify the intended local field and clock semantics. Any higher-order or spectral argument must apply to that actual map, including its gate, nonlinearity and surrounding residual path.
- A continuous-depth SwiGLU control isolates the effect of a shared dynamical backbone from the special polynomial operator. Matching parameters alone does not match training or sampling cost.
- If studying recurrence, keep intermediate supervision as a separate factor. The literature provides several incompatible prescriptions: same target, teacher prediction, or depth-dependent target. They should not be combined into one initial variant.
- Local synthetic results do not establish a JiT image-generation gain. Our recorded shared-potential comparison remains negative, and the saved depth-gradient decision does not authorize P-depth training. This review does not reopen those stopped branches.

The most defensible research question is: **which property of the local update makes iterative prediction more accurate or more efficient under a controlled budget?** “The sampler is an ODE, therefore the predictor should be an ODE” is a motivation to investigate, not a mathematical implication.

Recommended reading order: DiffEqFormer for an actual hidden ODE; Looped-DiT for a closely related clean-image predictor; LiFT for the importance of depth-specific targets. Use the mathematical perspective article when narrowing a theorem, and ELT/Neural-RDM when checking novelty and controls.

## Review provenance

Searches and access limits are in [notes.md](notes.md); primary-source metadata and inspected sections are in [sources.json](sources.json). The conceptual schematic was generated with the available image-generation tool using only public concepts, then visually checked. It contains no experimental measurements. Prior reports, training code, checkpoints, and queued jobs were not changed.

Workflow assistance used the literature-review, planning-with-files, and scientific-schematics skills. Their procedural-library reference is Kassis, Agarwal, He, Patel and Brueckner (2026), [Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://doi.org/10.48550/arXiv.2609.00065). This is a workflow acknowledgment, not scientific evidence for the architecture.
