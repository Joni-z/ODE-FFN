# Architecture, prediction targets, and learning dynamics

Checked 2026-10-23. This is a targeted critical reading, not a systematic review. Primary sources below were selected for questions raised by our experiments. Venue labels refer to verified publication records; an arXiv revision date is not a new conference publication. This note complements [literature-geometry.md](literature-geometry.md).

## 1. Spectral bias: a model for the kind of theory we need

**Binxu Wang and Cengiz Pehlevan, An Analytical Theory of Spectral Bias in the Learning Dynamics of Diffusion Models. NeurIPS 2025; arXiv v3 revised April 2026.** [Published record](https://papers.nips.cc/paper_files/paper/2025/hash/8a0d3f77bb435817807d463c5dcef1ab-Abstract-Conference.html), [full text](https://arxiv.org/html/2503.03206v3).

Read Sections 4.1–4.3, the commutativity condition, and the distinction between full-width and local convolutions. The paper solves restricted linear-denoiser training and sampling dynamics, connecting data covariance modes to learning rates. Its theoretical spectral bias concerns **data modes**, not the singular values of an arbitrary hidden FFN matrix. Full-batch gradient flow and linear/convolutional settings do not establish rates for our AdamW-trained nonlinear model. The useful research pattern is to derive a rate or approximation prediction on a specified model, then test its failure boundary. For ODE-FFN this motivates the function-matched P-versus-B comparison, not flattening P's singular values. Relevant sections and publication metadata inspected; the long appendices were not independently verified in full.

## 2. Architecture can change which prediction target works

**Anne Gagneux, Ségolène Martin, Rémi Gribonval, Mathurin Massias, Training Flow Matching: The Role of Weighting and Parameterization. ICLR 2026 DeLTa Workshop.** [Publication PDF](https://openreview.net/pdf?id=RYQBTBZxNl), [full text](https://arxiv.org/html/2603.06454v1).

This is a workshop paper, not an ICLR main-conference paper. Sections 3–5 separate the function class from loss weighting. Their comparisons favor velocity prediction for some U-Nets and small-patch transformers, but clean-data prediction for large-patch transformers/MLPs and some limited-data settings. Their controlled Fourier image dataset varies intrinsic dimension. This qualifies an unconditional reading of JiT's manifold explanation. It does not refute JiT's reported large-patch results or prove one universal architecture rule. For us, changing from velocity MSE to unweighted clean-data MSE would confound architecture, parameterization and time weighting. Main methods/results and relevant appendix descriptions inspected.

## 3. JiT establishes the task to which our block must eventually transfer

**Tianhong Li and Kaiming He, Back to Basics: Let Denoising Generative Models Denoise.** [Full text, January 2026 revision](https://arxiv.org/html/2511.13720v2).

Sections 3.2–3.3 distinguish prediction space and loss space; the synthetic experiment embeds two-dimensional data into larger ambient spaces and compares prediction targets under velocity loss. Sections 4.2 and Appendix B.2 examine large patches and output preconditioning. These are reasons to test an architecture's interaction with prediction target and capacity. They do not require a hidden FFN to approximate an exponential. Our direct-velocity 2D toy and historical normalized image ODE block do not yet constitute a test of the professor's original block inside JiT. Relevant full-text sections rechecked.

## 4. A solvable architecture-to-distribution analysis

**Hugo Cui, Cengiz Pehlevan, Yue M. Lu, A solvable model of learning generative diffusion: theory and insights. NeurIPS 2025.** [Published record](https://proceedings.neurips.cc/paper_files/paper/2025/hash/082d3d795520c43214da5123e56a3a34-Abstract-Conference.html), [full text](https://arxiv.org/html/2501.03937v2).

Read the setting, Results 2.1–2.3, and the Gaussian-mixture examples. A shallow autoencoder with tied encoder/decoder weights, skip connections and online SGD admits asymptotic descriptions of learning and generated low-dimensional projections. Those assumptions differ from our deep FFN and AdamW. This is valuable as a precedent for a genuinely mathematical project: characterize what an architecture learns, rather than only bounding its norm. It is **not** identified as the user's earlier “Tied Flow” paper. Relevant model/results/examples inspected; no claim of checking every asymptotic proof.

## 5. Input gating is an alternative explanation to beat

**Zihan Qiu et al., Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free. NeurIPS 2025.** [Published PDF](https://papers.neurips.cc/paper_files/paper/2025/file/904e89bb4e632e75fb47f093b620b257-Paper-Conference.pdf), [arXiv v1 full text](https://arxiv.org/html/2505.06708v1).

Unlike the October 21 review, this time Sections 4.1–4.2 and the ablations were inspected. The paper compares gate positions, granularity, alternative nonlinearities and input-independent variants. Its findings make generic representation and optimization effects a serious competing explanation for our gate. Attention sinks and query-specific sparsity in language models are not established mechanisms in our tokenwise toy FFN. Our duration is bounded below by 0.2 and modulates a polynomial operator; it is not their suppressive output gate. The proceedings PDF metadata/introduction and accessible arXiv mechanism sections were read; their table numbering/version differences should not be conflated.

## 6. Input-dependent duration already has substantial prior art

**Albert Gu and Tri Dao, Mamba: Linear-Time Sequence Modeling with Selective State Spaces. COLM 2024.** [Full text](https://arxiv.org/html/2312.00752v2).

Sections 3.2, 3.5.1 and Appendix C explicitly connect input-dependent discretization to gating; Theorem 1 recovers a gated recurrent update in a scalar special case. The sequence state, forcing term and discretization are materially different from our two-layer FFN. This is included as necessary prior art, not a new frontier recommendation. A novelty claim based only on learning s(h) would be weak; a claim about the specific shared-P polynomial restriction still requires evidence. Relevant derivation and limitations discussion inspected.

## 7. A recent linear theory to read with its assumptions visible

**Qing Jin and Chaoyang Wang, Revisiting Diffusion Model Predictions Through Dimensionality. 2026 preprint, v2 March 27.** [Record](https://arxiv.org/abs/2601.21419), [full text](https://arxiv.org/html/2601.21419v2).

Section 3 analyzes a single linear predictor shared across noise levels, with whitened intrinsic coordinates embedded in a larger space. Parallel and perpendicular subspaces have distinct training dynamics. This gives a tractable example of how prediction targets interact with dimensionality. It does not prove that intrinsic dimension alone decides the best target for a time-conditioned nonlinear transformer; the workshop study above supplies a useful empirical counterweight. No venue acceptance is inferred from template keywords. Relevant assumptions and equations inspected; the learned-target method is not proposed as another trick for our project.

## 8. A reference map, not a new architecture claim

**Yaron Lipman et al., Flow Matching Guide and Code (2024).** [Record](https://arxiv.org/abs/2412.06264), [authors' implementation](https://github.com/facebookresearch/flow_matching).

Useful for organizing path, coupling, prediction and solver choices. This pass verified the record and official code documentation. The large PDF exceeded the web extraction size limit, so this note does not claim a fresh full-text reading. It is background navigation, not evidence that a particular FFN is optimal.

## Reading order for this project

1. Spectral Bias: what a mathematical explanation of architecture and training can look like.
2. JiT together with Weighting and Parameterization: why the final objective/architecture pair matters.
3. A Solvable Model: how to choose an analyzable target and reduced model.
4. Stable Velocity and Lipschitz Regularity in the companion note: separate regression variance, target geometry and sampling error.
5. Gated Attention and Mamba: challenge generic-gating novelty claims.

No reviewed paper establishes that P-squared with coefficient s-squared/2 is optimal for our objective. Conversely, an ODE resemblance alone does not supply such a result.

## Search and access ledger

Public queries on 2026-10-23 included: `TiedFlow diffusion flow transformers weight tying`; `"Tied" "Flow Matching" diffusion architecture`; `"Gated Attention for Large Language Models"`; `"Training Flow Matching: The Role of Weighting and Parameterization"`; `"flow matching" "survey" 2025 2026 theory`; `"Flow Matching Guide and Code"`; `"flow matching" "feed-forward" "2026" architecture`; `"diffusion" "polynomial" "feed-forward" transformer 2025 2026`; `"A solvable model of learning generative diffusion"`; `"An Analytical Theory of Spectral Bias in the Learning Dynamics of Diffusion Models"`. Citation chaining from the weighting paper found the dimensionality preprint.

Secondary pages were used only for discovery; retained claims use primary papers/official proceedings. Unrelated application papers, inference caching and vague PDE analogies were excluded from recommendations. Exact/variant Tied Flow searches did not identify an unambiguous primary paper; no conclusion is based on that name. Guessed arXiv v3 links for Gated Attention and Guide returned 404; actual records were checked. The Gated Attention PDF timed out on a subsequent section request; arXiv v1 supplied the accessible mechanism analysis. The Guide PDF exceeded the extraction size limit. These are access limits, not evidence of paper absence.
