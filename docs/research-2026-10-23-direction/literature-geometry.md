# Literature audit: target geometry, conditional variance, and stability

Checked 2026-10-23. This is a targeted extension of the [October 21 review](../research-2026-10-21-reassessment/literature.md), not an exhaustive literature survey. Seven papers were inspected beyond the previous list; an eighth guide was checked as a source map. Publication status and reading depth are stated below. No experiments were launched.

## What changes the project decision

Three distinctions matter more than adding another matrix constraint:

1. The random conditional training label, its conditional mean, and the optimizer Hessian are different mathematical objects. A paper proving sensitivity of one does not prove instability of the others.
2. The geometry required by the complete velocity field is observable in our analytic examples. A constraint on a hidden matrix P need not produce that geometry after gates, nonlinearities, residual blocks, and the output projection.
3. A conditional straight interpolation is not generally a straight trajectory of the marginal flow. The internal Taylor polynomial also does not approximate that external trajectory merely because it contains a second-order term.

The best new readings for this project are **Stable Velocity**, **Carré du champ Flow Matching**, and the **Flow Map Matching** mathematical framework. The first separates label variance from approximation; the second makes the geometry being preserved explicit; the third shows how an objective can actually encode a dynamical identity. None establishes that the shared-P polynomial is the correct FFN.

## Selected sources and what transfers

### 1. Stable Velocity: A Variance Perspective on Flow Matching

Donglin Yang et al., **ICML 2026**. [Published record](https://proceedings.mlr.press/v306/yang26dq.html); [full text, v2](https://arxiv.org/html/2602.05435v2).

Read Section 2, Section 3.1, Theorem 3.1, Appendix E.1–E.3 and F.2. The central measurable quantity is the conditional covariance of the training velocity label. Their multi-reference construction reduces target variance while preserving the marginal optimum under its specified joint sampling construction. Unbiasedness is not a generic property of any self-normalized weighted estimator. The reported near-data low-variance regime is an empirical/distribution-dependent observation, not a theorem covering every smooth population distribution. Their synthetic example uses a 10-dimensional, 100-component anisotropic GMM and a numerical approximation to the target field.

**For us:** compare noisy CFM supervision with the exact marginal oracle on the same analytic distribution only if optimization versus approximation remains unresolved. This is a diagnostic, not a proposal to adopt their complete training recipe. We already have an exact Gaussian-mixture oracle, a useful advantage for this question.

**Does not transfer:** a low-variance time threshold, a hidden-P constraint, or an explanation for why our scalar gate helps.

### 2. Carré du champ flow matching: better quality-generalisation tradeoff in generative models

Jacob Bamberger et al., **ICLR 2026**. [Published record](https://proceedings.iclr.cc/paper_files/paper/2026/hash/dbc9f9d486bd1714896111b0fcd78cd1-Abstract-Conference.html); [full text, v2](https://arxiv.org/html/2510.05930v2).

Read Sections 3.1–3.2, 4.3 and Appendix B. It constructs spatially varying anisotropic target smoothing from estimated local geometry, then derives a conditional transport path for those covariances. The resulting path differs from simply adding target noise and using the original independent interpolation. Its dimension/data-density experiments expose limitations of estimating local geometry.

**For us:** this is an example of an explicit chain from geometric assumption to operator to objective. It argues against treating all directions identically by default. Its manifold examples also motivate controlled tangent/normal directions if our two-dimensional tests cannot resolve directional expressivity.

**Does not transfer:** its covariance lives in data space and changes the probability path; it is not the singular spectrum of an FFN matrix. It does not justify adding covariance estimation to our current architecture or declaring that our full-support Ring8 is a low-dimensional manifold.

### 3. Flow map matching with stochastic interpolants: A mathematical framework for consistency models

Nicholas M. Boffi, Michael S. Albergo and Eric Vanden-Eijnden, **TMLR, May 2025**. [Published paper](https://openreview.net/pdf/7066cf85c9c2a529244ff320f5cf3ea10cc697d4.pdf); [full text, v2](https://arxiv.org/html/2406.07507v2).

Read Sections 3.4–3.6. It learns two-time maps and derives their training identities rather than assuming that a familiar-looking architecture inherits flow properties. Section 3.6 is especially relevant: inserting a random conditional velocity into a squared derivative residual introduces an additional covariance term. Conditional expectation cannot be pushed through an arbitrary nonlinear objective. The paper also finds a practical difference between mathematically motivated Eulerian and Lagrangian formulations.

**For us:** if a later ODELayer design uses a derivative/consistency loss, its population minimizer must be derived explicitly. A plausible dynamical residual is not sufficient. A finite polynomial feature map is not automatically a two-time generative flow map.

**Does not transfer:** the paper does not recommend replacing a DiT FFN with a Taylor polynomial. Few-step generation is a different research objective and should not be added to the present project by default.

### 4. Jacobian Regularization Stabilizes Long-Term Integration of Neural Differential Equations

Maya Janvier, Julien Salomon and Etienne Meunier, **February 2026 preprint**. [Full text, v1](https://arxiv.org/html/2602.04608v1).

Read Sections 2.2, 3.1–3.2 and 4.4. For known dynamics, the method matches directional derivatives of the learned and true vector fields. For unknown dynamics it uses finite differences. The object being regularized is derivative **error**, not simply derivative magnitude. Tests concern physical ODE/PDE rollouts trained on short observed trajectories.

**For us:** our analytic targets allow an unusually clean comparison between shrinking a derivative and correcting it. If a derivative-training intervention is needed, it should initially be an oracle diagnostic applied to both ODE and SwiGLU, not a claimed ODE-specific method.

**Does not transfer:** physical long-horizon trajectories differ from our finite generative transport; access to a true Jacobian on toy data does not scale directly to images. A successful oracle intervention would identify a failure mechanism, not establish a deployable architecture.

### 5. Stability-Informed Initialization of Neural Ordinary Differential Equations

Theodor Westny et al., **ICML 2024**. [Published record](https://proceedings.mlr.press/v235/westny24a.html); [full text, v3](https://arxiv.org/html/2311.15890v3).

Included as a precise mathematical reference, not a recent headline. Read Sections 2–4 and 7. It relates initialization and learned dynamics to the actual numerical solver's stability region. Its construction is for a specific feedforward dynamics class with dimensional restrictions.

**For us:** continuous-time decay and stability of a truncated update are separate. If the polynomial block is used as a numerical approximation, analyze its stability polynomial with the effective step sP. If it is used only once as a feature map, insist on evidence before interpreting its gains as solver stability.

**Does not transfer:** a negative real spectrum of P is not enough to ensure stability of a finite Taylor map; a solver-aware initialization theorem does not cover our whole residual/gated architecture.

## Screened recent papers: useful cautions, not design authorities

### 6. Flow Matching in the Low-Noise Regime: Pathologies and a Contrastive Remedy

Weili Zeng and Yichao Yan, **September 2025 preprint**. [Full text, v1](https://arxiv.org/html/2509.20952v1). An ICLR submission PDF was discoverable, but acceptance was not verified; do not label it an accepted ICLR paper.

Read Section 4 and Appendix A.2. Proposition 1 fixes a clean example and compares noisy inputs and their conditional velocity labels. That sensitivity can diverge while the marginal regression optimum remains smooth. The distinction is material to our Gaussian experiments. Its broad interpretation therefore cannot serve as proof that every learned marginal velocity must have an exploding Jacobian near data. The counterexample below is our own calculation, not a result claimed by the paper.

### 7. A Lagrangian View of Flow Matching

Peyman Milanfar, **September 2026 preprint**. [Full text, v2](https://arxiv.org/html/2609.00198v2).

Read Sections 2–4 and the explicit qualification in Section 3. The derivation imposes invariance of the predicted endpoint along a chosen residual flow. Section 3 acknowledges that invariance is an additional choice. This is not a consequence of ordinary conditional FM regression. Some subsequent broad statements about static denoisers and straight FM trajectories should not be imported into our project. The exact Gaussian calculation below is sufficient to show why the imposed property need not hold for our marginal flow.

### 8. Flow Matching Guide and Code — source map only

Yaron Lipman et al., December 2024. [Official record](https://arxiv.org/abs/2412.06264). The authors' guide is an appropriate map of probability paths, conditional regression and extensions, not evidence for an ODEFFN. The abstract/metadata were verified; HTML was unavailable and the browser rejected the 18.6 MB PDF as too large, so a new full-text reading is **not** claimed here. The accessible primary papers above supply the specific mathematical claims used in this memo.

## Independent mathematical checks that prevent misusing this literature

Use our convention: epsilon is standard Gaussian, y is data, x_t=(1-t)epsilon+t y, and U=y-epsilon. Let D(x,t)=E[y|x_t=x] and V(x,t)=Cov(U|x_t=x). Under the usual differentiability/integrability conditions, for 0<t<1:

```text
v*(x,t) = (D(x,t)-x)/(1-t)
J D(x,t) = t Cov(y|x_t=x)/(1-t)^2
V(x,t) = Cov(y|x_t=x)/(1-t)^2
J v*(x,t) = (-I+t V(x,t))/(1-t)
```

This gives a useful exact relation between supervision variance and target geometry, but does not make them interchangeable. It also implies that Jv* is symmetric and its smallest eigenvalue is at least -1/(1-t). The complete target has this property; an individual hidden P need not.

For scalar y~N(0,q), q>0, define c(t)=(1-t)^2+t^2 q. Gaussian conditioning gives:

```text
v*(x,t) = [t q-(1-t)] x / c(t)
Jv*(x,t) = [t q-(1-t)] / c(t)
Var(U|x_t=x) = q / c(t)
```

As t approaches 1, Jv* approaches 1 and Var(U|x) approaches 1. Neither a divergent marginal Jacobian nor vanishing conditional velocity variance is universal for smooth target distributions. Conversely, on a finite empirical distribution away from mode boundaries, posterior identification of a data point can make conditional velocity variance small. The order of the finite-data, smoothing and zero-noise limits matters. Our finite training dataset and analytic population oracle answer related but different questions.

The same Gaussian supplies a counterexample to invariant denoising along every exact FM trajectory. The exact marginal flow from a standard-normal initial point z is:

```text
X_t = sqrt(c(t)) z
D(X_t,t) = t q z / sqrt(c(t))
d D(X_t,t)/dt = q(1-t) z / c(t)^(3/2)
```

The last expression generally does not vanish. No model approximation error is involved. A hidden-block semigroup, an invariant denoiser, and a marginal FM flow therefore require separate arguments.

## Implications for existing evidence and a small next decision

The [gate-geometry diagnostic](../research-2026-10-21-gate-geometry/findings.md) establishes that trained input gates often improve the local full-field derivative. It does not identify whether the benefit comes from a continuous-flow structure, generic gating, or optimization under noisy targets. In particular, Gaussian derivatives improve when gates are allowed to vary, yet fixed-gate retraining can recover similar quality. This is compatible with compensation inside an adapted model.

Before adding a constraint, two cheap questions are more informative:

- Decompose the already saved Jacobian error into symmetric error and skew error. Since Jv* is symmetric, `||Jtheta-Jv*||F² = ||sym(Jtheta)-Jv*||F² + ||skew(Jtheta)||F²`. Determine which part the native gates correct. Check spectral lower-bound violations as a diagnostic, not a new training penalty.
- If the shared-polynomial comparison remains ambiguous, train the smallest selected architecture comparison under exact marginal labels versus ordinary conditional labels on the **same** Gaussian/Ring8 path. The population objectives differ by the architecture-independent irreducible conditional variance, so a different finite-training ordering points toward optimization/noise interaction, subject to parameterization and compute controls. It does not by itself prove a unique inductive bias.

Derivative supervision, path redesign, manifold covariance estimation and flow-map learning should remain separate alternatives. Adding them together would remove the ability to explain what the original ODELayer contributed.

## Search record and unresolved naming

Queries were issued through web search on 2026-10-23 and followed to primary arXiv, PMLR, ICLR or OpenReview pages. Principal query groups:

```text
"Tied Flow" generative models
"Tied-Flow" paper
"TiedFlow"
"Tied" "Flow" "flow matching"
Tied Flow diffusion transformer architecture weight sharing
"Tied Flow" site:arxiv.org
"TiedFlow" site:github.com
"TiF" flow matching architecture
flow matching Jacobian covariance geometry 2025 2026 velocity field neural architecture
neural ODE regularization Jacobian stiffness training finite Taylor stability
flow matching low rank velocity field Jacobian 2025 2026
"Stable Velocity" arxiv
"Flow Matching in the Low-Noise Regime" arxiv
"Flow Matching in the Low-Noise Regime" site:proceedings.iclr.cc
"Carré du champ flow matching" ICLR 2026
"Flow map matching with stochastic interpolants" site:openreview.net/forum
"stability" "initialization" "neural ODE" Westny 2024
"Flow Matching Guide and Code"
```

**Tied Flow/TiedFlow was not reliably identified.** Results for other similarly named models were not substituted. No claims about its architecture, contribution or novelty comparison should be made without its actual title/link. Unrelated application papers and papers whose only relevance was the word “flow” were excluded. New preprints were screened for the actual mathematical object and assumptions, rather than selected by date alone.
