# After the generator comparison: what the literature actually licenses

Reviewed 2026-10-24 against the current [task plan](task_plan.md), [generator derivation](clock-generator-mathematics.md), and [Stage15 result](analysis-additive-generator/analysis-report.md). This is a bounded primary-source audit. It does not replace the negative experimental decisions or propose another training sweep.

**Decision:** no verified result found here establishes that a conditional linear generator or a matrix-exponential activation should beat SwiGLU for our FM regression. Three narrowly relevant readings clarify representation boundaries, but none selects the next architecture. The immediately useful question is whether our controller lacks useful response directions, has badly scaled responses, or has learned an unsuitable downstream representation. Those explanations need different interventions.

## Three papers, with exact transfer boundaries

### 1. DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products

Julien Siems et al., **NeurIPS 2025**. [Official publication](https://papers.nips.cc/paper_files/paper/2025/hash/e1ea2520fbf9cd1600b287dde67e0a3c-Abstract-Conference.html); [primary full text, v4](https://arxiv.org/html/2502.10297v4).

Sections 3–5 and Theorem 1 connect the actual transition family to specific finite-precision group-word problems. Each factor is `I-beta*k*k^T`, with unit k; allowing beta up to 2 admits reflections. Products create transitions absent from a single factor. The finite-precision theorem uses a particular two-layer construction, signed diagonal transitions in the first layer and 2D reflections in the second. It is not a universal approximation result for a single tokenwise FFN. The experiments also expose a representation/learning gap: two-layer DeltaNet struggles on a task its theorem says it can represent.

**Implication:** a useful structural argument must identify an operator the current family cannot express, then show that restriction matters in trained models. Our scalar witness satisfied the first requirement but did not establish the second. This paper does not justify importing its reflection factors, negative transition eigenvalues or sequence-speed claims into FM.

### 2. KoopGen: Koopman Generator Networks for Representing and Predicting Dynamical Systems with Continuous Spectra

Liangyu Su et al., **February 2026 preprint**. [Primary full text](https://arxiv.org/html/2602.14011v1).

Sections 2.3.1–2.3.3 use state-dependent convex combinations of skew-adjoint and self-adjoint basis matrices, then apply `exp(delta*G(z))*z`. Convex combinations preserve the stated matrix symmetry classes. The operator-theoretic discussion begins with a bounded generator on an observable Hilbert space; that setting must be distinguished from a finite matrix evaluated anew at each state. The experiments concern physical dynamical-system prediction, not FM FFNs. No dimension-matched or optimizer-convergence theorem found in these sections predicts superiority of its mixture over our diagonal controller.

**Implication:** it supplies a concrete alternative to a single fixed generator, not proof that the alternative is useful here. In particular, pointwise matrix structure does not establish stability of the complete state-dependent map. The independent counterexample below blocks using this paper as a license for a new skew/symmetric mixture experiment.

### 3. Intelligent Matrix Exponentiation

Thomas Fischbacher et al., **2020 preprint**, included for direct algebraic relevance rather than recency. [Primary paper](https://arxiv.org/pdf/2008.03936).

Sections 3.2–3.3 and Appendix A.1 construct an affine input-dependent matrix and read out its exponential. Strictly upper-triangular matrices encode monomials through products along matrix paths; sufficient matrix size therefore gives polynomial representation and compact-domain universal approximation. The matrix dimension grows with the construction. It is not a fixed-width result for `s(h)*P`, `D(h)*P`, or `cP+diag(s(h)-c)`. Its robustness bound also depends on the input-to-matrix map, rather than only the spectrum at one input.

**Implication:** the expressive object is the combination of input-dependent matrix entries, readout and size. Merely keeping more terms of a fixed-P polynomial does not inherit that theorem. This is a prior-art and theorem-scope check; it does not nominate an order increase or exact-exponential rescue.

## Independent algebra that sharpens the next diagnosis

The following statements are our derivations for the current code, not claimed results of the three papers.

### A. The vector change initially rescales directions rather than creating a new span

For a fixed hidden state h, write the clock vector as s and treat its entries as independent local coordinates. At K1:

```text
F_mul(h,s) = h + diag(s) P h
F_add(h,s) = h + c P h + diag(s-c) h

dF_mul/ds = diag(P h)
dF_add/ds = diag(h)
```

For channel j, both clock responses point along coordinate axis e_j. They differ in scale and possibly sign or whether the response is zero. When every receiving feature is nonzero, both local response spans are all of hidden space. This statement is about infinitesimal, unconstrained clock perturbations; finite bounded-clock reachable sets can still differ.

At the paired constant-clock initialization, the complete initial function and hidden input derivatives agree. If M is the shared derivative from this solve's output to the network output, the corresponding complete-network clock responses are:

```text
R_mul = M diag(P h)
R_add = M diag(h).
```

Their column spans agree when both diagonal factors are invertible. Normalized corresponding columns agree up to sign. Thus the vector additive arm does not unlock a new local response subspace **at that paired initialization**. The scalar case is different: `M P h` and `M h` need not be collinear. After training, different hidden states and downstream derivatives can change these comparisons.

This sharpens the Stage15 interpretation. The vector product16 gain is compatible with better gain allocation, fewer nearly inactive channels, or a more favorable learned representation. The scalar direction-switch witness is not a sufficient explanation for it. A learned controller's parameter tangent also contains its own parameter Jacobian; an independent-clock span is not a parameter-optimization theorem.

### B. A sufficiently general K1 generator is already an ordinary gated FFN

Consider a frozen conditional generator with rank-one basis matrices:

```text
A(h) = P0 + sum_i a_i(h) u_i v_i^T
F(h) = h + A(h)h
     = h + P0 h + U [a(h) * (Vh)].
```

Here `*` is elementwise multiplication, columns of U are u_i and rows of V are v_i^T. With `a(h)=SiLU(Wh+b)`, the adaptive term is exactly a SwiGLU-style gated FFN, subject to the explicitly chosen biases and static residual path.

This is a constructive identity, not an approximation theorem. Adding freely learned generator directions can therefore recover the baseline under ODE terminology. A useful new ODELayer must specify what restriction it keeps, why that restriction fits the target, and which matched control tests its value. Merely permitting a richer generator does not provide that argument.

### C. Pointwise generator structure does not certify the nonlinear map

For `F_delta(z)=exp(delta*G(z))*z`, freezing G during one application gives a conditional linear solve. Re-evaluating G at the next state generally destroys the autonomous linear semigroup identity. For scalar `G(z)=z`:

```text
F_delta(F_delta(z)) = z + 2 delta z² + 3 delta² z³ + O(delta³)
F_(2delta)(z)       = z + 2 delta z² + 2 delta² z³ + O(delta³).
```

Even an orthogonal conditional matrix need not yield a 1-Lipschitz map. Let `F(h)=Rot(k*h1)h` in two dimensions. Every evaluated rotation preserves the input norm. At `h=(0,r)`, however, the Jacobian is `diag(1-k*r,1)`, whose norm can be arbitrarily large. Derivatives of the controller matter. These examples do not disprove the physical prediction results of KoopGen; they delimit which guarantees can transfer.

## Concrete next mechanism test

Continue the already selected **read-only** paired-checkpoint diagnosis. It should report all of the following at the same declared anchors and checkpoints:

- Raw complete-network clock response columns, their norms, and zero/near-zero response counts.
- Projection of the residual velocity error onto their span, alongside a bounded-gain or singular-value-aware calculation. A direction with negligible gain may be present algebraically but practically inaccessible.
- Column-normalized comparisons and the downstream derivative M, so feature amplitude is not mistaken for a newly available direction.
- The actual controller-parameter response when making an optimization claim. Independent output-clock perturbations are an oracle relaxation of the learned controller.

At a fixed anchor, with residual `e=v_target-v_model` and response matrix R, an infinitesimal clock change gives predicted squared-error reduction

```text
2 e^T R delta_s - ||R delta_s||².
```

The unconstrained optimum only measures conditional local capacity. Inspecting the singular values and the norm/allowed range of delta_s distinguishes missing direction from excessive required gain. Across examples the controller is shared, so per-anchor optima need not be jointly realizable. This diagnosis cannot itself establish an architecture-wide lower bound.

Only if a material residual component is unavailable to the measured frame should a different generator basis become the next candidate. If normalized frames already cover the errors but useful gains are weak, another operator-capacity theorem is addressing the wrong failure. If the downstream representation dominates, changing the local receiver again is likewise poorly targeted. None of these outcomes presently warrants a temperature, order or spectral rescue.

## Search scope and exclusions

Search date: 2026-10-24. Queries included `"matrix exponential" "input-dependent" neural network representation 2025 2026`, `"matrix polynomial" "activation" neural network expressive 2025`, `"conditional" "linear generator" neural network flow matching`, `"state-dependent" "linear" "generator" "neural" 2025 2026`, `"selective state space" "universal approximation" 2025 2026`, and exact-title follow-ups. The selected primary PDFs/HTML, theorem statements and constructions were inspected. This is not an exhaustive novelty search.

Recent papers on approximating the matrix exponential *with* neural networks study a different direction of approximation. Physical-system solver papers without an applicable representation statement, application-only hypernetworks, and broad SSM analogies were excluded. The earlier Mamba-3/Kimi Linear review was not counted as new evidence. The direct matrix-activation theorem found was older; no more recent title was substituted merely to make the bibliography look current.
