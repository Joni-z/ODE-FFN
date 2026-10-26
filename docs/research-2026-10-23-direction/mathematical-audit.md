# What mathematical question remains in the original ODELayer?

This note audits the implemented professor-style block in `synthetic/models.py`, the previous [derivations](../research-2026-10-21-reassessment/mathematics.md), and the completed [gate-geometry diagnostic](../research-2026-10-21-gate-geometry/findings.md). No training was launched. The algebra below is a derivation from the model, not a claim of new literature or novelty. Column-vector notation is used.

The most defensible next question is: **does tying the input-dependent operator to one generator and its polynomial powers provide useful inductive bias or optimization geometry beyond ordinary input gating?** Existing evidence shows useful gate derivatives in an adapted network. It does not yet show a special continuous-flow advantage.

## 1. Distinguish three things the current K2 block bundles together

One layer computes

```text
s(h) = 0.2 + 0.8 sigmoid(g(h)/10)
F(h) = [I + s(h) P + s(h)^2 P^2/2] h.
```

It bundles (i) a static matrix parameterization, (ii) input-dependent gating, and (iii) a particular nonlinear relationship between the gate and the matrix coefficients. K1 versus K2 changes all of their interactions. Removing the gate changes capacity and the available input dependence. Increasing K changes matrix parameterization as well as exponential approximation. None of those comparisons alone isolates item (iii).

With a constant gate, the entire two-layer FFN is exactly `B2 SiLU(B1 h)`, where `Bi=T2(si Pi)`. The gate-geometry experiment showed that input gating improves derivatives in the trained model, including Gaussian and low-entropy contracting Ring8 states. That rules out a claim that its demonstrated usefulness is specifically mode-separation geometry. SwiGLU already supplies input-dependent gating and fits Ring8's field and derivatives better on the common interpolation panel.

### A new, exact decomposition that isolates clock curvature

Choose a fixed reference duration `s0` for a layer and define `delta=s(h)-s0`. Then

```text
B0 = I + s0 P + s0^2 P^2/2
B1 = P + s0 P^2
F(h) = [B0 + delta B1 + delta^2 P^2/2] h.        (exact)
```

This is an identity, not a local approximation. It exposes a very specific candidate mechanism: the last term is **curvature of the effective matrix as the scalar gate changes**. It is not the entire P-squared contribution: P-squared also appears in the static map and the first-order gate response.

A diagnostic comparator is

```text
F_linear_gate(h) = [B0 + delta B1] h.
```

This keeps the original trainable P and gate network, adds no trainable parameters, and uses the same P and P-squared computations. It preserves the static K2 map at `s=s0` and its first derivative with respect to the gate there. The only deleted quantity is known exactly:

```text
R(h) = F(h)-F_linear_gate(h) = delta^2 P^2 h/2
J_R = delta^2 P^2/2 + delta (P^2 h) grad(s)^T
||J_R|| <= delta^2 ||P^2||/2 + |delta| ||P^2 h|| ||grad(s)||.
```

Thus a useful small gate-gradient correction does not by itself show that the quadratic clock law is useful. The diagnostic asks whether this distinct curvature term matters after propagation through the complete model.

**Minimal use of existing checkpoints:** choose `s0` without looking at evaluation outcomes, e.g. each layer's mean gate on a fixed training calibration panel; retain the full range and variance of s. Compare the omitted term, native/modified full-model outputs and derivatives on the already declared common inputs, separated by time. For one-at-a-time layer modifications, distinguish direct block perturbation from its downstream amplification. A tiny effect would weaken the claim that the special quadratic gate law is essential; a large effect would show reliance, not prove that the law is preferable to generic gating. A distribution-dependent s0 makes this a local descriptive check and must be documented.

For a local intervention, write `F_gamma=F_linear_gate+gamma R`, with native gamma=1. The derivative of full-model validation error with respect to gamma at 1 measures whether slightly increasing or decreasing the existing curvature term would help locally. This is preferable to interpreting a large gamma=0 replacement as clean causal attribution after downstream feature distributions have shifted. Use a declared small perturbation to check the derivative, not a gamma sweep or test-set selection. Gamma is a diagnostic coordinate here; making it trainable is neither required nor a research contribution by itself.

**If a training comparison is then justified:** use fixed `s0=0.6` for both models, identical P/gate initialization, batches, optimizer, widths, and budget. Compare native K2 and this gate-linearized K2; keep SwiGLU as the external performance reference. The initial gates in the current code are near, but not exactly, 0.6, so this is not exactly function matched at initialization. Quantify that difference rather than silently change initialization. Initializing both gate output heads to zero would give exact agreement in function and first parameter derivatives, but would change the original initialization and must be a separate deliberate choice.

This comparator is a mechanism test, not a proposed improved architecture. If it performs the same, the original matrix-polynomial parameterization may still matter; what weakens is the need for its nonlinear scalar-clock relation.

## 2. A controlled comparison of matrix parameterizations is still missing

For the fixed-duration K2 network, initialize a plain, width-preserving, bias-free FFN with

```text
B1_init = T2(0.6 P1_init)
B2_init = T2(0.6 P2_init).
```

Copy every surrounding weight exactly. Its initial function then agrees for every input and time with the fixed-duration ODE network, and both have two active d-by-d matrices per FFN. One subsequently optimizes P; the other optimizes B directly. Existing SiLU uses hidden width 128 and biases, so it is not this control. Equality of total parameter count to the input-gated original is not intended here: this experiment targets the fixed-gate matrix parameterization rather than the extra gate network.

There is an important limit: **at K2 this is not globally a pure optimization comparison.** The map B=T2(A) satisfies `2B-I=(A+I)^2` and does not cover all real matrices. For example, B=diag(0,1,...,1) is impossible because det(2B-I)=-1 whereas a real matrix square has nonnegative determinant. Free B changes both the update geometry and the globally feasible matrix set. Where the derivative operator L_A below is invertible, the inverse function theorem gives local equivalence of matrix coordinates; only in that local setting can their difference be attributed solely to update geometry. If free B eventually wins, expanded representability remains an alternative explanation. If the polynomial wins, its restriction may act as implicit regularization. The K1 affine case below is a globally function-class-matched sanity control.

For `A=sP`, `B=I+A+A^2/2`, a matrix perturbation H gives

```text
L_A(H) = H + (A H + H A)/2.
```

Under Euclidean gradient flow on A, the induced update of B is `-L_A L_A* grad_B loss`. With symmetric A, the matrix-coordinate multiplier is `1+(a_i+a_j)/2`, squared in the induced update. It can amplify, attenuate, or eliminate learning directions; at A=-I all first derivatives vanish. This is a precise optimization hypothesis, not an automatic advantage. AdamW, clipping and finite learning rates need separate treatment: they are not this Euclidean gradient flow.

The fixed-gate K1 version has `B=I+sP`, an affine reparameterization. With zero weight decay, plain SGD on P is exactly SGD on B with learning rate multiplied by s squared. The analogous statement for Adam does not follow unchanged because coordinate moments and epsilon matter. This is a useful sanity case if interpreting a future optimizer experiment.

The lack of a universal optimization advantage is especially clear in a scalar example. Let `b=q(a)=1+a+a^2/2`, and let the loss be `c(b-b_target)^2/2`, where c>0 is an input second moment. Euclidean gradient flow gives

```text
da/dtau = -c (b-b_target) (1+a)
db/dtau = -c (2b-1) (b-b_target).
```

For an accessible target `b_target>1/2`, the local convergence rate in b is `c(2b_target-1)`, compared with c when optimizing b directly at the same gradient-flow time scale. The polynomial speeds this scalar mode when b_target>1 and slows it when 1/2<b_target<1; at a=-1, b=1/2 is stationary. This is a stylized scalar Euclidean-gradient-flow statement. It does not identify a hidden coefficient with the physical velocity derivative, prove a data-spectral advantage, or describe AdamW training. It gives a counterexample to a blanket claim of beneficial preconditioning.

**Minimal discriminating future comparison:** native fixed-gate K2 versus the collapsed B model, identical starting functions and parameter counts, at the current seed and declared training budget. Retain the same optimizer for empirical relevance, measure training/oracle errors versus both steps and wall time, and describe the result as a parameterization comparison. Do not claim the gradient-flow formula proved the cause of an AdamW outcome. A broader optimizer sweep is unnecessary initially.

If the collapsed model is equivalent or better, there is no current reason to attribute a fixed-gate advantage to an ODE mechanism. If the polynomial trains better in this controlled setting, the next mathematics concerns induced update directions, conditioning and the restricted feasible set, rather than truncation error. Training curves alone would not separate these explanations.

## 3. Spectral restrictions need to target the implemented map

The exponential series converges for every finite matrix. There is no convergence problem that must be fixed by restricting P's eigenvalues. Approximation error, conditioning, feature amplification and training stability are different questions.

### Stable eigenvalues do not control transient growth

Take

```text
P = [[-1, M], [0, -1]].
exp(sP) = exp(-s) [[1, sM], [0, 1]].
```

Both eigenvalues are -1 for every M, but the operator norm can be arbitrarily large. Eigenvalue clipping alone does not address nonnormality. A singular-value cap controls the spectral norm but also restricts useful directions; it is not the same experiment as making singular values equal.

### Even a perfectly norm-preserving generator expands under K2

Let `J=[[0,-1],[1,0]]`, `P=omega J`, and `z=s omega`. Exact exponentiation gives a rotation. But

```text
T2(zJ) = (1-z^2/2) I + z J
T2(zJ)^T T2(zJ) = (1+z^4/4) I.
```

Every nonzero z produces singular values greater than one. Therefore a skew-symmetric P would not establish norm preservation for the implemented K2 layer. Likewise, for scalar `P=-L`, K2 has gain `1-sL+(sL)^2/2`, which exceeds one once `sL>2`, although the exact flow contracts. Generator properties and the finite polynomial's stability region must be checked together.

### An input gate adds another derivative term

The original K2 derivative is

```text
J_F = T2(sP) + (P+sP^2) h grad(s)^T.
```

For exact exponentiation it is

```text
J_F = exp(sP) [I+(P h) grad(s)^T].
```

Even exact exponential contraction at fixed s does not ensure contraction of this input-dependent map. Temperature 10 only gives `||grad(s)|| <= 0.02 ||grad(g)||`; it does not bound the learned gate network's derivative. The surrounding normalization, residual connections and output projection also separate a hidden-layer norm from the full velocity Jacobian.

If a stability direction is reconsidered, first test **excess amplification of field error along trajectories**, not merely a large internal singular value. Compare learned versus oracle `sym(J_v)`, the associated variational propagation, and where drift starts. Some large positive oracle derivatives are necessary in Ring8. The previous late-time tail case is evidence of states leaving high-density regions, not proof that P caused it or that contraction would repair it.

No new P constraint is justified yet. A later constraint must specify whether it targets the polynomial map, full input-dependent Jacobian, or data-space error propagation. These are different objects.

## 4. Why attractive alternative designs remain premature

**Add explicit t to the gate.** The current input h already receives time conditioning through FiLM in every block. An additional t input may improve accessibility or optimization, but is not the first introduction of time awareness. Earlier input-versus-time-only experiments also do not test input-plus-time. A reason to change this would be evidence that a necessary time effect is inaccessible through the existing representation; none is established.

**Add multiple clocks or generators.** One layer's effective matrices are polynomials of the same P, hence commute and lie on a one-dimensional curve as s varies. This is a real restriction. It is not a rank-one restriction on the whole network's input-output Jacobian, nor a demonstrated bottleneck in 2D. More clocks increase flexibility, computation and often capacity, making generic capacity an alternative explanation. First isolate the value or cost of the existing scalar-clock restriction.

**Enforce a symmetric or negative P because the target Jacobian has structure.** Under the current Gaussian-source independent linear path, the optimal data-space Jacobian is symmetric and is a negative scalar identity plus a scaled posterior covariance. Hidden P lives in a different coordinate system and is followed by nonlinearities and other layers. Symmetry or contraction of hidden P does not enforce the desired data-space identity. Moreover, the target requires both expansion and contraction in different positions or directions.

**Replace T2 with exact exp because generation is continuous.** For frozen P, K2 has a measurable semigroup defect:

```text
T2(aP)T2(bP)-T2((a+b)P)
  = ab(a+b) P^3/2 + a^2 b^2 P^4/4.
```

An exact flow eliminates that defect. The regression objective, however, does not penalize this hidden semigroup defect. Existing frozen-checkpoint replacements already show that trained low-order models rely on their native parameterization. Training exp from scratch could answer a different question, but currently lacks a reason to outrank the mechanism controls above.

Even the exact Gaussian generative transport usually cannot be represented as one global scalar clock and one fixed generator. For target covariance eigenvalues lambda_i, the independent linear interpolation has transport gains `r_i(t)=sqrt((1-t)^2+t^2 lambda_i)`. If a smooth representation `R(t)=exp(s(t)P)` with `s(0)=0` held, `R'(0)=-I` would force `s'(0)P=-I`, hence P proportional to I. That cannot produce anisotropic gains for unequal lambda_i. This does not invalidate hidden ODELayers; it invalidates identifying a single hidden layer with the entire physical transport.

**Move directly to a JiT x-prediction experiment.** This would be a useful bridge after selecting the block mechanism, but it is a distinct controlled question. Away from the endpoint, `v=(D-x)/(1-t)` makes velocity regression equal to clean-endpoint regression weighted by `1/(1-t)^2`; unweighted D loss changes the objective. Conversely, defining `D=x+(1-t)f_theta` merely renames the same direct-v network and is not a genuine output-parameterization test. A raw D output changes initialization and inductive bias. Match time sampling, target path, declared loss and solver, and separately document any terminal clamp; do not conflate these changes with the ODE block. In the clamped interval, the unclamped algebraic identity no longer holds without a correspondingly changed regression target.

## 5. Recommended sequence and decision rules

1. **Use existing checkpoints to separate gate tangent from gate curvature.** The centered identity is exact, requires no training, and asks a sharper question than another gate-removal study. If the curvature effect is negligible across the declared inputs and times, do not motivate K2 by a nonlinear scalar-clock benefit.
2. **Choose one controlled training question, not both by default.** If parameterization is the remaining explanation, use function-matched fixed-K2 versus free-B. If clock curvature has a material, interpretable effect, compare native and gate-linearized K2 under a declared protocol. These answer different questions and should not be pooled into an architecture tournament.
3. **Only then choose a design change.** Multiple clocks, spectral structure or exact exponentiation should follow a measured failure of the original restriction. Otherwise they increase flexibility without explaining the professor's original block.

A credible mathematical contribution would connect an identified operator restriction to approximation or optimization on a stated target class, then test the predicted boundary of that argument. Algebraic resemblance to an ODE is insufficient. At present the evidence supports a study of constrained input-dependent operators; it does not yet establish that this restriction is preferable to SwiGLU or specifically suited to flow matching.

## Numerical checks

[verify_new_math.py](verify_new_math.py) checks the centered identity, its derivative by finite differences, the skew-generator counterexample and the K2 semigroup defect. [new-math-checks.json](new-math-checks.json) records errors: algebraic identities below 5.5e-16 and the finite-difference derivative below 9.8e-12. These checks supplement the displayed arguments and are not new training results.
