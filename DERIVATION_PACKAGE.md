# Derivation Package: native gate feedback

Zhizhe Zhang · October 28, 2026

## Target

Derive the complete differential of the professor's local polynomial map and its place in the native FFN and fixed Heun computation. Identify what a bound on P does and does not control. The empirical target remains useful image generation; the measurable local derivatives below are diagnostic slices, not substitutes for that target.

## Status

COHERENT AFTER REFRAMING / EXTRA ASSUMPTION. The identities and conditional bounds are coherent for real-valued differentiable maps. They do not explain the observed FID change or establish a new architecture. The [registered native-checkpoint analysis](docs/autoresearch/image-gate-feedback/protocol.md) will measure these quantities without changing any forward model.

## Invariant Object

The complete derivative of the native computational map. Its top-level form is the derivative of the fixed sampler S_theta from Gaussian input to output. A local ODELayer Jacobian is one factor inside that computation. Neither P alone nor an anchored local derivative is the sampler's Jacobian.

## Assumptions

- Work in real arithmetic with the stored fitted weights. Numerical diagnostics use their float64 extension at states captured from actual float32 forwards.
- External FM time t is fixed when differentiating a hidden state h. The hidden state can already contain time information through upstream conditioning.
- Each local clock s(h) is scalar and differentiable; P is a fixed d-by-d learned matrix. Native K is 1 or 2. The clock is evaluated once at the layer input.
- The FFN is F2(SiLU(F1(h))), with dropout inactive. No hidden normalization is omitted inside the two native ODELayer maps.
- Inverse-based determinant formulas require an invertible base matrix. No invertibility or contraction is assumed for the complete predictor.
- Local numerical geometry does not by itself order regression risk, sample quality or FID.

## Notation

T_K(A) = sum from k=0 to K of A^k/k!; T_0(A)=I.
F(h)=T_K(s(h)P)h. Define M=T_K(sP), g=grad_h s and v=P T_(K-1)(sP)h.
All vectors are columns. Outer products are written v g^T. Matrix norms are spectral norms and vector norms are Euclidean.
For the whole model, D_theta(z,t) predicts the clean image, delta(t)=max(1-t,.05), and f_theta(z,t)=(D_theta(z,t)-z)/delta(t).

## Derivation Strategy

Differentiate the actual input-dependent polynomial before taking a constant-clock slice. Then compose the two native maps and the fixed sampler. Keep identities, conditional bounds and possible interpretations separate.

## Derivation Map

1. The product rule gives the complete local state derivative and the noncommuting P differential.
2. A rank-one update identity gives a conditional volume formula; an explicit bounded-clock example refutes a P-only contraction guarantee.
3. The chain rule gives the complete FFN derivative and a rank-at-most-two feedback correction.
4. The chain rule through Heun states shows why local measurements cannot replace endpoint sensitivity or FID.

## Main Derivation

### 1. Local state and parameter differentials — identities

Since dT_K(sP)/ds = P T_(K-1)(sP),

```text
J_h F = M + v g^T.

K1: M=I+sP,             v=Ph.
K2: M=I+sP+.5s^2 P^2,  v=Ph+sP^2h.
```

Holding h fixed, the clock has no direct dependence on the layer's P. For a matrix perturbation E,

```text
D_P F[E] = sum(k=1..K) s^k/k! sum(j=0..k-1) P^j E P^(k-1-j) h.
```

The factors cannot generally be commuted. For K2 this is `sEh + .5s^2(PE+EP)h`. The finite K2 difference also has the remainder `.5s^2 E^2 h`. P is a fitted parameter, not an identified physical generator with a known estimation error.

Conditional norm bounds are:

```text
||J_h F|| <= ||M|| + ||v|| ||g||,
||D_P F[E]|| <= |s| T_(K-1)(|s| ||P||) ||E|| ||h||.
```

For the native clock, write `u=W1h+b1`, `a=SiLU(u)`, `q=(w2^T a+b2)/tau`, `p=sigmoid(q)`. Then

```text
g = scale/tau * p(1-p) * W1^T [w2 * SiLU'(u)].
```

Here `*` denotes elementwise multiplication. Since `|SiLU'(u)| <= 1+1/e`, one sufficient bound is `||g|| <= scale/(4 tau) * (1+1/e) ||W1|| ||w2||`. A bounded clock value alone does not bound its gradient uniformly over unrestricted gate weights.

### 2. Volume and a counterexample — conditional proposition and exact example

When M is invertible, the matrix determinant lemma gives

```text
det(J_h F) = det(M) * [1 + g^T M^(-1) v].
```

The bracket is not determined by P's eigenvalues. Nor does a small bracket or a large condition number automatically imply poor denoising.

For a concrete counterexample take `P=-.5I`, `h=(5,0,...)` and `s(h)=.2+.8 sigmoid(4h1-20)`. At this point `s=.6` and `grad s=(.8,0,...)`. Every singular value of P is .5 and every eigenvalue is -.5, but the first diagonal entry of the full K1 Jacobian is

```text
1 + .6(-.5) + (-.5*5)(.8) = -1.3.
```

The fixed-value base matrix is `.7I`; its contraction does not survive the clock derivative. This gate belongs to the original two-layer SiLU class: `SiLU(a)-SiLU(-a)=a` lets two hidden units implement the affine logit `40h1-200`, with tau10. Increasing that slope gives unbounded full local gain while P and the clock range stay fixed. This is a class-level example, not evidence that fitted checkpoints reach it.

Eigenvalue limits alone also fail to bound a nonnormal P: the matrix `[[0,L],[0,0]]` has only zero eigenvalues and arbitrary norm. Equal singular values alone do not fix their scale. These facts motivate measuring the complete map; they do not select a spectral penalty.

### 3. Complete FFN feedback — identities

Let `u=F1(h)`, `z=SiLU(u)`, `A=diag(SiLU'(u))`, and `Jj=Mj+vj gj^T`. Then

```text
J_FFN = J2 A J1.
R     = M2 A M1.
J_FFN - R = (J2 A v1) g1^T + v2 (g2^T A M1).
```

Thus the derivative correction has rank at most two. R retains the native clock values at this input while suppressing their derivatives algebraically. R is not the Jacobian of the earlier globally mean-collapsed model. No forward inference ablation is performed by calculating it.

For an ordinary SiLU FFN, `J=W2 diag(SiLU'(W1h)) W1`. For SwiGLU, with affine branches a and b, `J=W3[diag(b*SiLU'(a))Wa + diag(SiLU(a))Wb]`. Both ordinary families already receive time-conditioned hidden inputs; SwiGLU already has state-dependent gates.

### 4. The native sampler — identities and an unmeasured quantity

At each fixed time, `J_f=(J_D-I)/delta(t)`. For Heun step size a=1/128, trial state y=z+a f(z,t),

```text
Psi(z) = z + a/2 [f(z,t) + f(y,t+a)],
J_Psi = I + a/2 [J_f(z,t) + J_f(y,t+a)(I+a J_f(z,t))].
J_S   = ordered product of all 128 step Jacobians.
```

Attention, normalization, residual modulation and readout enter J_D. The proposed local measurements do not evaluate J_S. Products of local norm bounds can be loose and do not decompose FID.

## Remarks and Interpretation

- Separating between-time, between-image and within-image token variation is an exact finite-array variance identity. It is not a causal separation of time conditioning and image feedback.
- Differences between paired-data and generated hidden states concern the declared finite probe cohort. They do not certify an input distribution or identify the cause of the fixed-clock FID change.
- Small singular values and local volume contraction can be compatible with denoising. Invertibility, contraction and determinant sign are not quality objectives by default.

## Boundaries and Non-Claims

No new architecture, loss, clock calibration or training is proposed here. No stopped frozen-clock replacement is rescued. The local formulas do not establish that gate derivatives caused the native K1 lead, that P is unstable, or that a particular constraint will improve sampling.

## Open Risks

The global endpoint effect remains unexplained. The native study must preserve the original sampling paths, distinguish float32 captures from promoted derivatives, retain ordinary controls and report all fixed probes. A local geometric observation can guide the next question only within that scope.
