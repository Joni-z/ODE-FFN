# Diagonal latent flow maps: prior art and the claim that remains testable

Reviewed 2026-10-24. This is a bounded primary-source review and algebraic critique after the failed Stage17 shared-feature study. It does not reopen that family or authorize implementation/training. The operative proposal below uses the **fixed** clock `s(z)=.2+.8*sigmoid(z)`, two learned channelwise eigenvalue vectors, and `|lambda|<=1`. It has no learned clock slopes, biases or temperature.

The proposed computation is a coherent way to spend less on internal matrix products and more on expanded features. Its exact function is nevertheless an MLP with a learned coordinatewise activation. Neither that form nor using scalar ODE transformations as activations is new. The potentially useful claim is narrower: this particular constrained activation could deliver a better quality/cost tradeoff than plain SiLU, a generic adaptive activation, and practical SwiGLU baselines. Current literature does not establish that claim for our flow-matching tasks.

## Three directly relevant primary sources

### 1. KAT: Kolmogorov–Arnold Transformer — ICLR 2025

[Primary full text, Sections 4.2–4.4, Eqs. 12 and 15–17](https://arxiv.org/html/2409.10594v1); [authors' repository confirming the venue](https://github.com/Adamdad/kat).

KAT uses safe rational functions `F(x)=P_m(x)/(1+|Q_n(x)|)` and shares their coefficients across groups of input channels. Its exact rewrite is `GR-KAN(x)=W F(x)`: a coordinatewise learned function followed by a linear map. The two-layer transformer replacement initializes its first rational activation near identity and the second near Swish. The paper itself describes the resulting module as a specialized MLP and notes that its optimized activation remains slower than standard activations.

**Relevance:** inexpensive learned univariate functions around dense projections are established FFN design territory. A rational activation is a relevant generic comparator if broad activation superiority is claimed. KAT's grouping, input activation placement, rational family and initialization differ from this proposal; its vision results do not predict our FM result. Similar dense MAC counts do not establish similar runtime.

### 2. DiTAC: Trainable Highly-expressive Activation Functions — ECCV 2024

[Official proceedings record](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/7743_ECCV_2024_paper.php); [primary PDF, Sections 3.1–3.3, Eqs. 1–2](https://arxiv.org/pdf/2407.07564).

DiTAC integrates one-dimensional continuous piecewise-affine velocity fields to obtain scalar diffeomorphisms `T_theta`. These transformations act elementwise on feature values. Its main activation is `T_tilde(x)*Phi(x)`, with `T_tilde=T_theta` inside a specified interval and identity outside; this is not simply `GELU(T_theta(x))`. The paper discusses parameter sharing and lookup-table computation. Its underlying scalar transformation is an actual fixed-time flow of its state-dependent vector field.

**Relevance:** the idea of an ODE-derived learned scalar activation already has direct prior art. Our low-parameter exponential formula is different from its CPAB family. Its monotonicity argument does not automatically transfer to an input-dependent stopping time, and its experiments supply no FM-objective guarantee for our block. The full PDF was checked because the arXiv HTML omitted its method section.

### 3. ACON: Activate or Not: Learning Customized Activation — CVPR 2021

[Primary PDF, Section 3, Eq. 4](https://arxiv.org/pdf/2009.04759); [authors' repository confirming the venue](https://github.com/nmaac/acon).

ACON-C is `(p1-p2)*z*sigmoid(beta*(p1-p2)*z)+p2*z`, with channelwise parameters. It includes Swish as a special case and learns the linear/nonlinear mixture and asymptotic slopes. This older paper is retained for an exact algebraic match, not as a performance recommendation: the proposed K1 control belongs to this function family away from its identity limit. The coordinate change required for that match becomes singular at zero eigenvalue, so ACON's original parameterization and optimization are not identical to ours. No CNN performance result is imported.

## Exact reduction of the proposed block

The following are project-specific deductions, not theorems claimed by the papers above. Let `E:d→r`, `R:r→d`, and let `b_E,b_R` denote optional affine biases. Define each scalar map separately for each channel:

```text
s(z)       = .2 + .8*sigmoid(z)
T_exp(z;λ) = z*exp(λ*s(z))
T_K1(z;λ)  = z*(1 + λ*s(z))
T_K2(z;λ)  = z*(1 + λ*s(z) + .5*(λ*s(z))²)

F(h) = R T2(SiLU(T1(Eh+b_E))) + b_R.
```

Everything between E and R is coordinatewise. Thus, exactly,

```text
psi_i = T2_i composed with SiLU composed with T1_i
F(h)  = R psi(Eh+b_E) + b_R.
```

This is a learned-activation MLP. E and R provide all cross-coordinate mixing within this FFN. Calling the two scalar maps ODE layers does not add a distinct function class beyond that expression. Conversely, `psi` is generally **not exactly a standard one-parameter Swish**: its two adaptive maps can change the curve beyond a single slope parameter. “Not exactly Swish” is a much weaker novelty claim than “new ODE architecture.”

At both eigenvalue vectors zero, every listed map is identity, so the complete FFN is exactly the ordinary same-width SiLU FFN. This holds for arbitrary E/R and their biases; it is not a large-width approximation.

For a static clock `s=c=.6`, each map is a constant diagonal scaling D. For K2, `D_ii=1+c*lambda_i+.5*(c*lambda_i)^2`. Then

```text
R D2 SiLU(D1(Eh+b_E)) + b_R
  = R' SiLU(E'h+b'_E) + b_R,
E'=D1 E, b'_E=D1 b_E, R'=R D2.
```

The static control changes parameterization, not the unconstrained function family. Its extra scales may still change Adam updates, regularization and conditioning; functional absorbability does not imply identical training trajectories.

## The generic activation control is exact, not cosmetic

K1 can be written without any ODE language:

```text
T_K1(z;λ) = (1+.2λ)*z + .8λ*z*sigmoid(z).
```

In ACON's notation set `p2=1+.2λ`, `p1=1+λ` and fix the complete sigmoid argument coefficient `k=beta*(p1-p2)=1`. For nonzero λ, this requires `beta=1/(.8λ)`. The smooth K1 identity point at λ=0 is a limit of those ACON coordinates, not a finite-beta ACON point. It is therefore precise to call this an **exact constrained ACON-type function with a different regular parameterization**. Setting the published ACON beta to 1 would not produce the proposed K1 control.

For fixed s, the three adaptive maps satisfy

```text
T_K1(z;0) = T_K2(z;0) = T_exp(z;0) = z
partial T / partial z at λ=0 = 1
partial T / partial λ at λ=0 = z*s(z).
```

With identical E/R, surrounding parameters and eigenvalue coordinates, they therefore have identical initial functions and full parameter Jacobians, hence identical initial NTKs. This includes both maps and the middle SiLU. If `lambda=tanh(rho)`, the equality persists because all use the same coordinate derivative at rho=0. It says nothing about finite training trajectories or superiority of the exponential.

This makes K1 the necessary generic-activation comparison for an exponential-specific claim. A K1 gain alone would support cheap adaptive feature nonlinearities, not an advantage of an exponential solve or higher terms. A separate rational implementation is not needed to establish this narrow comparison; it becomes relevant to a much broader claim of best learned activation.

## What the ODE interpretation does and does not buy

For fixed initial z, `T_exp` is the exact solution of `dy/du=lambda*y`, evaluated at `u=s(z)`. The stopping time is chosen from the initial value and frozen during that solve. It is not the exact solution of `dy/du=lambda*s(y)*y`, where the clock would change along the trajectory. Re-evaluating s after each application generally breaks the linear flow's semigroup identity. Hidden feature time is also distinct from the external FM generation time.

The derivative includes a clock term. Write `sigma=sigmoid(z)`:

```text
T_exp'(z) = exp(lambda*s(z)) * [1 + .8*lambda*z*sigma*(1-sigma)].
```

An elementary maximization gives `max_z |z*sigma'(z)| = .223871602347…`; the positive maximizing z solves `z*tanh(z/2)=1`. Therefore `|lambda|<=1` gives a global lower derivative bound `exp(-1)*(1-.8*.223871603) > .30199`. Each exponential scalar map is strictly increasing and onto. For K2,

```text
T_K2'(z) = 1+lambda*s+.5*(lambda*s)²
           + .8*lambda*z*sigma'(z)*(1+lambda*s).
```

The same bound gives `T_K2' >= .5-1.6*.223871603 > .14180` when `|lambda|<=1`. These are valid global scalar-map statements for the fixed clock and stated bound. They do not establish that the complete FFN is invertible: the middle SiLU is nonmonotone, and E/R change dimensionality. Nor do they guarantee a well-conditioned complete velocity Jacobian.

K1 does not share this global monotonicity over the whole eigenvalue interval. Its derivative can be negative for sufficiently negative lambda (approximately below −.92604). At lambda=−1 its minimum is about −.07987. Thus an eventual K2/exponential advantage over K1 could concern monotonic scalar warping, finite curvature or optimization. It would not by itself demonstrate that hidden ODE semantics improve flow matching. Even with monotone T1/T2, the scalar composite is a monotone input/output deformation of SiLU with the same single critical point, rather than an arbitrary learned univariate function.

## Compute accounting and the scope of a clean comparison

The dense matrix work is `2dr`, and there are `2r` learned eigenvalue parameters. At d=64 and r=128, dense MACs are 16,384 and, including affine biases, parameters are 16,832. The exact same-width SiLU baseline has 16,576 parameters. A bias-enabled SwiGLU has `3dm` dense MACs and `3dm+2m+d` parameters: at widths 85/96/128 these are respectively 16,320/18,432/24,576 MACs and 16,554/18,688/24,896 parameters.

Thus width85 approximately matches matrix work; the practical Swi96/128 controls have different declared budgets. Do not call them exact MAC matches. All diagonal variants also perform clock sigmoids and extra pointwise arithmetic; exponential maps add exponentials. Their arithmetic/memory and backward costs must be counted separately from dense MACs. Reduced matrix work alone is not a measured throughput improvement.

Relative to the old dense-P architecture, this is **not one isolated algebraic factorization**. It changes the latent width, location of mixing, controller family and solve map. Those changes can be organized under one coherent compute-allocation hypothesis, but an old-parent comparison cannot isolate their causes. Within the new `R psi(Eh)` family, the declared same-width SiLU, static K2, adaptive K1, adaptive K2 and adaptive exponential controls separate the proposed claims far more cleanly. The old family need not be rescued by temperature, order or width sweeps.

The review supports the following interpretation rules, not a launch recommendation:

- Beating an old dense-P arm is insufficient: plain wider SiLU could explain the gain.
- Beating plain SiLU but not its static scaling control does not establish useful input adaptivity.
- Beating those controls with K1 establishes a generic learned-activation result; it does not establish an ODE-specific contribution.
- K2/exponential must survive K1 and the practical SwiGLU controls before a stronger claim is considered. Quality and actual compute must both be reported. Equality of the initial NTK improves the comparison but is not a theorem about final quality.
- No source or existing tangent measurement supplies a new physical-time FM guarantee, and none predicts this family's success on the synthetic tasks or JiT transfer.

## Search scope and verification limits

Searches included `Kolmogorov Arnold Transformers KAT rational activation grouped rational`, `activation functions ODE 2025`, `activation diagonal flow`, `flow-based activation`, `activation function exponential sigmoid 2024`, and `activation flow diagonal site:arxiv.org 2026`. Primary KAT HTML, DiTAC PDF and ACON PDF were read for the equations above. KAT's OpenReview record triggered a browser-verification page; its authors' repository confirms the ICLR2025 venue. DiTAC's official ECCV record confirms its venue. No exact prior occurrence of the fixed-clock two-map exponential formula was verified in this bounded search. That absence is not a novelty proof. The most consequential verified prior art is already sufficient to rule out claiming that ODE-derived scalar activations, or learned coordinatewise FFNs, are themselves new.
