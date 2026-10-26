# Diagonal-flow activation family: design review

**Conditional PASS as a coherent budget-allocation family with internal controls.** This changes the old dense ODEFFN substantially: expanded features, diagonal generators and local clocks replace its dense inner operators and MLP clocks. It is not a pure one-factor swap. Comparing members against the same expanded SiLU FFN makes the new question interpretable. The failed shared-feature axis remains stopped. This review does not run training or claim predicted success.

Prefer the fixed-clock refinement over the initial learned-slope proposal. It removes an initialization choice and inactive slope parameters, matches eigenparameter counts across controls, and allows exact initial tangent matching. Longer proofs, costs and the learned-slope proposal are preserved in [diagonal-flow-math-notes.md](diagonal-flow-math-notes.md).

## Operative family

```text
z = E h + bE
H(h) = R F2(SiLU(F1(z))) + bR
F(z) = Q(lambda*s(z))*z                 (coordinate-wise)
s(z) = .2+.8 sigmoid(z)
lambda = beta*tanh(eta), eta=0 initially.
```

Q is K1, K2 or exp. Keeping tau=10 means fixing the clock logit to 10*z; the effective scale is one. This is not the old learned clock. Each clock is evaluated once per solve. Exp is exact for the frozen conditional linear diagonal system, not the nonlinear ODE that continuously recomputes its clock.

E/R supply feature mixing and expansion; the two solves and middle SiLU retain the professor's composition. Equally, this is an ordinary expanded FFN with learned coordinate activation F2(SiLU(F1(z))). Its ODE interpretation alone gives no FM-specific advantage.

## Verified bounds and limitations

The required inequality is valid:

```text
sigma'(u)=1/[4cosh(u/2)^2]
cosh(u/2)^2 >= 1+u^2/4 >= |u|
=> |u sigma'(u)|<=1/4, hence |z s'(z)|<=.2.
F'(z)=Q(lambda*s)+lambda*z*s'(z)*Q'(lambda*s).
```

The actual maximum of |u sigma'(u)| is about .22387160. For the proposed beta=1, exp has derivative between .8/e and 1.2e; K2 has derivative between .1 and 2.9. The K2 lower bound follows from Q2(y)>=.5 and the clock term bounded by .2*beta*(1+beta). These individual solves are globally increasing and bi-Lipschitz.

K1 is **not** uniformly monotone at beta=1: at lambda=-.99 near z=2.39936 its derivative is approximately -.06907. Thus the arms do not have identical shape/stability restrictions; their comparison is not purely numerical approximation accuracy.

The full state Jacobian is R diag(F2') diag(SiLU') diag(F1') E. SiLU is not globally monotone, and unrestricted E/R remain in its Lipschitz bound. No complete-network invertibility, contraction or generative stability follows. For the initial learned-slope proposal the same first-derivative bound holds without gate bias, but Fexp''(0)=.4*lambda*(a/10)*exp(.6lambda) can grow without bound with a. Fixed clocks avoid that additional freedom.

## Initialization and the exact static control

At eta=0 every flow is the identity, giving exactly R SiLU(Eh+bE)+bR, including projection biases. Adaptive K1/K2/exp also have identical complete initial parameter Jacobians and NTKs because Q(0)=Q'(0)=1. Their added eta directions are generally nonzero; plain SiLU has no such parameters and is not tangent-equivalent. Verify backbone statements with a common nonzero probe head, since the formal zero velocity head otherwise masks them.

For static c=.6, put Dj=diag(Q2(c*lambda_j)). Then

```text
R D2 SiLU(D1(Eh+bE))+bR = R' SiLU(E'h+bE')+bR'
E'=D1E; bE'=D1bE; R'=RD2; bR'=bR.
```

Static K2 and base SiLU have exactly the same function class with unconstrained projections; the reverse inclusion uses lambda=0. Static eigenparameters duplicate scaling directions but can change optimization. This is the matched static contrast for K2, not an identical static optimizer for exp.

## What adaptive and higher-order terms add

The adaptive second-flow eta direction at initialization contains q(z)=s(SiLU(z))*SiLU(z). For one scalar neuron it is outside the ordinary parameter tangent span {1,phi,phi',z*phi'}. At negative infinity, q=.6*z*exp(z)+O(z^2 exp(2z)); matching coefficients forces its putative span representation to be .6*phi. But q/phi tends to 1 at positive infinity. This is a new one-neuron functional direction shared by K1/K2/exp, not a proof about full wide networks, optimization or FM advantage.

K1 is exactly a generic adaptive activation:

```text
F_K1(z)=(1+.2lambda)z+.8lambda*SiLU(z).
```

It is ACON-type. In the [official ACON-C code](https://github.com/nmaac/acon/blob/main/acon.py), choose p1-p2=.8lambda, p2=1+.2lambda and switching parameter 1/(.8lambda) for nonzero lambda. Those ACON coordinates are singular at lambda=0; this is function equivalence, not identical regular initialization or optimization. K1 success alone cannot establish a higher-order ODE benefit.

K2 introduces a concrete extra feature:

```text
F_K2(z)=(1+.2lambda+.02lambda^2)z
        +(.8lambda+.48lambda^2)SiLU(z)
        -.32lambda^2*z*sigmoid'(z).
```

The last term is odd and localized. With T=tanh(z/2), the individual-flow odd parts are

```text
K1: z*(1+.6lambda)
K2: z*[1+.6lambda+.5lambda^2*(.36+.16T^2)]
exp: z*exp(.6lambda)*cosh(.4lambda*T).
```

All identities are exact. They distinguish single-flow feature families, not full-network capabilities. The complete K1–SiLU–K1 sandwich already produces nonlinear odd functions. Central symmetry of the targets does not establish higher-order superiority. In fact, the scalar bimodal teacher's local Euler-map gain is larger near the center than in the tails, whereas these K2/exp odd-part gains increase toward the tails. Projections may combine or reverse effects, but the one-flow identity does not prove target alignment. Any symmetry-based explanation would need a later asymmetric evaluation boundary.

## Necessary controls and decision

The proposed family needs plain SiLU at the same r, static K2, adaptive K1/K2/exp, and both established SwiGLU budgets. These separately test expansion alone, redundant parameterization, generic adaptation, higher-order terms and practical usefulness. Static and adaptive arms add the same two r-vectors. Counts are 2dr+r+d for base SiLU and 2dr+3r+d for flow variants: at d=64,r=128, 16576 versus16832 parameters and16384 dense MACs. SwiGLU128 has16554 parameters and16320 MACs. Pointwise work differs; actual speed must be measured.

Pair projections, biases, backbone, FiLM and runtime buffers. Keep one fixed clock/bound and common eigenparameterization; do not add learned slopes, gate biases or another width/order/temperature rescue. Comparing to historical dense ODE results cannot isolate the several old components replaced here.

A higher-order candidate must beat plain/static/generic K1 controls and the established SwiGLU references under the registered practical rule. Improvement over a failed old ODE alone is insufficient. If only static or K1 helps, retain that narrower explanation. If K2/exp fail, stop this family. An initial passing screen would only nominate subsequent confirmation and a new holdout.

**Defensible design claim:** reallocate dense hidden ODE computation into expanded features, then test whether cheap diagonal conditional flows supply useful constrained activation shapes beyond ordinary and generic adaptive FFNs. The construction and controls are coherent. Its usefulness remains unproved.
