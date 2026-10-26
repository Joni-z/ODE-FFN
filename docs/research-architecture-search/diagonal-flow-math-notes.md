# Review of a diagonal-flow activation sandwich

This is a mathematical design review, not an experiment registration or a recommendation to launch training. No model code was changed. The completed [shared-feature study](analysis-shared-features/analysis-report.md) failed the practical screen, including its larger untied controller. That axis remains stopped.

## Judgment

The proposed expanded diagonal family is more coherent than adding another change to the failed dense hidden block. Its question is whether the ODELayer idea is useful as a **cheap, constrained learned activation**, while ordinary input/output projections provide feature mixing and expansion. There is a concrete ordinary-FFN reference, exact initial function matching, and a static control with the same function class as that reference.

However, this is a substantial refactor of the professor's block: it introduces an expanded basis, diagonal generators and local clocks while removing dense inner generator/controller computation. It is not a one-factor change to the old ODEFFN. The experiment can become interpretable by comparing members of this new family against the same expanded SiLU FFN, rather than attributing a comparison against the old model to one cause.

Of the two proposals, the **fixed local-clock version is preferable for a first design review**. It removes learned clock slopes and their initialization choice, equalizes the eigenparameter count across the substantive controls, and permits adaptive K1/K2/exp to have the same complete initial tangent kernel. Its bounds are valid. These facts make a bounded comparison defensible; they do not predict that it will beat SwiGLU or establish an FM-specific reason to prefer it.

## 1. The family and its relation to the original layer

Let E map d hidden features into r expanded features, and R map them back. Include the ordinary projection biases explicitly:

```text
z = E h + bE
y = F1(z)
w = SiLU(y)
H(h) = R F2(w) + bR.
```

Each F acts independently by coordinate:

```text
F_i(z_i) = Q(lambda_i*s_i(z_i))*z_i
lambda_i = beta*tanh(eta_i).
```

The choices are `Q1(y)=1+y`, `Q2(y)=1+y+y^2/2`, and `Qexp(y)=exp(y)`. For each input, s is evaluated once and held fixed inside Q. These are respectively the first and second Taylor polynomials, or exact solution, of the **frozen conditional linear feature system** with diagonal generator. The exponential is not the exact solution of the different nonlinear ODE `dy/du=lambda*s(y)*y`, where the clock is recomputed continuously.

The two feature solves and middle SiLU retain the professor's composition. E/R now perform feature mixing, whereas the original P matrices mixed hidden channels inside both solves. The new operator is also a conventional expanded FFN with a coordinate-specific learned activation

```text
A_i(z)=F2_i(SiLU(F1_i(z))).
```

This ordinary-activation description is equally exact. It prevents treating a hidden solve as a generation-time integration step or claiming that the architecture must suit flow matching because of its name.

## 2. The sigmoid bound is valid

For sigmoid sigma,

```text
sigma'(u) = 1/[4 cosh(u/2)^2].
```

Using `cosh(u)>=1+u^2/2` gives

```text
cosh(u/2)^2 = [cosh(u)+1]/2
              >= 1+u^2/4
              >= |u|.
```

Therefore `|u sigma'(u)|<=1/4` globally. This is conservative. The actual maximum occurs at positive u satisfying `u*tanh(u/2)=1`, approximately 1.54340464, and is approximately 0.22387160. These numerical values supplement the analytic inequality.

For the initial learned-slope proposal,

```text
s(z)=.2+.8 sigma(a*z/tau),    tau=10,
z*s'(z)=.8*u*sigma'(u),      u=a*z/tau,
```

so `|z*s'(z)|<=.2`, independent of a, and `.2<s(z)<1`. The absence of an independent clock bias is material: with a bias in u, z*s' contains `(u-b/tau)*sigma'(u)` and the same parameter-uniform bound no longer follows.

The fixed-clock refinement uses `s(z)=.2+.8 sigma(z)`, which is the same expression with the **fixed** clock logit `t_func(z)=10*z` and tau=10. This preserves the displayed original temperature but fixes the effective logit scale to one. It is not the original learned two-layer clock network.

## 3. State derivatives and monotonicity of individual solves

For one coordinate, put `y=lambda*s(z)`. The general derivative is

```text
F'(z)=Q(y)+lambda*z*s'(z)*Q'(y).
```

This includes the clock derivative. Treating s as constant when differentiating would miss the term needed for the following results.

### Exact exponential

For `|lambda|<=beta`,

```text
Fexp'(z)=exp(lambda*s(z))*[1+lambda*z*s'(z)]
exp(-beta)*(1-.2beta) <= Fexp'(z)
                          <= exp(beta)*(1+.2beta).
```

Any beta<5 therefore supplies a sufficient uniform positive derivative bound. Each finite-parameter scalar map is globally increasing and bi-Lipschitz, and tends linearly to both infinities. For the refinement's proposed beta=1, the conservative derivative interval is `[.8/e,1.2e]`. This verifies that proposed value's mathematical sufficiency; it does not select it from data or authorize an experiment.

### Second-order polynomial

Since `Q2(y)=((1+y)^2+1)/2>=.5` and `|1+y|<=1+beta`,

```text
F2'(z)=Q2(y)+lambda*z*s'(z)*(1+y)
F2'(z)>=.5-.2beta*(1+beta)
F2'(z)<=1+beta+beta^2/2+.2beta*(1+beta).
```

At beta=1 this gives `0.1<=F2'(z)<=2.9`. Thus the proposed K2 solve is also globally increasing and bi-Lipschitz. Q2 itself is positive for every real y, but positivity of its value alone would not prove monotonicity when the clock depends on z.

### First-order polynomial: an important difference

For K1,

```text
F1'(z)=1+lambda*[s(z)+z*s'(z)].
```

The bound `1-beta*(1+.2)` is negative at beta=1. This is not merely a loose-proof artifact. For the fixed clock,

```text
s(z)+z*s'(z)=.2+.8*SiLU'(z),
max = approximately 1.07987146.
```

At lambda=-.99 and the maximizer z approximately 2.39936, F1' is approximately -0.06907. Therefore adaptive K1 with the same proposed beta is **not globally monotone**, although its value multiplier remains positive for finite eta and finite z. It is still a valid generic activation control, but the arms are not equally monotonicity-constrained. A later difference cannot be described solely as better integration accuracy; their allowed activation shapes and parameter geometry differ too.

## 4. The complete FFN is not certified monotone or globally stable

For the full sandwich, with diagonal derivative matrices,

```text
J_H = R diag(F2'(w)) diag(SiLU'(y)) diag(F1'(z)) E.
```

If the scalar solve derivative bounds are M1,M2 and the SiLU Lipschitz constant is Lphi,

```text
Lip(H) <= ||R||_2 * M2 * Lphi * M1 * ||E||_2.
```

A simple rigorous bound is `Lphi<=1.25` from the sigmoid inequality above; its actual value is approximately 1.09983932. E and R remain learned and unrestricted, so bounding diagonal eigenparameters does not bound the complete model by a small predetermined constant. An outer residual adds another derivative term, and the network's other blocks and projections remain relevant.

SiLU is not globally monotone. Even when both feature solves are increasing, their sandwich with SiLU is not an increasing scalar function everywhere. Under positive solve derivatives, the sign of the composite activation's derivative follows SiLU's derivative at the transformed input. The activation has the same single-negative-minimum pattern as SiLU, with its location and shape altered; it does not acquire arbitrary oscillations. Dense E/R and dimension changes provide no end-to-end invertibility or monotonicity guarantee.

The learned-slope proposal illustrates another boundary: despite the uniform first-derivative bound, its exponential solve has

```text
Fexp''(0)=.4*lambda*(a/tau)*exp(.6lambda).
```

This curvature can grow without bound with a. The fixed-clock refinement removes that free slope, which is a concrete reason to prefer it over stacking learned clock slopes onto the initial comparison. Neither version's first-derivative proof is a guarantee about generative solver errors.

## 5. Exact initial functions, gradients and tangent kernels

Set both eta vectors to zero. Every Q above has Q(0)=1, hence

```text
H(h)=R SiLU(Eh+bE)+bR
```

exactly, including the ordinary projection biases. E/R and every surrounding parameter can be copied from one base SiLU model, giving the same complete initial functions and input Jacobians. A nonzero probe output head is needed to check nontrivial backbone derivatives when the real training head is initialized to zero.

For the learned-slope exponential proposal,

```text
partial F/partial lambda = s(z)*exp(lambda*s(z))*z
partial F/partial a = (.8/tau)*lambda*exp(lambda*s(z))*z^2*sigma'(a*z/tau).
```

Thus eta responses at eta=0 are generally nonzero, scaled by beta, while a responses are zero. Choosing a=10 makes the initial clock sigma(z) despite keeping tau=10; choosing a=0 gives a constant initial clock. These choices leave the initial forward function unchanged but alter the eta feature directions and therefore the initial tangent kernel. That is why “same initial function” alone would not make the first proposal's learned-clock initialization innocuous.

The fixed-clock refinement avoids a and its choice. Since all three adaptive Q functions satisfy Q'(0)=1, **adaptive K1, K2 and exp have the same complete initial parameter Jacobian and tangent kernel**, with identical E/R, eta, beta, clocks and surrounding tensors. Their extra eta directions differ from those of the base SiLU FFN; the base and adaptive arms are not tangent-equivalent.

In lambda coordinates, K2 and exp also agree through second order at zero; K1 lacks that second-order term. Exp and K2 first differ at cubic order. These are parameter-function identities, not statements about numerical approximation of the external FM trajectory. The common tanh parameterization preserves their initial first-order equality and introduces saturation of eta gradients away from zero; it does not prove a favorable optimizer geometry.

For one channel at initialization, with z=Eh+bE and p=SiLU(z), the eta feature directions of the full FFN are

```text
partial H/partial eta1_i
 = beta*R[:,i]*SiLU'(z_i)*s(z_i)*z_i
partial H/partial eta2_i
 = beta*R[:,i]*s(p_i)*p_i.
```

Downstream network Jacobians multiply these expressions in the full velocity model. K1/K2/exp share them exactly.

## 6. Static diagonal clocks are an exact ordinary-FFN control

For a constant clock c=.6, let

```text
D1=diag(Q2(c*lambda1)),  D2=diag(Q2(c*lambda2)).
```

Then the static sandwich is

```text
R D2 SiLU(D1(Eh+bE))+bR
 = R' SiLU(E'h+bE')+bR',

E'=D1 E, bE'=D1 bE, R'=R D2, bR'=bR.
```

Every static model is therefore a base SiLU FFN, and every base model occurs at lambda1=lambda2=0. Their function classes are exactly equal when the ordinary E/R parameters are unconstrained. This statement includes input and output biases. The bounded eigenparameter range does not break equality because the identity choice is available.

At initialization the static eta directions are also redundant with existing E/bE/R parameter directions. For example, the first static direction is c times the response to scaling one row of E and its input bias; the second is c times the response to scaling one output-projection column. The additional parameters can nevertheless change the Euclidean tangent kernel, optimizer trajectory and implicit regularization. Static versus base therefore tests a redundant parameterization effect.

Static K2 gives a precisely matched static-versus-adaptive contrast for K2. It is not the identical static parameterization of exp, so a claim specifically attributing an exp gain to clock input dependence would need that scope qualification rather than silently treating all static optimizers as equivalent.

## 7. A bounded functional-direction lemma

There is a concrete distinction between an adaptive eta direction and redundant static scaling. Consider one scalar SiLU neuron at unit input/output scale and zero input bias. Its ordinary parameter tangent functions lie in

```text
span{1, phi(z), phi'(z), z*phi'(z)},   phi=SiLU.
```

The adaptive second-solve direction is, up to constant beta,

```text
q(z)=s(phi(z))*phi(z).
```

This q does not belong to the displayed span globally. To prove it, suppose

```text
q=A+B*phi+C*phi'+D*z*phi'.
```

As z tends to negative infinity,

```text
phi=z*exp(z)+O(|z|exp(2z))
phi'=(z+1)*exp(z)+O(|z|exp(2z))
q=.6*z*exp(z)+O(z^2 exp(2z)).
```

First A=0. Divide by exp(z) and compare the resulting polynomial coefficients: D=0, C=0, B=.6. But as z tends to positive infinity, `q/phi=s(phi)` tends to 1, contradicting B=.6.

Thus the adaptive eigenparameter contributes a genuine new functional tangent direction in this **one-neuron** comparison. The static direction c*phi is redundant. This lemma gives a more specific mechanism than merely calling the activation stable. It applies equally to adaptive K1, K2 and exp, so it supplies no higher-order preference. It does not prove independence from the full wide network's other neurons, an approximation-rate improvement, successful optimization, or an FM-specific benefit.

## 8. The generic K1 control and existing activation work

For the fixed clock,

```text
F_K1(z)=(1+.2lambda)*z+.8lambda*SiLU(z).
```

It is exactly an affine mixture of the identity and SiLU, a generic learned activation. It is closely related to ACON-C, whose official implementation uses `d*z*sigmoid(b_ACON*d*z)+p2*z`. For nonzero lambda, choose `d=.8lambda`, `p2=1+.2lambda`, and `b_ACON=1/(.8lambda)` to recover this K1 map. At lambda=0 the output has a continuous identity limit, but these ACON coordinates are singular. Therefore the function relation does not imply identical finite ACON initialization or optimization. The official AconC parameter b_ACON is unconstrained. [Official implementation](https://github.com/nmaac/acon/blob/main/acon.py), [CVPR 2021 paper record](https://openaccess.thecvf.com/content/CVPR2021/html/Ma_Activate_or_Not_Learning_Customized_Activation_CVPR_2021_paper.html).

This is why K1 is an essential control rather than a proposed ODE novelty. If only K1 is useful, the evidence supports adaptive activations, not the benefit of higher-order diagonal solves. The publisher PDF was unavailable during this check; the comparison above was verified against the authors' code, not claimed as a full-paper review.

## 9. Costs and the minimal interpretable comparison

The learned-slope proposal adds four r-vectors, lambda1/lambda2 and a1/a2, beyond the two ordinary projections. The fixed-clock refinement adds only two r-vectors eta1/eta2. Static K2 and adaptive K1/K2/exp then have exactly the same additional trainable count, while the plain SiLU control has none of these extra parameters.

With projection biases, the base count is `2dr+r+d`; the fixed-clock flow variants have `2dr+3r+d`. At the proposed d=64,r=128 these are 16576 and 16832 parameters per FFN, respectively, with 16384 dense matrix MACs in both cases. SwiGLU nominal128 has actual inner width85, 16554 parameters and 16320 dense MACs. These are close budgets, not identical counts. Pointwise sigmoid, tanh, polynomial and exponential operations differ, so actual timing remains necessary. No numerical matrix exponential or SVD is involved.

The proposed seven arms answer distinct questions:

| Control | What it distinguishes |
|---|---|
| Plain SiLU FFN at the same r | Whether the added flow activation does more than the expanded basis alone |
| Static diagonal K2 | Redundant parameterization with exactly the base function class |
| Adaptive K1 | Generic input-dependent identity/SiLU activation with matched eigenparameter count |
| Adaptive K2 | Whether the quadratic generator term contributes beyond K1 |
| Adaptive exp | Whether the exact frozen diagonal flow contributes beyond K1/K2 |
| Both established SwiGLU budgets | Whether any candidate meets the practical quality/cost reference |

All three adaptive arms should copy the same E/R and biases, have the same fixed clock and eigenparameterization, and be checked for exact initial function and Jacobian equality. Their initial full tangent identity is an additional check, not a substitute for training outcomes. The ordinary backbone and FiLM must remain paired across all arms. If claiming progress specifically over the old dense ODEFFN, use its unchanged paired reference or clearly identify earlier numbers as historical context; the new-family controls alone cannot isolate which old architectural restriction mattered.

This review does not set an experiment grid, choose a clock or beta from validation data, or authorize these runs. It verifies the mathematical properties of the refinement proposed for consideration. A later protocol should retain the existing requirement to beat generic controls, declare stopping criteria, and avoid rescuing a failed result with another order, width or gate-scale sweep.

## 10. What remains unsupported

The expanded basis is plausible for allocating a fixed budget toward useful nonlinear features. The existing data do not prove that dense hidden P, lack of monotonic scalar warps or a specific activation shape caused the earlier failures. The conditional-covariance formula for the FM target does not imply that its hidden diagonal generators should have these eigenvalues or that these activation functions approximate the target better.

Likewise, lifting h into a larger space can be represented by a nilpotent shear generator: on `(h,z)`, the linear flow with generator `[[0,0],[E,0]]` maps `(h,0)` to `(h,Eh)` after unit time. A second shear can produce `h+Rz`. These are exact algebraic constructions available for ordinary projection networks as well. Middle SiLU and discarding auxiliary coordinates prevent using the construction to claim that the complete FFN is an invertible flow. The lift representation adds no FM-specific evidence by itself.

The defensible claim before any experiment is narrower: **this family converts the professor's two-solve idea into a bounded, analyzable activation perturbation of an ordinary expanded FFN, with controls that separate static reparameterization, generic adaptive activation and higher-order terms.** It is a coherent alternative structural factorization. Whether it is useful remains open.
