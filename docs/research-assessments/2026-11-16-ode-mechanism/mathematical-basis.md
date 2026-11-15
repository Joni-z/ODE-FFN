# What the present ODE interpretation guarantees

16 November 2026. These are mathematical deductions and small numerical
checks, not new trained-model findings. Hidden time is `s`; generative time
is `t`. They are different variables throughout.

## 1. The exponential coefficients are a testable structural restriction

For a fixed generator A, consider an analytic family R(sA) with R(0)=I
and derivative at zero A. If it satisfies

```text
R((s+u)A) = R(sA) R(uA),
```

differentiation in u at zero gives dR(sA)/ds = R(sA)A. The unique solution
is exp(sA). For scalar analytic r(z)=sum c_j z^j with c_0=c_1=1, this
imposes c_(j+1)=c_j/(j+1), hence c_j=1/j!.

This identifies a specific restriction to release experimentally. A finite
Taylor polynomial satisfies this only approximately; a polynomial with
free coefficients has no such general guarantee. For a submultiplicative
norm, `||exp(A)-T_K(A)|| <= exp(||A||)||A||^(K+1)/(K+1)!` is a valid, often
loose remainder bound. Neither the bound nor the coefficient identity
implies better flow-matching regression or lower distribution error.

## 2. Input-conditioned exponentials are not generally a flow on h alone

Our exact-exponential idealization is

```text
A(h) = P D(h)
F_s(h) = exp(s A(h)) h.
```

Conditional on the *original* h, evolution in y is linear with constant
A(h). But composing two layer calls recomputes A at the intermediate state.
In general `F_u(F_s(h)) != F_(s+u)(h)`, even with exact exponentials.
Keeping A as persistent extra context restores a conditional flow on an
augmented state; our ordinary hidden-only block does not keep that context
across separate calls. This is a property of the chosen parameterization,
not a bug in Taylor evaluation.

Let f(h)=A(h)h. Expanding both sides at small positive s,u gives

```text
F_u(F_s(h)) - F_(s+u)(h)
  = s*u * (J_f(h) - A(h)) f(h) + O(s*u*(s+u))
  = s*u * (D A(h)[f(h)]) h + O(s*u*(s+u)).
```

Here `D A(h)[v]` is the directional derivative of the matrix-valued A.
For componentwise ISRU, `d_i=(1+1.2*h_i^2)^(-1/2)`, q(h)=D(h)h and
q'_i=d_i^3. Therefore the leading defect is

```text
s*u * P (D^3 - D) P q(h).
```

Higher Taylor order cannot remove this term: it comes from changing the
generator on restart, not truncation. If it is small on learned states,
that would be an empirical property of the trained block.

The full Jacobian also contains input conditioning. For a direction v,

```text
J_F(h)v = exp(A)v
        + integral_0^1 exp((1-r)A) (D A(h)[v]) exp(rA)h dr.
```

The spectrum of P, or invertibility of exp(A), does not control this entire
expression. A scalar counterexample uses P=3, h=1, ISRU scale1.2: the
derivative of `h*exp(3*d(h))` is -0.780 there, but positive at zero. The map
is nonmonotone and cannot be an injective scalar autonomous ODE flow map.

## 3. A state-evolving flow is mathematically different

For `dy/ds = f(y) = P q(y)`, q'=D^3 and |q'|<=1 imply global Lipschitz
constant at most ||P||. Thus a unique global solution exists, and the exact
autonomous maps Phi_s satisfy composition. Their expansion is

```text
Phi_s(h) = h + s*Pq(h) + (s^2/2)*P D^3 Pq(h) + O(s^3).
F_s(h)   = h + s*Pq(h) + (s^2/2)*P D   Pq(h) + O(s^3).
```

They share the first-order term but solve different equations beyond it.
Positive D does not imply contraction for arbitrary P. The exact single
flow's invertibility also does not make the complete FFN invertible: its
middle SiLU, projections and other blocks matter.

A four-step explicit midpoint solve on s in[0,1] uses eight P applications:
`mid = y + dt*Pq(y)/2; y = y + dt*Pq(mid)`, dt=1/4.
It gives the same dense P work as frozen K8, with more response evaluations.
Under regularity and adequate resolution it converges with second-order
global error when the mesh is refined. A trained coarse model must be
checked against that limit; numerical integrators can themselves become
part of the learned discrete architecture.

Outer generative refinement holds the learned velocity field fixed. Hidden
refinement instead changes the network's evaluated map at fixed trained
weights: v_theta^(4) need not equal v_theta^(16). Training at four substeps
optimizes the former. Comparing their generated distributions diagnoses
deployment sensitivity to the hidden continuous limit, rather than comparing
two models separately trained at their own integration resolutions.

## 4. The necessary bridge to the actual generative task

A hidden flow is not the output velocity field or its flow map. The useful
hypothesis needs evidence in data space. Ring8/Product16 provide exact
marginal velocity v*(x,t), already implemented in the native data module.

Define e=v_theta-v*, v_lambda=v*+lambda*e, and integrate from the same noise
x0. At lambda=0, differentiating the solution with respect to lambda gives
the exact first-order sensitivity equation

```text
z'(t) = J_v*(x*(t),t) z(t) + e(x*(t),t),   z(0)=0
z(1)  = integral_0^1 Phi(1,t) e(x*(t),t) dt.
```

Phi is the transition operator of the oracle variational equation. For
small lambda, `x_lambda(1)-x*(1) = lambda*z(1)+O(lambda^2)`. At lambda=1
this is a hypothesis about approximation accuracy, not an exact identity.
The full modeled displacement must be compared with the linear prediction.

This separates error magnitude from where, when and in which direction the
error occurs. Equal average field MSE need not imply equal endpoint error:
transport can amplify errors, and transported errors can cancel. Paired
endpoint error is a coupling upper bound for W2, not the optimal W2 itself
and not identical to our sliced-Wasserstein metric. Continue measuring SW2
and mode allocation directly.

The continuous variational operator obeys a bound involving
`mu_2(J)=lambda_max((J+J.T)/2)`. It is not valid to replace J with P, nor to
assume an explicit Euler step obeys the identical exponential bound. For
skew J, ||exp(dt*J)||=1 whereas ||I+dt*J|| can exceed1. We do not introduce
a contraction penalty: multimodal transports can need local expansion.

## 5. What single-generator sharing actually restricts

The released-stage control uses f_j(y)=P_j q(y). Its exact interpretation
is a piecewise nonautonomous ODE, while a single shared P is autonomous.
Both are ODEs. The structural question is the usefulness of time homogeneity
and parameter reuse, not whether one model is continuous and the other is not.

Using [f_i,f_j]=J_f_j f_i-J_f_i f_j gives

```text
[f_i,f_j](y) = P_j diag(q'(y)) P_i q(y)
              - P_i diag(q'(y)) P_j q(y).
```

The bracket is zero for equal P_i and P_j. A consistent midpoint step is
T_j^eps(y)=y+eps*f_j(y)+(eps^2/2)*J_f_j(y)f_j(y)+O(eps^3).
Substitution therefore gives

```text
T_j^eps(T_i^eps(y)) - T_i^eps(T_j^eps(y))
    = eps^2*[f_i,f_j](y) + O(eps^3).
```

This predicts an ordering sensitivity that can be checked at learned
states. It does not predict better samples. Averaging P_j preserves the
first-order sum of stage fields while removing their differences; reversing
the schedule changes its ordering. Their frozen-weight effects show which
computation is active. Only the separate trained controls test its usefulness.

The chosen nested parameterization also changes optimization. If g_j is the
loss gradient with respect to effective P_j, then at any fixed parameter state

```text
dL/dP = sum_j g_j;       dL/dDelta_j = g_j, j=2,3,4.
```

Under ordinary SGD, the effective update of P_1 is -eta*sum_j g_j; that of
P_j for j>1 adds -eta*g_j. Adam and clipping modify these directions/scales
further. Identical initial functions and common gradients do not make the
optimization geometries identical. A worse-fitting released model therefore
cannot by itself establish a helpful autonomous constraint.

The probe independently checks the bracket with autograd, the eps^2 leading
term with decreasing step sizes, and the exactly shared null. It then measures
actual drift differences rather than substituting matrix norms for them.

A stationary P still has a state-dependent Jacobian P diag(q'(y)). Sharing
therefore leaves nonlinear adaptation along the path and the matrix's spatial
anisotropy available to learning. It constrains variation with hidden time,
which differs from constraining singular-value scale or flattening the spectrum.
At a fixed mesh the resulting map is also a shared recurrent FFN; the
experimentally identified ingredient is the stationary nonlinear generator.

## Numerical check and scope

[verify_math.py](verify_math.py) uses NumPy/SciPy to verify the conditional
composition-defect coefficient, true-flow composition, midpoint refinement,
the scalar derivative counterexample and first-order sensitivity convergence.
[math-verification.json](math-verification.json) stores full precision.
These checks confirm the derivation on deterministic examples; they neither
measure the trained networks nor establish a research performance claim.
