# Exact oracle transport for the Product16 law

## Purpose

Product16 has a stronger analytic reference than a generic mixture: in the
known rotated basis its target coordinates and Gaussian source coordinates
are independent. The native FM oracle velocity separates coordinatewise.
Derive its exact flow and variational transition before interpreting a failed
linear error approximation. This concerns the target oracle, not a new FFN.

## Assumptions and notation

The target latent coordinate is an equal mixture of Normal(-m,sigma²) and
Normal(m,sigma²), m>0 and sigma>0. Source coordinates are independent Normal(0,1).
The native interpolation is tX+(1-t)epsilon. The fixed orthogonal rotation R
maps latent row vectors to world coordinates by u R^T. All formulas below
are scalar per coordinate; different coordinates may have different m,sigma.

Write c_t=(1-t)²+t²sigma², b_t=sqrt(c_t), and let Phi,phi be the standard
normal CDF and density. The latent marginal CDF and density are

```text
F_t(u) = [Phi((u-t*m)/b_t) + Phi((u+t*m)/b_t)] / 2
p_t(u) = [phi((u-t*m)/b_t) + phi((u+t*m)/b_t)] / (2*b_t).
```

## Derivation

1. Independence and isotropic Gaussian corruption make the latent oracle
   velocity diagonal: each coordinate depends only on that coordinate and t.
2. Its scalar continuity equation is partial_t p + partial_u(p v*)=0.
   Integrating from minus infinity, with vanishing Gaussian-tail flux, gives
   partial_t F_t(u)=-p_t(u)v*(u,t).
3. Hence d F_t(u_t)/dt=0 along the native oracle flow. Since p_t>0,

```text
u_t = inverse(F_t)(Phi(u_0)).
```

   This is the native oracle trajectory, not merely a different map with the
   same endpoint law. Conversely implicit differentiation gives u_t'=v*;
   uniqueness holds because the native field is globally Lipschitz on the
   finite time interval with strictly positive component variances.
4. Between times a and b, differentiate
   F_b(u_b)=F_a(u_a) with respect to u_a:

```text
d u_b / d u_a = p_a(u_a) / p_b(u_b).
```

   In world coordinates the transition matrix is
   R diag(p_a(u_a)/p_b(u_b)) R^T, evaluated along one oracle path. This is the
   actual variational transition, not the Jacobian of an internal P.
5. Therefore the first-order endpoint error in latent coordinates is

```text
z_i(1) = integral_0^1 [p_t(u_i(t)) / p_1(u_i(1))] * e_i(u(t),t) dt.
```

The model error e may couple coordinates. The oracle transition remains
diagonal; this formula does not assume the learned model separates.

## Consequence and falsification

An error near a low-density oracle endpoint can be amplified by the density
ratio. This is a possible explanation for sensitivity concentration, not a
claim already established for the learned models. Finite perturbations can
leave this linear regime, particularly when they change a mode assignment.
Compare analytic oracle paths and transition derivatives with the saved
native arrays before using the ratio to interpret a calibration failure.

The earlier Product16 CDF endpoint control was described as an endpoint-law
map rather than the native oracle trajectory. For this factorized Gaussian-
source law, the derivation above shows their exact mathematical equivalence.
The older numerical artifacts and scores are retained; a checked addendum
will distinguish exact equivalence from finite-solver discrepancies.

No corresponding coordinatewise formula is assumed for Ring8, whose marginal
coordinates are dependent. No generation or architectural advantage follows
from having a closed oracle reference.

## A distribution-preserving control on the actual non-Gaussian law

Let Psi_t(z) be the coordinatewise quantile map above, and let Q_t=exp(t Omega)
for a fixed skew-symmetric Omega. The paths y_t=Psi_t(Q_t z_0), with Gaussian
z_0, have exactly the same Product16 marginal p_t at every time because Q_t
preserves the standard Gaussian law. Inverse-map Gaussian coordinates are
z_i=Phi^{-1}(F_t(y_i)), and g_i=phi(z_i)/p_t(y_i). Differentiation gives

```text
v_new(y,t) = v*(y,t) + e(y,t)
e_i(y,t) = g_i * (Omega z)_i.
```

Since d g_i/d y_i = -z_i-g_i partial_i log(p_t), the weighted divergence is
div(e)+e dot grad(log p_t)=-z^T Omega z=0. Thus this is an actual-law version
of the Gaussian rotational null: nonzero velocity error can preserve every
target marginal, not just the terminal law.

For v*+lambda e, the exact paths are Psi_t(exp(lambda*t*Omega)z_0), and the
endpoint tangent at lambda0 is diag(g_1)Omega z_0. This supplies a closed-form
calibration of the same first-order endpoint approximation on Product16.
Failure of that approximation in this control cannot mean generation failure.
Conversely it does not prove the learned fields have this special structure.

Before observing the control, fix all256existing selection sources, the four
registered lambda values and Omega=.2(C-C^T), where Ccyclically shifts all16
coordinates by one. Its norm is at most.4; no axis or rate is selected from
learned-model outcomes. Check skew/orthogonality, the weighted-divergence
identity and exact flow derivatives. Report paired displacement and linear
residual only. Population marginals are correct by the derivation, not by a
small-source SW2estimate. This is a mathematical control, not a candidate FFN.
