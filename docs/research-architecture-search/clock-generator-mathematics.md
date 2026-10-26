# What should the input-dependent controller control?

The original K1 solve and the proposed alternative are

```text
parent:     F_mul(h) = h + D(h) P h
candidate:  F_add(h) = h + c P h + (D(h)-cI) h
D(h) = diag(s(h)); c = shift+scale/2 = .6.
```

For a scalar gate, D=sI. Both are Euler polynomials of frozen, conditional
linear generators: DP versus cP+diag(s-c). The controller is evaluated at the
solve input and held fixed inside the polynomial. The second formula changes
the generator; it is not a more accurate numerical solve of the first one.
The actual experiment remains K1. In particular, neither hidden solve time
nor c is the outer flow-matching time t.

## The restriction to test

Recenter the original layer without making an approximation:

```text
F_mul(h) = (I+cP)h + (D(h)-cI) P h.
```

P defines both the static linear map and the frame receiving the conditional
correction. The alternative keeps the same static map but applies that
correction to h. A scalar positive clock on a fixed one-dimensional linear
generator changes speed while preserving the sign of the relative increment.
This is a local restriction worth testing because a simple FM field changes
that sign across both state and time. Dense hidden P, learned embeddings and
multiple solves can circumvent the one-dimensional restriction. It cannot be
used as a lower bound on the trained VelocityNet.

## An exact FM example and a restricted error bound

Take the one-dimensional latent component of the existing product16 target:
Y is an equal mixture of N(-mu,var) and N(mu,var), Z is standard Gaussian,
X=(1-t)Z+tY. Set a=1-t and r=a^2+t^2 var. Conditioning the mixture gives

```text
v(x,t) = alpha*x + beta*tanh(kappa*x)
alpha = (t*var-a)/r; beta = a*mu/r; kappa = t*mu/r
q(x,t) = v(x,t)/x; q(0,t)=alpha+beta*kappa.
```

Consider the local Euler target Phi=x+delta*v, delta=.001. This target is
strictly increasing: Phi' >= 1+delta*alpha > .9975 on the declared panels.
It is a local flow approximation, not the exact finite-time FM flow map.

For every positive controller, either x+p*s(x,t)*x or exp(p*s(x,t))*x has a
relative increment with fixed sign determined by p. Its squared error must
therefore include the target increment energy in the opposite sign region.
Writing E+ and E- for E[(delta*v)^2 1_{x*v>0}] and its negative counterpart,
the risk is at least min(E+,E-). With p fixed across times, sum E+ and E-
over times before taking the minimum. This also relaxes the finite MLP
controller and its actual range, so it remains a valid bound for that class.

The calculation covers all 16 declared latent coordinates and five existing
diagnostic times, without choosing a favorable panel. The exact inequality's
integrals are evaluated numerically using sign-separated Gauss-Legendre
160/320 rules and a mode+12-sigma truncation. Their maximum absolute
disagreement is 5.00e-20 in increment-energy units. This is numerical evidence
for the integral values, not an interval-arithmetic proof of them.

Allowing a different p in each panel gives a relative floor of .005112.
Keeping one p per coordinate across the five times raises it to .086608
(velocity-unit MSE .144361; target energy 1.666836). The reference panel is
preselected as the coordinate with mu closest to one, at t=.5. Its relative
floor is .262012. These numbers apply only to the restricted scalar map
families. They are not predicted errors of the full network.

In contrast, the alternative admits a constructive witness. Over a coordinate's
declared times let q_min,q_max bound q, m=delta*(q_min+q_max)/2, p=m/c and
s=c+delta*q-m. Then the same fixed p and an admissible s in (.2,1) give
F_add=Phi exactly. For the exact conditional linear flow, apply the same
construction to log(1+delta*q). The numerical witness error is at most
1.78e-15. These analytic controllers are capacity witnesses, not trained MLPs.
The formula agrees with the independently implemented mixture oracle to
1.03e-14 on all declared panels.

## A witness inside the actual finite controller

There is also a direction-changing example using only two SiLU hidden units:

```text
r(x) = SiLU(x)+SiLU(-x) = x*tanh(x/2)
g(x) = 2*tau*(1-r(x))/4
P=0; s(x)-c = .4*tanh((1-r(x))/4)
Phi(x) = x*(1+s(x)-c).
```

The actual controller realizes this g with first weights (1,-1), zero first
bias, last weights (-tau/2,-tau/2), and last bias tau/2. Near zero the map
expands relative to x; in the tails it contracts. No single positive-clock
scalar generator realizes this map, irrespective of its controller capacity.

The map is nevertheless strictly increasing globally. Put y=(1-r)/4.
From cosh(x/2)^2 >= x^2/2, x*r' <= r+1, so
(x*r')/4 <= .5-y. Also |y|*sech(y)^2 <= .5, obtained from
cosh(y)^2 >= 2|y|. Therefore

```text
Phi' >= 1+.4*(tanh(y)+y*sech(y)^2-.5*sech(y)^2) >= .2.
```

The analogous exact conditional exponential map is increasing too:
1+x*(s-c)' >= .6, hence its derivative is at least .6*exp(-.4).
Unit tests check the finite controller exactly and check, but do not replace,
the analytic derivative bound on a grid.

The families are not nested. For the old layer choose p=2 and
g=2*tau*x, exactly realizable via SiLU(x)-SiLU(-x)=x. Its relative increment
2s(x) spans 1.6. For the new layer c*p+s(x)-c spans at most .8 for every p,
so it cannot represent that old map. The new design trades a multiplicative
conditional range for an independent signed diagonal correction.

## Initial functions and training responses

Zeroing only the last controller affine initializes D=cI. With identical raw
tensors, both complete networks then have the same initial function, including
a nonzero output head. Counts and dense matrix MACs match exactly. Controllers
are trained normally afterward. Each FFN retains two independent solves and
the intervening SiLU; no additional residual path is added.

For a single solve, Js is the controller input Jacobian and z is its output
adjoint. The complete Jacobians and P gradients are

```text
J_mul = I+DP+diag(Ph)*Js;   grad_P L = (D z) h^T
J_add = I+cP+diag(s-c)+diag(h)*Js; grad_P L = c z h^T.
```

At constant-gate initialization the P responses agree. Controller parameter
responses receive z*(Ph) versus z*h. At P=0 the parent controller response
vanishes, while the candidate's need not; both initial maps can still be the
identity. In the full experiment the zero velocity head masks backbone
gradients initially, and P is not initialized to zero. This identity is a
mechanism boundary, not a claim about the first formal update.

The feature change also changes gradient magnitudes. A training gain alone
does not isolate directional freedom from conditioning or ordinary gating.
There is no uniform tangent-kernel or convergence dominance. A useful primary
result would require a matched gain/direction control and a fresh independent
holdout before accepting this architecture. A failed result stops this
setting; it does not prove that generator conditioning is impossible.

## Reproduction

`scripts/certify_clock_generator.py` and `certify_clock_generator_time.py`
produce the local certificates. Exact scripts, the oracle dependency, manifest
and their hashes are frozen alongside the raw results under
`../ode-workspace-prep/architecture-search/clock-generator*-20261024/`.
`tests/test_additive_generator.py` covers the production algebra, full input
Jacobian, P gradients, finite witness, initial functions/RNG and budget pairing.
