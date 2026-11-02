# What the two interventions test

## Frozen generator versus a state-dependent ODE

Write the layer as `F(h) = h + A(h)h + A(h)^2 h / 2`.
The original scalar gate uses `A(h)=s(h)P`; the proposed alternatives use
`A(h)=PD(h)` or `D(h)P`. For each input, the gate is held fixed across the
two polynomial terms. Backpropagation still differentiates that gate.

If `u` is an input perturbation and `dA` the directional derivative of A along u,
the exact derivative of the implemented map is

```
dF[u] = u + A u + dA h
        + (A^2 u + dA A h + A dA h) / 2.
```

Thus small singular values of P alone do not establish a small full Jacobian:
the controller derivative and the current hidden state also enter. This is why
the protocol records fixed-probe full Jacobians alongside P and gate statistics.
The float64 dense-polynomial and directional finite-difference tests include
all these derivative paths.

For a genuine continuously recomputed field `f(h)=A(h)h`, a second-order ODE
step would instead contain `Df(h)[f(h)]/2`, including `DA(h)[A(h)h] h/2`.
That additional term is absent from the professor's frozen-generator map.
The present study tests the map actually implemented; it does not silently
replace it with a different ODE integrator.

## What changing PD to DP can and cannot change

At the same h and P, all diagonal entries of D lie in (0.2,1). Therefore D is
invertible and `PD = D^(-1) (DP) D`. The frozen generators have identical
eigenvalues, as do their polynomial maps. An advantage of one placement cannot
be explained by different eigenvalues **at the same P and D**. Their singular
values, input action and state-dependent Jacobians may differ; the trained
parameters may also diverge. Eigenvalue equality is not functional equality.

The norm of the polynomial obeys `||T2(A)|| <= 1 + ||A|| + ||A||^2/2`.
Both placements obey `||A|| <= ||P|| max_i |d_i|`. These are upper bounds,
not evidence that the trained operators attain them. A finite-dimensional
matrix exponential converges for every finite P; the concern is amplification,
truncation and optimization, not a finite spectral-radius convergence threshold.

## What Muon changes

The official algorithm approximately orthogonalizes a momentum-derived update
using five BF16 Newton–Schulz steps. It does not project P onto the orthogonal
group, and the approximation does not make every update singular value exactly
one. Adding successive such updates to P provides no general bound on P's norm,
condition number or full layer Jacobian.

The causal comparison is consequently empirical: with the model and auxiliary
optimizer fixed, does this update geometry improve optimization and the learned
field? Record pre-step clipped gradient, post-step momentum, actual Delta P,
and P separately. Compare the same intervention on SwiGLU. If both families
benefit, report an optimizer effect; if only P's spectrum looks flatter while
field/sample errors do not improve, there is no demonstrated model benefit.
