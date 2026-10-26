# The two zero-order terms have different differential roles

Let z=gamma1*h+D1(h)P1h and u=phi(z), with phi=SiLU. The output is
B(h)=gamma2*u+D2(u)P2u. Define each complete solve derivative

```text
A_i(w) = gamma_i I + D_i(w) P_i + diag(P_i w) J_si(w).
J_B(h) = A_2(u) diag(phi'(z)) A_1(h).
```

This includes controller derivatives. Replacing either A_i by gamma_i I+DP
would omit the input-dependent term. Unit tests compare the full product
with autograd for all four patterns and both gate families.

## Why all initial hidden functions can agree

Zero each controller's last affine. D_i=cI and J_si=0 initially. For any
chosen gamma_i, set P_i^(gamma)=P_i^legacy+(1-gamma_i)I/c. Then

```text
gamma_i I + c P_i^(gamma) = I + c P_i^legacy.
```

Every initial solve, intermediate activation, block derivative and complete
backbone function is identical in exact arithmetic. Vector/scalar controllers
also agree at this point despite different hidden sizes, because all their
logits are zero and their non-controller draws are paired. Actual float32
reconstruction errors are checked and recorded; ordinary formal training
still uses float32.

## Which new response each removal creates

At a common initial function the derivative of the output with respect to
the duration vectors is

```text
response_s1 = A_2(u) diag(phi'(z)) diag(P1 h)
response_s2 = diag(P2 u).
```

Removing the first identity changes P1 to P1+I/c and adds
A_2 diag(phi') diag(h/c) to the first response. Removing the second adds
diag(u/c) directly to the second response. Scalar controls restrict each
duration perturbation to the all-ones direction; the same identities apply
after that contraction. Subsequent trained states need not stay paired.

There is no general ordering of these responses or of their tangent kernels:
signed cross terms can enlarge or reduce a kernel, as the preceding
zero-order counterexample demonstrates. Therefore mixed cases must be
measured rather than extrapolated from the both-increment result.

## Zero-P boundary

At P1=P2=0 and generic nonzero h:

| Pattern | Block value | First P variation | Second P variation |
|---|---|---|---|
| MM | phi(h) | active | active |
| MI | 0 | zero | active |
| IM | 0 | active | zero |
| II | 0 | zero | zero |

For MI the nonzero feature phi(h) reaches P2; for IM, phi'(0)=1/2 lets P1's
variation pass through the second identity. II has neither path. Tests check
this table on the actual two-solve implementation. The table concerns a
block differential, not a gradient through the initially zero velocity head.
The paired initial matrices are generally nonzero, so no convergence claim
follows from this boundary alone.

The factorial isolates location and gate granularity under common initial
functions. A useful architecture still requires a practical gain over
SwiGLU, nonlinear transfer, derivatives, generation and fresh-holdout evidence.
