# What swapping DP and PD tests

Use column-vector notation. Each duration is positive, and all durations are
computed once from the solve input h. This is the original frozen-controller
interpretation, rather than an exact solution of an arbitrary nonlinear ODE.
K1 is the Euler map for either chosen internal drift. FM path time t and this
hidden-layer duration are different quantities.

## Scalar extension leaves an ordering choice

For the professor's scalar duration, s(h)Ph=P(s(h)h) as functions of h, even
when the controller depends on h. Their complete derivatives also agree.
Changing scalar rates to D(h)=diag(s(h)) introduces two possible generators:
A_after=DP and A_before=PD. The current vector parent chose the first.
Changing the order tests that choice; it adds no matrices or rate channels.

The difference of the two K1 maps is [D,P]h. This vanishes for scalar D or
when D and P commute, so a placement gain cannot exist in that boundary.
For strictly positive D,

```text
A_before = D^(-1) A_after D
T_K(A_before) = D^(-1) T_K(A_after) D
```

for every finite Taylor polynomial T_K. Frozen eigenvalues, traces and
determinants match at the same D/P. Their actions on h generally differ.
For any subordinate norm the corresponding polynomial norms are within
the factor cond(D), at most5 for the original duration range. This is a
comparison of frozen maps at matched weights and anchors, not a stability
guarantee for their learned full state-dependent maps. No K2 training is
introduced by stating the general polynomial identity.

## The complete derivative contains the distinction

Differentiating the state-dependent K1 map gives

```text
J_after  = I + D P + diag(P h) J_s
J_before = I + P D + P diag(h) J_s.
```

The last term cannot be reconstructed from eigenvalues of P or DP alone.
With q the loss derivative at a solve output, the local gradients are

```text
grad_P_after  = (D q) h^T
grad_P_before = q (D h)^T
grad_s_after  = diag(P h) q
grad_s_before = diag(h) P^T q.
```

These formulas include all downstream dependence through q. They explain
how gate placement changes both feature weighting and the error delivered
to the controller. They do not prove that one orientation converges faster.
Ordinary Adam, clipping, changing features and two composed solves remain
part of the actual training dynamics.

## An exact two-dimensional case inside the controller class

Take P=[[0,1],[1,0]] and s_j(h)=.2+.8 sigmoid(h_j). This controller is
represented exactly by the existing two-layer SiLU network: use first
projection [I;-I], last projection10[I,-I], zero biases and tau10, since
SiLU(z)-SiLU(-z)=z.

At h=(2,-1), let a=s_1, b=s_2 and a',b' be their scalar derivatives. Then

```text
J_after  = [[1-a', a], [b, 1+2b']]
J_before = [[1, b-b'], [a+2a', 1]].
```

Both frozen generators have eigenvalues +/-sqrt(a*b), but the complete
Jacobians above have different spectra. The exact arithmetic identities
and numerical values are independently checked in rate-order-case.json.
This case demonstrates why frozen spectral diagnostics alone cannot
choose between the two operators. It is not a performance argument and
uses no trained checkpoint or evaluation outcome.

## What an empirical gain could establish

A matched-budget gain would support a useful ordering of feature selection
and learned mixing in this backbone. It would not by itself demonstrate an
FM-specific advantage, global flow stability, a new general function class,
or a novel gating primitive. The registered nonlinear, generation, derivative
and cost controls still decide whether to continue. An unsuccessful screen
will retain the derivative insight without accepting the architecture.
