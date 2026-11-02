# Two ODE interpretations of an input-dependent diagonal

This is a mathematical distinction, not a claim that the inherited implementation
is wrong. The existing layer legitimately freezes its input-dependent operator
at the entrance. It is a different construction from evolving that operator with
the internal state. This note alone does not select another training family.
The general distinction was already derived in the earlier
[state-clock review](../../research-architecture-search/state-clock-experiment-review.md).
This note specializes it to current PD/raw input features; it is not a newly
discovered motivation to rerun the previously stopped midpoint setting.

## Entrance-frozen linear solve

At input h, form A_h=Pdiag(d(h)) and hold it fixed during the internal solve:

```
dy/dtau=A_h*y,   y(0)=h.
y(tau)=exp(tau*A_h)h.
Q2=h+tau*A_h*h+(tau^2/2)*A_h^2*h.
```

The current PD recurrence is exactly this polynomial at tau=1. Derivatives of
d(h) are included by ordinary backpropagation when the whole map is differentiated;
they are not omitted from training. They do not appear in the internal frozen
matrix power because A_h is constant along that auxiliary solve.

## State-dependent nonlinear flow

Alternatively define f(y)=P[d(y)*y] and solve dy/dtau=f(y). Its second-order
time expansion at the same entrance is

```
y(tau)=h+tau*f(h)+(tau^2/2)*J_f(h)f(h)+O(tau^3),
J_f(h)z=P[d(h)*z]+P[h*(J_d(h)z)].
```

The second contribution is state feedback. The difference from the frozen
second-order expression is

```
(tau^2/2)*P[h*(J_d(h)f(h))].
```

For d(h)=h, f=P(h*h) and J_d=I. Both pieces of J_f f are equal:

```
Frozen K2:    h+tau*P(h*h)+(tau^2/2)*P[h*P(h*h)].
Nonlinear K2: h+tau*P(h*h)+ tau^2   *P[h*P(h*h)].
```

Thus even this elementary diagonal admits two different ODE explanations. An
ODE label alone does not specify which function class or derivative is intended.
For d=h+1 the additional feedback term is still P[h*f]/2 at tau=1. For constant
d it vanishes. For coordinatewise bounded d it uses the known sigmoid derivative.
For an MLP controller J_d is generally dense and requires its Jacobian-vector
product, not merely a diagonal derivative.

The scalar original has the same distinction if one reinterprets the duration
as a state-dependent rate: f(h)=s(h)Ph,
J_f f=s(h)^2 P^2 h+s(h)(grad(s)^T Ph)Ph. Its original interpretation instead
chooses an input-dependent duration for a linear solve, which is valid in itself.

## What would justify an experiment

First verify consistency on a known flow, then ask whether the resulting
representation helps the learned field under matched ordinary controls. A
small-tau consistency check is not a quality result. At unit internal time and
large learned P, a Taylor remainder need not be small, and a more accurate
nonlinear-flow discretization need not be a better FFN. Cycle3's K1 result is
already a reason to be skeptical of increasing this tied cubic contribution.

The diagnostic scalar equation y'=p*y^2 has exact solution h/(1-p*tau*h) before
its pole. Entrance-frozen exp(tau*p*h)*h approximates a different equation.
The associated numerical witness checks these orders without changing any model,
checkpoint, training budget or primary metric. This branch remains a conceptual
alternative until complete D-design evidence motivates a finite comparison.
