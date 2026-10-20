# Mathematical notes for the original ODELayer

These are standard identities derived from the implemented layer, not claims of
new theory. Column-vector notation is used; PyTorch Linear applies the equivalent
transposed operation to batches stored as rows.

## 1. What is being approximated?

Write `F_K(h) = sum_{k=0}^K s(h)^k P^k h / k!`.
The code computes `s(h)` once before the recurrence. Therefore it is a Taylor
polynomial for the **constant-P linear system with its duration held fixed**,
not K steps of a nonlinear system whose gate is recomputed at every inner step.
For each fixed h and s, `F_K(h) -> exp(sP)h` for every finite P.

With `r = s ||P||2 >= 0`, submultiplicativity gives

`||exp(sP) - T_K(sP)||2 <= sum_{k=K+1}^infty r^k/k!`
`                         <= exp(r) r^(K+1)/(K+1)!`.

The last inequality follows from `(K+1+j)! >= (K+1)! j!`.
The bound concerns an operator approximation, not learned-model quality.
The complete input-gated mapping need not satisfy a semigroup property:
composing it generally recomputes a different duration at the new input.

## 2. The full derivative

Holding s fixed gives the direct derivative `T_K(sP)` with respect to h.
Differentiating with respect to s gives

`d/ds [T_K(sP)h] = sum_{k=1}^K s^(k-1) P^k h/(k-1)!`
`                 = P T_(K-1)(sP)h`.

The chain rule therefore yields

`J_F = T_K(sP) + [P T_(K-1)(sP)h] [grad_h s]^T`.

The second term has rank at most one, but its norm is the product of two vector
norms and need not be small. In this gate,

`||grad s|| <= (0.8 / (4*10)) ||grad g|| = 0.02 ||grad g||`.

The sigmoid derivative is bounded; the learned gate network's derivative is not
bounded by this coefficient alone. For an external-time gate, the correction
vanishes when differentiating with respect to h at fixed t, but retraining changes
P and s as well. Hence the observed time-gate comparison is not a pure removal
of one term from an otherwise fixed model.

If `B_K(r)=sum_{k=0}^K r^k/k!`, then

`||J_F|| <= B_K(r) + ||P|| B_(K-1)(r) ||h|| ||grad s||`.

This explains both why controlling P can help limit part of the sensitivity and
why it cannot establish a complete block/global-network bound by itself.

## 3. Why eigenvalues do not settle finite-time gain

For the linear system, differentiating `||y||^2` gives

`d/du ||y||^2 = y^T(P+P^T)y <= 2 mu_2(P) ||y||^2`,

where `mu_2(P) = lambda_max((P+P^T)/2)`. Consequently
`||exp(sP)||2 <= exp(s mu_2(P))` for nonnegative s.
Negative-real-part eigenvalues alone do not force this finite-time gain below 1
for a non-normal P. A truncated polynomial also has its own amplification behavior;
an exact-exponential bound is not automatically a bound on that polynomial.

The constructed family uses `P=-I/4 + a N`, with upper-diagonal entries of N equal
to `cos(theta)` and `sin(theta)`, and all other entries zero. Since `N^3=0`,

`exp(sP) = exp(-s/4) [I + saN + (sa)^2 N^2/2]`.

All eigenvalues remain -1/4. Choosing a by bisection to match `||P||2=1` retains
variable finite-time gain (1.225180 to 1.397337 at s=1). This construction controls
the eigenvalues and maximum singular value, not the entire singular spectrum or
all measures of non-normality. It is a counterexample to sufficiency of those
summaries, not an FM training result.

## 4. Why exact action need not help a trained network

For squared flow-matching regression, let `Y=x1-x0` and
`v*(x,t)=E[Y | x_t=x,t]`. The conditional-expectation decomposition is

`E||v(x_t,t)-Y||^2 = E||v(x_t,t)-v*(x_t,t)||^2 + E||Y-v*(x_t,t)||^2`.

There is no term rewarding a hidden layer for approximating an exponential.
Changing `T_K` to `exp` changes the parameterized velocity function, even when
all weights are frozen. Surrounding layers may have adapted to the original
polynomial. The intervention demonstrates sensitivity to that change, but does
not compare the optima of models separately trained with polynomial and exponential
layers. It also does not identify which layer or gate causes the adaptation.

Even a smaller aggregate oracle regression error does not order endpoint SW2:
error location along time and state, and propagation through the learned flow,
can matter. The Ring8 K4 example documents such a mismatch without proving its
specific cause. Any future explanation should test these mechanisms separately.
