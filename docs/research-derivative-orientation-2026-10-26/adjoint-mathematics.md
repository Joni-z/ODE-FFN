# What the useful transpose computation contains in the professor's layer

This derivation follows the completed orientation comparison. It is not a pre-outcome hypothesis or a newly trained architecture. The directional setting remains stopped.

For the unchanged scalar K2 layer, a=a(h) is computed once from the input and

```text
L(h) = h + a P h + a²/2 P² h
u(h) = P h + a P² h
J_L(h) = I + a P + a²/2 P² + u(h) grad(a)^T.
```

For an input direction r, its ordinary forward sensitivity is

```text
J_L r = r + a P r + a²/2 P² r + u(h) * dot(grad(a),r).
```

For an output sensitivity q, the adjoint sensitivity is

```text
J_L^T q = q + a P^T q + a²/2 (P^T)² q
          + grad(a) * dot(u(h),q).
```

The same learned P supplies a transposed polynomial, plus a rank-one clock-derivative term. Dropping the latter is not an exact adjoint. Both formulas differentiate the once-computed input clock; they do not silently recompute time along a numerical solve.

For clock preactivation r=W1*h+b1, logit g=w2*SiLU(r)+b2, probability p=sigmoid(g/tau),

```text
grad(a) = scale/tau * p*(1-p) * W1^T[SiLU'(r) .* w2]
SiLU'(r) = sigmoid(r) * [1+r*(1-sigmoid(r))].
```

These closed forms contain no input-autograd call and add no learned parameter. They are currently verified identities, not an implemented full-model inference-speed result.

The whole FFN applies L1, SiLU and L2. Its adjoint applies J_L2^T first, then the SiLU derivative, then J_L1^T. The full network additionally reverses the order of residual-block sensitivities, LayerNorm, time FiLM, input and output projections. Consequently, replacing every P by P^T, or symmetrizing individual P matrices, does not reproduce the successful complete readout.

For complete G, the gradient velocity is .5*(G+J_G^T x). It starts an output sensitivity proportional to x and propagates it back through the complete feature computation. The radial JVP instead propagates an input direction forward. The measured A/B disparity is evidence that this distinction matters to the tested original ODE parameterization. It does not prove conservativity alone caused the gain, identify a particular gate term as necessary, or establish that a local symmetric field would recover it.

A natural implementation question is whether an explicit primal/adjoint sweep can evaluate the existing good function more efficiently than nested input-autograd. Such a replacement must preserve complete velocities, input derivatives and parameter gradients, and must include the same generic controls. It would be a function-preserving implementation study, not another potential variant or a reason to ignore the remaining Gaussian/strong-baseline failures. No new training is selected by this note.
