# Post-result algebra: C and B change optimization coordinates

This explanatory derivation was added after the fixed screen. It changes no model, optimizer, threshold, outcome or continuation decision. It is not presented as a preregistered causal explanation.

The two proposed implementations have the same real-arithmetic class under B=P²+C. Let L(P,B) denote the complete differentiable loss, keeping all other parameter coordinates unchanged, and let Gp/Gb be its gradients in independent P/B coordinates. For T(H)=PH+HP, its Frobenius adjoint is `T*(G)=P^T*G+G*P^T`. Chain rule gives

```text
grad_P in C coordinates = Gp + T*(Gb)
grad_C = Gb.
```

Under ordinary Euclidean gradient flow in P/C, the induced evolution of the independent P/B matrices is

```text
Pdot = -Gp - T*(Gb)
Bdot = -T(Gp) - (I+T T*)(Gb).
```

The joint gradient is multiplied by a positive definite coordinate metric

```text
M = [ I       T*       ]
    [ T       I+T T*   ]
  = [ I  0 ] [ I  T* ]
    [ T  I ] [ 0   I ].
```

The triangular factors are invertible, so M is positive definite and has determinant one. It can still be poorly conditioned. The instantaneous loss derivative contributed by these coordinates is

```text
Ldot = -||Gp+T*(Gb)||_F² - ||Gb||_F².
```

Direct independent P/B gradient flow instead contributes `-||Gp||²-||Gb||²`. The tied-coordinate implementation therefore changes coupled update directions despite representing the same functions. It need not decrease the loss faster for every gradient: the cross term can have either sign.

The actual experiment uses AdamW with clipping and stochastic minibatches. Its evolution is not this Euclidean gradient flow; the small float32 initial-value differences also remain recorded. Thus the better finite-budget C result cannot be attributed uniquely to M from the present evidence. The theorem explains why the direct parameterization is a necessary control and why a difference is not an expressivity or solver-order theorem. It does not select an optimizer or initialization rescue.

[verify_coordinate_metric.py](verify_coordinate_metric.py) independently checks the chain rule, pushforward Jacobian/metric, determinant and loss-rate identities on a small float64 fixture. No fitted checkpoint is used.
