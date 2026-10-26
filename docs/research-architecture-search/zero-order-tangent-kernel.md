# The paired control changes gradient directions, not just their scale

This is an additional derivation during the registered Stage13 run; it adds
no training change or retrospective screen criterion.

At D=cI and Q=P+I/c, both solve functions and their P responses agree.
The last-controller response changes from J to J+H, where H is the response
of the identity component through the duration. All other parameter responses
of a single solve agree while the last controller weight is zero. Thus

```text
K_increment - K_map = J H^T + H J^T + H H^T.
```

The cross terms need not be positive. This differs from a uniform coordinate
scale, whose kernel contribution is multiplied by a positive scalar. A larger
P norm after the paired shift does not imply a uniformly larger tangent kernel.

An exact single-solve counterexample uses a scalar duration and P=alpha*I/c.
At a common nonzero h, the map's duration response is alpha*h/c, while the
increment's response is (alpha+1)*h/c. All last-controller parameters share
the same feature factor, so their kernel difference has the sign of

```text
(alpha+1)^2 - alpha^2 = 2*alpha + 1.
```

It is positive for alpha=0 and negative for alpha=-3/4. In either example
the full solve functions initially match and the P-response contribution
is identical. Full autograd over every actual layer parameter verifies both
signs in test_zero_order_kernel.py. No synthetic training outcome enters
this construction.

The argument does not predict which directions matter to FM risk, or how
Adam/clipping and moving features respond. At the actual zero output head,
backbone tangent responses first vanish in both architectures. Any useful
fit must still be established by the registered practical controls.

## What a P penalty means under the static equivalence

For fixed c, the complete solve matrix is A=I+cP in map form and A=cQ in
increment form. Penalizing ||P||² centers A at I; penalizing ||Q||² centers
A at zero. Likewise, identical raw spectral bounds describe different sets
of complete maps. A paired shift can therefore leave the entire initial
hidden function unchanged while moving the raw P spectrum substantially.
Regularization of raw P is not representation invariant under this choice.

For learned state-dependent durations the complete Jacobian also contains
controller derivatives. Neither a raw P penalty nor a bound on gamma*I+DP
alone is a complete-Jacobian or data-transport stability certificate. No
regularizer is added to the current experiment by this observation.
