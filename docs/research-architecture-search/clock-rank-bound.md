# A measurable conditional rank limit for the complete velocity field

This is a diagnostic bound at fixed trained weights and fixed external time. It is not a lower bound on the full architecture's function class.

Collect every scalar duration in the three-block professor network into c(x) in R^6. Write the complete network as f(x)=G(x,c(x)), treating the six gate outputs as additional independent inputs of G. G includes all projections, the intermediate SiLU, normalization, conditioning and outer residuals. At an anchor x0, hold c at its native forward values and differentiate the resulting function:

```text
J_full = G_x + G_c J_c
J_anchor = G_x
rank(J_full - J_anchor) <= 6.
```

This remains true when later gates depend on earlier layers; c(x) contains their total derivatives. Forward outputs at the anchor are identical. The argument uses only the chain rule and the number of scalar gate outputs. It does NOT say rank(J_full)<=6: J_anchor can have full rank and vary with x through the other nonlinearities.

Let J* be the exact target field derivative and R=J*-J_anchor. For any rank-at-most-r correction C, best rank-r approximation in the Frobenius norm gives

```text
||J_full-J*||_F^2 = ||C-R||_F^2
                  >= sum_{i>r} singular_value_i(R)^2.
```

The optimal unrestricted rank-r matrix retains the r leading singular components of R. This is the classical [Eckart–Young result](https://doi.org/10.1007/BF02288367), used here as an analysis tool. The network's actual attainable C can be more restricted than an arbitrary rank-r matrix, so the tail is a valid lower bound but need not be attainable.

For scalar gates r=6 (capped by the input/output dimension). Vector gates have many more outputs; in16D the same bound can be vacuous, with r=16 and zero tail. That by itself is not a performance advantage. Measure the actual correction singular values, target residual spectrum, full/anchored errors and the tail's share of the remaining full error.

If the tail explains much of the scalar model's residual, then its trained anchored map leaves a deficit that six scalar clocks cannot fully correct at that point. If the tail is small, a rank-capacity explanation is weak there; orientation, magnitude, optimization or the learned base map may matter more. Either conclusion is conditional on the learned J_anchor. Retraining could move J_anchor and invalidate a claim of irreducible architecture error. A better vector endpoint with a different base map is also insufficient to attribute all its gain to this bound.

The product16 experiment is a useful setting because its target has several simultaneous state-dependent directions. Report all five times and pointwise distributions; do not choose only a high-rank tail case. This diagnostic adds no loss, constraint or training trick. It is read only and is planned before the product16 outcome is read.
