# A mechanism question before another shared-field design

Post-result note, not a selected architecture or another training run. The autonomous shared factor is stopped. Its paired-clone potential has6.82x lower ODE field error, despite the same initial functions and executed block work. Clipping does not explain that endpoint difference on its own.

At any shared weight state, make three independent copies of the block with the same values. These copies define exactly the same complete predictor. If g1,g2,g3 are their per-visit gradients, then the shared-block gradient is g1+g2+g3, and

```text
||g_shared||² = sum_i ||g_i||² + 2*sum_(i<j) dot(g_i,g_j).
```

The existing nonzero-head tests verify gradient addition at the matched initial state. A distinct future frozen diagnostic could examine whether cross-depth gradients at actual learned shared states repeatedly oppose each other, using matched cloned copies and the same complete FM loss. That would distinguish a concrete interference observation from simply inferring failure from parameter count. It would still not prove causality, an approximation barrier, an Adam convergence result or that an internal-depth conditioner will work. No conditioner or optimizer rescue is selected by this note.
