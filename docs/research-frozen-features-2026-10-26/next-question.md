# Where might the response be lost?

This is a post-result question, not a nominated architecture or a selected next run. The frozen final/pre-normalization features did not expose a useful linear repair. That leaves open whether an earlier nonlinear representation contains the relevant response and fails to transmit it, or never learns it sufficiently.

For one original scalar K2 layer,

```text
L(h) = h+a(h)Ph+.5*a(h)^2*P^2*h
J_L = I+aP+.5*a^2*P^2 + (Ph+aP^2h)*grad(a)^T.
```

The clock-dependent part of this local input Jacobian has rank at most1. The controller's first affine/SiLU hidden layer is then compressed to a single logit before it influences the map. This is an exact local identity, not a capacity lower bound: the middle SiLU supplies64 coordinate nonlinearities, different layers compose, and the full learned representation can change. Earlier vector-clock models already failed to yield practical superiority, so simply increasing gate count is not justified by this identity.

A distinct bounded diagnosis could compare equal-width probes of existing controller hidden SiLU outputs, the original middle SiLU and successive residual states, with the same held-out response/field targets. It would need its own fixed source/sites, ordinary controls, rank accounting and a protocol before outcomes. If a response is accessible upstream but unavailable downstream, that supports studying transmission/compression. If upstream probes also fail, a local readout change is not supported and feature learning itself remains the question. Richness of a concatenated feature bank alone would not justify extra architecture width or cost.

No controller bypass, new loss, temperature, P constraint, extra state, activation or revival of a stopped branch is selected from this note. A later design still needs a mechanism and a cold quality/cost comparison against strong SwiGLU.
