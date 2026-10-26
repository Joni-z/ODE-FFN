# Saved-head output-envelope diagnostic

Fixed before inspecting the new envelope results. This read-only check uses the
same Stage17 vector parent, SwiGLU96 and SwiGLU128 final states, Ring8/Product16,
five common times and all512anchors. It selects no architecture or training.

For the current final LayerNorm of width w, its normalized vector z has
`sum(z)=0` and `||z||<=sqrt(w)`. With learned affine parameters gamma/beta and
output matrix W/bias b, every prediction lies in the ellipsoid

```text
E = {c + B u : ||u||<=1}
c = W beta + b
B = sqrt(w) W diag(gamma) [I - 11^T/w].
```

This is an outer envelope: the backbone need not realize every interior point.
The trained output can be bounded even when the exact Gaussian-mixture velocity
has linear tails. However, that mathematical class distinction is relevant to
the measured error only if the actual target fields materially leave E.

For every existing oracle velocity, compute its squared Euclidean distance to
E using the singular axes and the scalar secular equation. Check every saved
model prediction is inside E within floating-point tolerance. Compare the
mean envelope-distance floor with the actual squared velocity error, dividing
both by data dimension to match the recorded field-MSE units. Report every
time panel and all-time aggregates, with target outside fractions and minimum
ellipsoid semiaxis. Reproduce each native field MSE and bind all checkpoint,
array, source-metadata, protocol and script hashes.

The floor is conditional on each saved output head. Retraining can change its
size and orientation without changing architecture; it is not a positive
architecture-wide approximation bound. A negligible floor weakens the specific
claim that output-range saturation explains today's gap. A large floor would
still not prove that a different prediction head or ODE block will train better.
No labels, data, model state, loss or previous frozen runtime is modified.
