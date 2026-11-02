# Cycle7 registration: singular-value shape versus contraction

This is a frozen-checkpoint diagnostic, registered after the complete cycle6
comparison and before executing any intervention. It directly tests the
professor's suggestion to make P's singular values more even. It does not
train a new architecture, change the objective, or establish training efficacy.

## Question and mathematical contrast

Muon reshapes updates without enforcing a flat spectrum of P. Earlier decay
and norm-cap controls conflate weight amplitude with spectral shape. Can a
small change of singular-value shape preserve or improve a learned function
when a scalar contraction with the same largest singular value cannot?

For each frozen P = U diag(sigma) V^T of rank dimension r, set

```
alpha = ||sigma||2 / sqrt(r)
b(k) = (1-k) sigma + k alpha 1
t(k) = ||sigma||2 b(k) / ||b(k)||2
P_shape(k) = U diag(t(k)) V^T
P_scale(k) = [max(t(k))/max(sigma)] P
```

At k=0 use an exact tensor copy, not an SVD reconstruction. The zero matrix
stays zero. Use k = .02, .1, .5, 1. For each nonzero setting, preserve all
singular vectors and Frobenius norm on the shape branch. The scale branch
matches the resulting leading singular value separately for every P.
Both branches act on all six FFN P matrices simultaneously. Other tensors,
buffers, scalar gates or direct input diagonals remain frozen.

The singular-value ratios contract: if a >= b >= 0, the ratio
[(1-k)a+k alpha]/[(1-k)b+k alpha] decreases with k. Normalization does not
change ratios, so k=1 is a scaled orthogonal matrix. The leading singular value
cannot increase: with q=||sigma||2 and m=max(sigma)/q >= 1/sqrt(r),
the normalized blend's leading coordinate is at most m, since the squared
difference after multiplying by the blend norm equals a sum of nonnegative
terms. Explicitly put a=1-k, b=k, c=1/sqrt(r), L=sum(sigma)/q. Then
`m^2 ||a sigma/q + b c 1||^2 - (a m+b c)^2
= b^2(m^2-c^2)+2abmc(mL-1) >= 0`, since m>=c and mL>=1.
The implementation checks this numerically;
it does not assume eigenvalues or nonnormality stay fixed.

Shape and scale branches intentionally differ in Frobenius energy. Their common
operator norm removes a simple maximum-amplification difference, not all
functional confounds. Singular vectors of PD, hidden activations, cancellations,
and the complete field can change. Rank-deficient P uses the SVD's orthogonal
completion at k>0; record its minimum singular value rather than assuming a
unique polar factor. This is not a coordinate-equivalent reparameterization.

## Six fixed checkpoints, 54 interventions

For each task use exactly three previously fixed checkpoints: original scalar
K2 at20k (Ring8 Muon .006; Product16 Muon .02), raw K1 at20k, and its exact-prefix
80k child (Ring8 Muon .02; Product16 Muon .006). No selection from the new
diagnostic. Each checkpoint has one untouched anchor, four shape interventions,
and four paired scalar contractions: nine cases, 54 total across both tasks.

Use the original field-validation bank and separate test sampling bank, seed0,
4,096 samples, 256 directions and Heun32. Record all original metrics, field
prediction displacement, complete matrix spectra and numerical constraint
errors. Also retain the existing test target-versus-target reference distance;
it is a finite-sample reference, not a lower bound. No checkpoint selection,
new loss, parameter update, new sampler, extra training seed or image claim.
The untouched case must reproduce the archived scores and arrays before the
remaining cases are interpreted. Explicitly record any nonfinite output.

The outcome can justify a separate training experiment if a local shape
intervention improves field fitting and behaves better than contraction.
A negative frozen intervention does not disprove a constraint trained from
initialization; neither result alone accepts an FFN replacement. Keep all
negative cases and ordinary controls from cycle6 visible.

## Execution checklist

- [ ] Implement and verify the matrix identities and exact k=0 native function.
- [ ] Publish immutable source and deploy new worktrees on145/Lenovo.
- [ ] Fresh agent executes documented CUDA self-test on both environments.
- [ ] Run all54 frozen cases, preserve bindings and monitor processes/resources.
- [ ] Independently verify anchors, constraints and complete outcomes; publish
      the result before selecting any constrained-training design.
