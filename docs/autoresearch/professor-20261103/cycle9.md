# Cycle9 registration: reconcile P norms with actual layer responses

This is a frozen diagnostic after cycle8's complete negative training result.
No new training, whitening, loss, optimizer, sampler or candidate selection.
The earlier input/projection-geometry study already found anisotropic features
in both ODE and SwiGLU and rejected a P-only whitening interpretation. Preserve
that decision. The narrower new question is how the recently trained constraint
arms changed their features and finite-polynomial responses.

## Measured motivation and fixed cases

In Product16 scalar K2, shape relaxation lowers mean P norm19.17->16.20 while
first-term/input RMS increases3.88->4.47. In Product16 raw K1 the same response
ratio is nearly unchanged, whereas on Ring8 it decreases strongly. These are
normalized hidden responses, not necessarily changes in absolute output energy.

Use all twelve cycle8 final20k checkpoints, plus the two previously fixed
raw-K1 final80k checkpoints from cycle6. Fourteen cases, seven per host, six
ODE layers each. Hash-bind the published replayed checkpoints/configs/manifests.
No new family is chosen from this diagnostic. Ordinary controls remain necessary
for any future claim of an ODE-specific optimization benefit, which is outside
this descriptive question.

Use exactly the existing native256-point hidden probe: copy the same probe.npz
loading lines as fixed_eval_data, and retain its actual time values/hash. This
is a finite mixed-time bank, not a new population estimator or an independence
claim about the layer records. Record the time range and distinct count.

## Exact identities and readouts

For each layer capture its actual incoming h, duration d, and outgoing tensor
with hooks that return nothing. Verify the complete predictor is bitwise
unchanged when the hooks are removed. For the scalar duration or PD raw layer,
let q=d*h and r1=Pq. Compute float64 moments on these captured native features:

```
H = E||h||^2
M = E[q q^T]
S = P M P^T
E||r1||^2 = tr(S)
first-term/input RMS = sqrt(tr(S)/H).
```

Record raw P singular values, feature and increment singular energies,
uncentered mean fractions, input energy and both absolute/normalized increment
energy. Do not invert M or interpret structural/null directions as training
failures. Compute energy spectra directly from q/sqrt(N) and r1/sqrt(N) to
avoid spurious negative eigenvalues from squaring ill-conditioned matrices.
Check every energy identity against direct sample sums and archived native
first-term amplitudes. A fixed admissible algebraic C rescaling verifies the
PMP^T identity without changing the actual model.

For scalar K2 also reconstruct the second term and full finite-layer output:

```
r2 = .5 d P r1
E||r1+r2||^2 = E||r1||^2 + E||r2||^2 + 2E<r1,r2>.
```

Raw K1 has r2=0. Record term alignment and cancellation, with native-output
reconstruction errors scaled by the largest component RMS. The first-term
moment is not passed off as a description of the entire K2 polynomial.

For every matched native-versus-shape/scale pair, and raw20k-versus80k pair,
compare per-layer normalized energies using B=P^T P and N=M/H. The symmetric
algebraic decomposition is

```
T1-T0 = tr[(B1-B0)(N1+N0)/2] + tr[(B1+B0)(N1-N0)/2],
T = tr(B N).
```

The two terms describe parameter and feature changes in the recorded coordinates.
They are order-symmetrized accounting, not independent causes, loss-relevant
curvature, invariant causal attributions or predictions of generalization.
Hidden coordinates and both factors adapt during training.

## Verification and decision boundary

First check the matrix/decomposition formulas and actual native layer outputs
on seeded models; then freeze/publish/deploy and pass fresh documented CUDA
witnesses on145/Lenovo. Formal analysis preserves all14cases and raw sufficient
arrays on each host. Independent NumPy replay checks moments, energies, spectra,
pair decomposition and saved hashes without another model execution.

This can explain the recent response statistics. It cannot establish a new
useful FFN, a global stability bound or an optimizer-convergence bottleneck.
No whitening/inverse or additional training is selected merely because a
moment is anisotropic or a cancellation is large.

- [ ] Implement/source-bind all14cases and pass local algebra/native witnesses.
- [ ] Publish/deploy and obtain fresh documented CUDA witnesses.
- [ ] Collect all84layer records and ten matched pairs without parameter updates.
- [ ] Independently replay saved sufficient arrays; publish the constrained
      interpretation and choose the next question from actual evidence.
