# Does sharing help, or does its coordinate system help?

The shared-generator sampling advantage survives the tested solver refinements.
The released-stage comparison also changes optimizer coordinates. This control
separates that change from the stationary-generator constraint. It does not
introduce another response function.

## What the current release changes

The implemented release uses four stage matrices:

```text
W0 = P,   Wj = P + Delta_j, j=1,2,3.
```

Write Gj for the loss gradient with respect to the effective stage matrix Wj.
Under SGD without momentum or decay, one step gives:

```text
P_new       = P - eta * (G0 + G1 + G2 + G3)
Delta_j_new = Delta_j - eta * Gj
mean(W)_new = mean(W) - eta * [sum_j Gj + (G1+G2+G3)/4].
```

Thus the effective mean receives an extra update when the offsets are released.
The common parameter P has the same initial gradient as the shared model,
but the mean of the four effective matrices does not have the same update.
For AdamW or Muon the linear SGD formula is not the actual optimizer update;
the extra mean-offset contribution still exists algebraically. Current results
therefore do not isolate the restriction to one stationary generator from every
optimizer-coordinate effect.

## A mean-preserving release

Keep P and replace the three anchored offsets with three contrasts Bk:

```text
Wj = P + sum_k C[j,k] * Bk

C = 1/2 * [[ 1,  1,  1],
           [ 1, -1, -1],
           [-1,  1, -1],
           [-1, -1,  1]].
```

The columns of C are orthonormal and sum to zero. Consequently:

```text
mean_j Wj = P                       (for every possible optimizer state)
gradient_P = sum_j Gj
gradient_Bk = sum_j C[j,k] * Gj
```

Initialize all Bk to zero without consuming random draws. The initial network,
input gradients and common parameter gradients then match the shared model.
The first update of P matches when the same common gradient reaches an
identical optimizer state. The native global clipping norm also includes
the new contrasts, so it can change the gradient scale before that step.
Measure both models' native clip factors and their actual first updates;
do not claim that clipping or whole optimization trajectories are identical.
Later gradients also change as the networks learn different functions.

This release has exactly the same effective function class and parameter count
as the current anchored release. Any four matrices are represented uniquely by
P=mean(W) and Bk=sum_j C[j,k]Wj. It is a coordinate control, not extra capacity.
Both releases use four midpoint steps and eight effective-matrix applications.
Report matrix-materialization overhead and measured runtime separately.

## Registered comparison

Train only the new mean-preserving release on Ring8 and Product16, using the
existing shared and anchored-stage trajectories as references after matching
their initial state, recipe, training-data stream and checkpoint provenance.
Keep seed0, width64/depth3, ISRU response, AdamW0.0003,20k updates and the
existing four-stage integration. Preserve native loss, sampler and metrics.
Reuse the prebound eight4096-sample sharing banks for every model; no new
validation selection or response search is needed.

Before training, verify the orthogonal identities, forward/input/common-gradient
identity at zero contrasts, effective-matrix mean after an actual optimizer
step, all parameters active, and native8-to16 checkpoint continuation.
Independent fresh documented execution must pass before launch.

Primary question: does the shared model retain its SW2 advantage over this
same-class, mean-preserving release? Retain both ordinary controls. Secondary
checks are oracle field MSE, actual stage drift, computation and solver
sensitivity. Keep the full density comparison as a separate mechanism probe.
For the two new checkpoints, also reuse the existing density-source banks
and numerical qualification rules to measure joint KL and Product16 component
widths. Reuse the sixteen completed reference cells; do not rerun them. This
secondary measurement is specified before training because the completed
reference study already showed opposite SW2 and KL rankings.

If the new release closes the gap, revise the explanation toward optimizer
coordinates rather than attributing the old gap to sharing alone. If sharing
still wins, the positive constraint result survives this particular mean-update
control. Neither outcome makes finite-step ODE unrolling exclusive to ODEs:
the same computation can also be written as a recurrent network.

Status: mathematically prepared; not implemented or launched.
