# The constant-clock update is an ordinary FFN update in other coordinates

This read-only check extends the existing algebra to six real, trained image
states. Correctly transporting an ordinary SiLU model and its Adam state
reproduces the P model's next effective update. Maximum parameter error is
2.22e-16 and post-update prediction error 1.55e-15 in the literal float64 model.
This is an implementation check of the coordinate identity, not a new
generative result or a proof of what causes the earlier FID gap.

## What was compared

Each saved state precedes one registered update at 1k, 2.5k, 5k, 10k, 15k or
20k. Its original RNG supplies the next 64-image training batch. Three models
start at the same effective function and corresponding nonzero moments:

- P: the fixed-0.6 professor core with the native optimizer.
- Transported W: ordinary SiLU, W=I+cP, matrix rate c*eta, epsilon/c, moments
  mP/c and vP/c², and the P-coordinate weighted clipping norm.
- Native-rule W: the same direct-W state and moments, with unchanged native
  rate, epsilon and ordinary W-coordinate clipping.

The last case is a one-step counterfactual. It is not the historical matched
SiLU training trajectory. All models retain the same shared backbone state.
The [derivation](DERIVATION_PACKAGE.md) specifies the exact conditions.

## All six checks

| Update | P gradient norm | Ordinary W norm | P clip | W clip | Post-prediction error, P versus transported W |
|---|---:|---:|---:|---:|---:|
| 1000 | 1.167511 | 1.185817 | 0.856523 | 0.843300 | 8.88e-16 |
| 2500 | 0.244317 | 0.250874 | 1.000000 | 1.000000 | 1.33e-15 |
| 5000 | 0.737095 | 0.765818 | 1.000000 | 1.000000 | 1.42e-15 |
| 10000 | 0.649572 | 0.678103 | 1.000000 | 1.000000 | 1.55e-15 |
| 15000 | 0.344257 | 0.361975 | 1.000000 | 1.000000 | 1.50e-15 |
| 20000 | 0.216620 | 0.227813 | 1.000000 | 1.000000 | 1.39e-15 |

All 49 active gradient tensors match the pullback at every state. Transported
first/second moments also match, with maximum error 1.93e-17. The unmodified
direct-W matrix increments are 1.66655–1.66981 times the effective P increments
across all 36 layer/state pairs. The near-1/c ratio is measured here; it is
not an identity when epsilon or clipping differences matter.

At update 1000 both models clip, but their factors differ. At update 5000
neither clips, yet the ordinary rule still takes the larger effective matrix
step. Thus a clipping-only explanation of these one-step differences is
insufficient. Ordinary-rule post-update predictions differ from P by RMS
0.000396–0.000729 on the six batches. These local differences do not quantify
their eventual effect on generated images.

## Precision and interpretation

All six native 64-image validation prefixes replay bitwise, as do their
literal float32 paths. Promoted native errors remain below 1.36e-6. The exact
coordinate check uses the cast-free float64 extension because x+cPx and
(I+cP)x reassociate arithmetic in float32. Native float32 optimizer-trajectory
equivalence is not claimed.

The current fixed-clock core is exactly an ordinary bias-free SiLU function
class. Its training coordinates change the effective learning rate, epsilon
and clipping geometry. Consequently, its FID advantage over a default or
matched-init ordinary run cannot establish a distinct continuous-flow
computation. Conversely, this check does not prove that those optimizer
differences cause the measured FID advantage.

No new model was trained, no sampling was performed and no quality gate was
changed. The stopped SwiGLU initializer remains stopped. A next architectural
test must specify the useful operation or restriction beyond this fixed-core
identity and compare against the stronger ordinary controls.

The documented command ran once through a fresh agent, session 64675,
terminal bffec2, exit 0, with no discrepancies. It took 19.24 seconds on one
Mac mini CPU thread. All raw checkpoints, previous studies and Report1 are
unchanged. Complete measurements and the actual receipt are in [results](results/).
