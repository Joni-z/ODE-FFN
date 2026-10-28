# One ordinary SwiGLU initialization control

Zhizhe Zhang · October29,2026. Register before new training outcomes.

## Why this experiment
Cold fixed-clock training with trainable P retained native K1's image margin, and ordinary SiLU64
with the same initial function improved FID190.76to178.06. The default Swi86
reference195.31 still has a different initial FFN. This is a concrete missing
baseline control, rather than a consequence of the failed PSD priority test.
Keep matched SiLU's earlier failed allocation margin and every other stopped
setting unchanged. A positive result is a stronger ordinary baseline, not an
exclusive ODE mechanism or a reason to reopen a failed family.

## Exactly one new arm
Use native Swi86, actual hidden86, unchanged class/parameters/forward. Begin
from its original seed0 cold state, retaining every shared tensor/buffer and
all default extra gate/value rows. For each FFN, obtain W1,W2 from the original
cold matched-SiLU64 checkpoint. With native gate/value/output matrices G,V,O:

```text
G[:64]=W1; gate_bias[:64]=0
V[:64]=0;  value_bias[:64]=1
O[:,:64]=W2; O[:,64:]=0; output_bias=0
```

Leave the22extra gate/value channels at the stored default Swi86 values.
All parameters remain trainable. The initial FFN is exactly W2*SiLU(W1*h)
in real arithmetic, with matching input derivative and shared-backbone
derivatives. It is not parameter-tangent matched: the main value branch,
gate biases and extra output columns supply additional directions. Extra
gate/value gradients initially vanish at zero extra readout; the extra readout
gradient must be nonzero and open these paths after an update. No random
initializer, channel count or bias value is searched.

## Fixed training and retained comparisons
One new cold20,000-update trajectory, seed0, original stream100000, unchanged
45k/5k CIFAR split, batch64, clean-image JiT prediction and capped weighted
loss, logit-normal times, AdamW1e-4/(.9,.99)/wd0/clip1, constantLR, no EMA/AMP.
Use the original native draw/update/optimizer/checkpoint/restoration functions.
Keep original architecture, patch4/width64/depth3, labels fixed0 and no dropout.

Snapshots0/1000/2500/5000/10000/15000/20000, including complete optimizer/RNG/
buffers/stream and the immediately preceding state for one-update audit.
Validation and all five denoising panels use unchanged banks. Record native
gate/value/output matrices, singular values, extra-readout and value adaptation.
These Swi matrices are not relabeled as physical P. All original P histories
remain available in their sealed baseline runs.

Retain the complete32-row cold-control table, including native K1/K2,
Swi86/129, SiLU64/131/193, fixed06, matched SiLU64 and frozen interventions.
These are reused outcomes, not new trajectories. Recheck relevant native
predictions/features before use. New/default Swi86 have exactly the same
211412trainable values and16512FFN forward MACs/token/block; measure rather
than assume any whole-model speed change.

## Quality measurement and decision
Generate unchanged1024Gaussian starts with native CPU Heun128 at5k,10k,20k.
All are reported;20k alone is primary. Use the same2048real references,
uint8 conversion, MPS Inception weights/features and float64 empirical FID.
No test data, checkpoint selection, clipping sampler states or recalibration.

Support for this initialization control requires final FID<=.95times default
Swi86's20kFID, with weighted/clean validation MSE no more than5%worse than
that same default reference. Report continuous comparisons against native K1,
fixed06, matched SiLU and all stronger ordinary references regardless of this
gate. If it succeeds, subsequent ODE claims must face the strengthened baseline.
If it fails, stop this initializer without extra-channel, value-bias, learning-
rate, width, duration or checkpoint rescue. No new ODE method is declared.

## Readiness, audit and resources
Before formal launch, replay default/native baselines and verify exact shared
cold tensors/buffers, literal initial FFN/input derivatives, nonzero-head/
active-backbone shared gradients, all additional routes, and physical five-
update resume. Probe backbones may be active20k references, but formal weights
always rebuild cold; no readiness update is a warm start. Check actual costs.
Use an isolated fresh agent to execute both documented native and metric
preflights verbatim; no source fixes/retries inside those passes.

Automatic audit: reconstruct the complete20kinput stream; replay all seven
checkpoint predictions/panels and matrix summaries; reproduce every saved
last-update boundary with native Adam/RNG/buffers bitwise; independently
replay all three Heun image banks, MPS features and FID arithmetic. Preserve
all formal hashes and actual tool exits before conclusions.

Mac mini, one one-thread CPU training/audit child. Existing metrics environment
uses MPS only for Inception extraction. No new packages or GPU reservation.
Monitor primed CPU/RSS every5seconds and progress every100updates. Training
ceiling7200s, training audit1200s, sampling/audit2700each, metric/audit1200each;
expected new storage below500MB. Poll existing live handles; preserve failures.
Original source/data/checkpoints/Report1 remain unchanged.
